"""Credit active PPO time only when the corresponding optimizer update is saved."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
import time
from unittest.mock import patch


METADATA_KEY = "_learning_meter"
PROGRESS_FILE = "training-progress.json"
NS_PER_SECOND = 1_000_000_000


def write_json(path, value):
    stream = NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}-",
                                suffix=".tmp", delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            json.dump(value, stream, allow_nan=False, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class ActiveCycle:
    """Collect monotonic intervals for one rollout and its PPO optimization."""

    started_ns: int | None = None
    excluded_ns: int = 0
    pending: tuple[int, int] | None = None

    def begin(self):
        if self.started_ns is None:
            self.started_ns = time.monotonic_ns()
            self.excluded_ns = 0

    def excluding(self, call, *args, **kwargs):
        if self.started_ns is None:
            return call(*args, **kwargs)
        start = time.monotonic_ns()
        try:
            return call(*args, **kwargs)
        finally:
            self.excluded_ns += time.monotonic_ns() - start

    def optimize(self, call, *args, **kwargs):
        if self.started_ns is None or self.pending is not None:
            raise RuntimeError("A timed PPO update requires exactly one unsaved training cycle")
        start = time.monotonic_ns()
        rollout_ns = max(0, start - self.started_ns - self.excluded_ns)
        result = call(*args, **kwargs)
        self.pending = (rollout_ns, time.monotonic_ns() - start)
        self.started_ns = None
        return result


def _paths(argv):
    parser = argparse.ArgumentParser(add_help=False)
    for name in ("config", "out", "resume", "stop-file"):
        parser.add_argument(f"--{name}", type=Path)
    known, _ = parser.parse_known_args(argv)
    config = json.loads(known.config.read_text()) if known.config else {}
    output = known.out or Path(config.get("out", "runs/gru"))
    resume = known.resume or config.get("resume")
    stop_file = known.stop_file or config.get("stop_file")
    return output, Path(resume) if resume else None, Path(stop_file) if stop_file else None


def _initial_state(meter_id, saved):
    steps, updates = saved.get("trained_steps", 0), saved.get("updates", 0)
    state = saved.get("args", {}).get(METADATA_KEY)
    if state and state["meter_id"] == meter_id:
        if (state["checkpoint_steps"], state["checkpoint_updates"]) != (steps, updates):
            raise ValueError("The saved learning meter disagrees with its checkpoint counters")
        return dict(state)
    return dict(meter_id=meter_id, start_steps=steps, start_updates=updates, checkpoint_steps=steps,
                checkpoint_updates=updates, active_rollout_ns=0, active_optimization_ns=0)


def _active_ns(state):
    return state["active_rollout_ns"] + state["active_optimization_ns"]


def _receipt(state, checkpoint, starting_ns, active_limit_seconds, status):
    total = _active_ns(state)
    seconds = total / NS_PER_SECOND
    decisions = state["checkpoint_steps"] - state["start_steps"]
    return dict(state, status=status, checkpoint=str(checkpoint.resolve()) if checkpoint else None,
                active_training_ns=total, active_training_seconds=seconds,
                invocation_active_training_seconds=(total - starting_ns) / NS_PER_SECOND,
                active_limit_seconds=active_limit_seconds,
                added_decisions=decisions, active_decisions_per_second=decisions / seconds if seconds else None,
                added_updates=state["checkpoint_updates"] - state["start_updates"])


def run(argv, *, meter_id, active_limit_seconds, select_training_map=None, course_reward=None):
    """Run the pinned trainer with a separate, recoverable active-time budget.

    Reusing a meter ID continues its total from the exact resume checkpoint. A
    new ID starts a new ledger even when restoring model and optimizer state.
    """
    from training import train

    if not isinstance(meter_id, str) or not meter_id.strip():
        raise ValueError("Set a nonempty meter ID for this training block")
    if not math.isfinite(active_limit_seconds) or active_limit_seconds <= 0:
        raise ValueError("Active training limit must be positive and finite")
    limit_ns = round(active_limit_seconds * NS_PER_SECOND)
    argv = list(map(str, argv))
    output, resume, manual_stop = _paths(argv)
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "latest.pt"
    saved = train.torch.load(resume, map_location="cpu", weights_only=True) if resume else {}
    state = _initial_state(meter_id, saved)
    del saved
    starting_ns = _active_ns(state)
    checkpoint = resume
    progress_path = output / PROGRESS_FILE
    summary_path = output / "summary.json"
    cycle = ActiveCycle()
    original_batch, original_update = train.tensor_batch, train.update
    original_save, playback_type = train.save_checkpoint, train.Playback

    def publish(status):
        row = _receipt(state, checkpoint, starting_ns, active_limit_seconds, status)
        write_json(progress_path, row)
        return row

    if starting_ns >= limit_ns:
        row = publish("completed")
        summary = dict(steps=state["checkpoint_steps"], updates=state["checkpoint_updates"],
                       invocation_steps=0, invocation_updates=0, stop_reason="active-budget", **row)
        write_json(summary_path, summary)
        return summary

    class MeteredPlayback(playback_type):
        def start(self, *args, **kwargs):
            return cycle.excluding(super().start, *args, **kwargs)

        def wait(self, *args, **kwargs):
            return cycle.excluding(super().wait, *args, **kwargs)

        def step(self, *args, **kwargs):
            return cycle.excluding(super().step, *args, **kwargs)

    def tensor_batch(*args, **kwargs):
        cycle.begin()
        return original_batch(*args, **kwargs)

    def update(*args, **kwargs):
        return cycle.optimize(original_update, *args, **kwargs)

    with TemporaryDirectory(prefix=".active-meter-", dir=output) as work:
        stop = Path(work) / "stop"
        if manual_stop and manual_stop.exists():
            stop.touch()

        def save(path, model, optimizer, steps, updates, args, rng, **kwargs):
            nonlocal state, checkpoint
            current = Path(path) == latest
            snapshot = state
            if current:
                if cycle.pending is None or updates != state["checkpoint_updates"] + 1 or steps <= state["checkpoint_steps"]:
                    raise RuntimeError("A learning-time commit requires a new PPO update and rollout")
                rollout_ns, optimization_ns = cycle.pending
                snapshot = dict(state, checkpoint_steps=steps, checkpoint_updates=updates,
                                active_rollout_ns=state["active_rollout_ns"] + rollout_ns,
                                active_optimization_ns=state["active_optimization_ns"] + optimization_ns)
            metered_args = argparse.Namespace(**{**vars(args), "stop_file": manual_stop,
                                               METADATA_KEY: snapshot})
            original_save(path, model, optimizer, steps, updates, metered_args, rng, **kwargs)
            if current:
                state, checkpoint, cycle.pending = snapshot, latest, None
                row = publish("running")
                with (output / "active-learning.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                if _active_ns(state) >= limit_ns or (manual_stop and manual_stop.exists()):
                    stop.touch()

        publish("running")
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(train, "tensor_batch", tensor_batch))
                stack.enter_context(patch.object(train, "update", update))
                stack.enter_context(patch.object(train, "save_checkpoint", save))
                stack.enter_context(patch.object(train, "Playback", MeteredPlayback))
                train.main([*argv, "--stop-file", str(stop)], select_training_map=select_training_map,
                           course_reward=course_reward)
        except BaseException:
            publish("interrupted")
            raise

    summary = json.loads(summary_path.read_text())
    summary.update(publish("completed"))
    if _active_ns(state) >= limit_ns:
        summary["stop_reason"] = "active-budget"
    write_json(summary_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meter-id", required=True)
    parser.add_argument("--active-seconds", required=True, type=float)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.trainer_args
    if argv and argv[0] == "--":
        argv = argv[1:]
    run(argv, meter_id=args.meter_id, active_limit_seconds=args.active_seconds)


if __name__ == "__main__":
    main()
