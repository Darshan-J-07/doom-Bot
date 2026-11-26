"""Bound timed campaign jobs and publish progress without changing the learner."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time

import psutil


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def elapsed_utc(clock):
    return (datetime.now(timezone.utc) - datetime.fromisoformat(clock["start_utc"])).total_seconds()


@contextmanager
def locked_status(root):
    with (root / ".status.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read(root / "status.json")
        yield state
        clock = read(root / "clock.json")
        time_now = time.monotonic()
        utc_seconds = elapsed_utc(clock)
        live = min(clock["deadline_monotonic"] - time_now, clock["budget_seconds"] - utc_seconds)
        train = min(clock["training_cutoff_monotonic"] - time_now,
                    clock["budget_seconds"] - clock["reserved_final_seconds"] - utc_seconds)
        state.update(elapsed_seconds=round(time_now - clock["start_monotonic"], 2),
                     elapsed_utc_seconds=round(utc_seconds, 2), remaining_seconds=round(max(0., live), 2),
                     training_time_available_seconds=round(max(0., train), 2),
                     last_update_utc=now_utc())
        completed = state.get("jobs", {}).values()
        credited = [row for row in completed if row.get("kind") == "train"]
        state["added_decisions"] = sum(row.get("added_decisions", 0) for row in credited)
        state["ppo_updates"] = sum(row.get("added_updates", 0) for row in credited)
        write(root / "status.json", state)


def record(root, *, message=None, job=None, status=None):
    with locked_status(root) as state:
        if status:
            state.update(status)
        if job:
            state.setdefault("jobs", {})[job["label"]] = job
            active = [row for row in state["jobs"].values() if row["status"] == "running"]
            state["active_jobs"] = active
            training = next((item for item in active if item["kind"] == "train"), None)
            current = training or (active[-1] if active else job)
            state.update(phase=current["phase"], current_experiment=current["label"],
                         phase_status=current["status"],
                         active_child=current["pid"] if active else None,
                         active_checkpoint=current.get("active_checkpoint"),
                         next_evaluation_milestone=current.get("milestone"),
                         resources=current.get("resources"))
        if message:
            with (root / "progress.log").open("a") as log:
                log.write(f"{now_utc()} {message}\n")


def resources(process, root):
    m = psutil.virtual_memory()
    usage = psutil.disk_usage(root)
    processes = []
    try:
        processes = [process, *process.children(recursive=True)]
    except psutil.NoSuchProcess:
        pass
    rss = 0
    for child in processes:
        try:
            rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    result = dict(available_ram_mib=round(m.available / 2**20, 1),
                  tree_rss_mib=round(rss / 2**20, 1), disk_free_gib=round(usage.free / 2**30, 2),
                  cpu_load_1min=round(os.getloadavg()[0], 2))
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"], timeout=2, text=True, stderr=subprocess.DEVNULL)
        vram, temperature, utilization = [float(x.strip()) for x in output.splitlines()[0].split(",")]
        result.update(gpu_used_mib=vram, gpu_temperature_c=temperature, gpu_utilization_pct=utilization)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return result


def tail_row(path):
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        size = stream.seek(0, os.SEEK_END)
        stream.seek(max(0, size - 131072))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except ValueError:
            pass
    return None


def training_progress(row, directory, start_steps, start_updates):
    latest = tail_row(directory / "learning.jsonl")
    if latest is None:
        return False
    row["added_decisions"] = max(0, latest["steps"] - start_steps)
    row["added_updates"] = max(0, latest["updates"] - start_updates)
    row["cumulative_decisions"] = latest["steps"]
    fields = ("decisions_per_second", "learning_rate", "entropy", "approx_kl", "policy_loss", "value_loss",
              "clip_fraction", "explained_variance", "grad_norm", "loss")
    metrics = {k: latest[k] for k in fields if isinstance(latest.get(k), (float, int))}
    nonfinite = [k for k, value in metrics.items() if not math.isfinite(value)]
    row.update({k: v for k, v in metrics.items() if k not in nonfinite})
    if nonfinite:
        row["nonfinite_training_metrics"] = nonfinite
    if (directory / "latest.pt").is_file():
        row["active_checkpoint"] = str(directory / "latest.pt")
    episodes = latest.get("episodes", [])
    rewards = [ep["shaped_return"] for ep in episodes if "shaped_return" in ep]
    if not rewards:
        rewards = [ep["reward"] for ep in episodes if "reward" in ep]
    if rewards and all(math.isfinite(v) for v in rewards):
        row["latest_batch_mean_episode_reward"] = sum(rewards) / len(rewards)
    if latest.get("invalid_actions", 0):
        row["invalid_training_actions"] = latest["invalid_actions"]
    return True


def supervise(args):
    root = args.out.resolve(strict=True)
    clock = read(root / "clock.json")
    current_boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if clock["boot_id"] != current_boot:
        raise SystemExit("The campaign monotonic clock belongs to a previous boot.")
    stamp = time.monotonic()
    cutoff = clock["training_cutoff_monotonic"] if args.kind == "train" else clock["deadline_monotonic"]
    utc_budget = clock["budget_seconds"] - (clock["reserved_final_seconds"] if args.kind == "train" else 0)
    if args.seconds + 35 > min(cutoff - stamp, utc_budget - elapsed_utc(clock)):
        raise SystemExit("Job cap plus shutdown allowance exceeds the campaign time remaining.")
    if (root / "STOP").exists():
        raise SystemExit(f"Campaign stopped by {root / 'STOP'}")
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("Supply a command after --.")
    logs = root / "jobs"
    logs.mkdir(exist_ok=True)
    with (logs / f"{args.label}.log").open("x") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)
        row = dict(label=args.label, phase=args.phase, kind=args.kind, status="running",
                   pid=proc.pid, command=command, started_utc=now_utc(), started_monotonic=stamp,
                   log=str(Path(log.name)), milestone=args.milestone, added_decisions=0, added_updates=0)
        reason = None
        stop_at = None
        last_progress = stamp
        last_decisions = -1
        next_log = stamp + 60
        interrupted = {}

        def interrupt(sig, _frame):
            interrupted["reason"] = signal.Signals(sig).name

        original = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
        def stop(sig, *, group=False):
            try:
                if group:
                    os.killpg(proc.pid, sig)
                else:
                    proc.send_signal(sig)
            except ProcessLookupError:
                pass

        try:
            record(root, job=row, message=f"{args.phase} started {args.label} (PID {proc.pid}).")
            child = psutil.Process(proc.pid)
            while proc.poll() is None:
                tick = time.monotonic()
                if args.training_dir:
                    training_progress(row, args.training_dir, args.start_steps, args.start_updates)
                if row["added_decisions"] != last_decisions:
                    last_progress, last_decisions = tick, row["added_decisions"]
                row["resources"] = resources(child, root)
                row["peak_tree_rss_mib"] = max(row.get("peak_tree_rss_mib", 0),
                                               row["resources"]["tree_rss_mib"])
                if stop_at is None:
                    if interrupted or (root / "STOP").exists():
                        reason = interrupted.get("reason", "STOP file")
                    elif tick >= cutoff - 30 or elapsed_utc(clock) >= utc_budget - 30:
                        reason = "campaign deadline"
                    elif tick >= stamp + args.seconds:
                        reason = "job wall-clock cap"
                    elif row.get("invalid_training_actions") or row.get("nonfinite_training_metrics"):
                        reason = "invalid action or nonfinite learning metric"
                    elif args.stall_seconds and tick - last_progress > args.stall_seconds:
                        reason = "training stopped advancing"
                    elif (row["resources"]["available_ram_mib"] < args.min_ram_mib
                          or row["resources"]["disk_free_gib"] < 1
                          or row["resources"].get("gpu_temperature_c", 0) >= 89):
                        reason = "resource floor or GPU thermal limit"
                    if reason:
                        record(root, message=f"{args.label} stopping for {reason}.")
                        stop(signal.SIGTERM)
                        stop_at = tick
                elif tick >= stop_at + 25:
                    stop(signal.SIGKILL, group=True)
                row["wall_seconds"] = round(tick - stamp, 2)
                announcement = None
                if tick >= next_log:
                    announcement = (f"{args.label} is active after {row['wall_seconds']:.0f} s, "
                                    f"{row['added_decisions']} decisions, {row['added_updates']} PPO updates.")
                    next_log = tick + 60
                record(root, job=row, message=announcement)
                time.sleep(3)
        finally:
            if proc.poll() is None:
                stop(signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    stop(signal.SIGKILL, group=True)
                    proc.wait()
            # Spawned engine workers share this isolated job process group.
            stop(signal.SIGTERM, group=True)
            for sig, previous in original.items():
                signal.signal(sig, previous)
            if args.training_dir:
                training_progress(row, args.training_dir, args.start_steps, args.start_updates)
            code = proc.returncode
            row.update(status="stopped" if reason else "completed" if code == 0 else "failed",
                       stop_reason=reason, return_code=code, finished_utc=now_utc(),
                       wall_seconds=round(time.monotonic() - stamp, 2))
            record(root, job=row, message=f"{args.label} {row['status']} (exit {code}, "
                   f"{row['added_decisions']} decisions, {row['wall_seconds']:.1f} s).")
            write(logs / f"{args.label}.json", row)
    return code if code >= 0 else 128 - code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    run = subs.add_parser("run")
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--phase", required=True)
    run.add_argument("--kind", choices=("train", "evaluate", "profile", "test", "data", "audit"), required=True)
    run.add_argument("--label", required=True)
    run.add_argument("--seconds", required=True, type=float)
    run.add_argument("--training-dir", type=Path)
    run.add_argument("--start-steps", type=int, default=314496)
    run.add_argument("--start-updates", type=int, default=1002)
    run.add_argument("--stall-seconds", type=float, default=0)
    run.add_argument("--min-ram-mib", type=int, default=800)
    run.add_argument("--milestone", type=int)
    run.add_argument("command", nargs=argparse.REMAINDER)
    note = subs.add_parser("note")
    note.add_argument("--out", required=True, type=Path)
    note.add_argument("--status-json", type=Path)
    note.add_argument("message")
    args = parser.parse_args()
    if args.mode == "note":
        record(args.out, message=args.message, status=read(args.status_json) if args.status_json else {})
        return 0
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+", args.label) or not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("A simple job label and positive finite time cap are required.")
    return supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
