"""Watch genuine frozen-checkpoint gameplay while a headless PPO job runs."""
from __future__ import annotations

import argparse
from hashlib import file_digest
from itertools import count, cycle, islice
import json
import os
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory

from scripts.campaign_eval import TIER, exported_agent, play, read_rows
from scripts.exit_switch_eval import load_rows
from scripts.export_candidate import export
from training.visualization import Playback, add_playback_arguments, validate_playback_arguments


DEFAULT_RUN = Path("runs/adaptive-campaign-20261127")
DEVELOPMENT = Path("runs/exit-recovery-study/maps/manifest.json")
FAMILIAR = Path("runs/limit-campaign-20261127/original-120-panel.json")
STARTING = Path("runs/limit-campaign-20261127/exports/approach-41-12000")
WEIGHT = "submission/weights/best_model.pt"


def read_status(root):
    path = root / "status.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def current_snapshot(root, state):
    for name in ("current_checkpoint", "last_successful_checkpoint"):
        candidate = state.get(name)
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    active = [job for job in state.get("active_jobs", []) if job["kind"] == "train"]
    for job in reversed(active):
        candidate = job.get("active_checkpoint")
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    completed = list(root.glob("training/*/step-*.pt"))
    return max(completed, key=lambda path: path.stat().st_mtime_ns, default=None)


def show_training_status(state):
    if state.get("active_training_seconds") is not None:
        print(f"  Headless learner active {state['active_training_seconds']:.1f}s | "
              f"remaining {state['remaining_training_seconds']:.1f}s | "
              f"decisions {state.get('training_decisions', 0):,} | PPO updates {state.get('ppo_updates', 0):,}",
              flush=True)
    scores = state.get("latest_development") or state.get("latest_validation", {})
    print(f"  Latest development results {json.dumps(scores, sort_keys=True)}", flush=True)


def pin_snapshot(snapshot, target):
    try:
        os.link(snapshot, target)
    except OSError:
        with snapshot.open("rb") as source, target.open("xb") as pinned:
            copyfileobj(source, pinned)
    with target.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def checkpoint_for_episode(args, work, state):
    checkpoint = args.checkpoint if args.checkpoint is not None else (
        current_snapshot(args.run, state) if args.repo is None else None)
    if checkpoint is None:
        repo = args.repo or STARTING
        with (repo / WEIGHT).open("rb") as stream:
            digest = file_digest(stream, "sha256").hexdigest()
        receipt = repo / "export.json"
        steps = json.loads(receipt.read_text())["trained_steps"] if receipt.is_file() else None
        return repo.resolve(), repo.name, digest, steps
    pinned = work / "snapshot.pt"
    digest = pin_snapshot(checkpoint, pinned)
    destination = work / "export"
    result = export(pinned, destination)
    return destination, checkpoint.parent.name, digest, result["trained_steps"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    chosen = parser.add_mutually_exclusive_group()
    chosen.add_argument("--checkpoint", type=Path)
    chosen.add_argument("--repo", type=Path)
    parser.add_argument("--panel", choices=("switch", "post-lock", "familiar"), default="switch")
    parser.add_argument("--map", action="append", dest="maps", default=[])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=0, help="0 runs until interrupted")
    add_playback_arguments(parser, visible=True)
    args = parser.parse_args()
    validate_playback_arguments(parser, args)
    if args.episodes < 0:
        parser.error("Episode count must be nonnegative.")
    if args.panel in ("switch", "post-lock"):
        manifest = DEVELOPMENT if args.panel == "switch" else args.run / "post-lock-maps" / "manifest.json"
        rows = load_rows(manifest, "validation", ids=args.maps)
    else:
        rows = [row for row in read_rows(FAMILIAR, "test") if not args.maps or row["id"] in args.maps]
        if set(args.maps) - {row["id"] for row in rows}:
            parser.error("A requested map does not occur in the familiar regression panel.")
    visits = cycle((row, seed) for seed in args.seeds for row in rows)
    iterator = islice(visits, args.episodes) if args.episodes else visits
    playback = Playback("FROZEN CHECKPOINT EVALUATION", args.speed, args.telemetry_every) if args.visualize else None
    try:
        for number, (row, seed) in zip(count(1), iterator):
            state = read_status(args.run)
            with TemporaryDirectory(prefix="doom-campaign-view-") as temporary:
                work = Path(temporary)
                repo, label, digest, steps = checkpoint_for_episode(args, work, state)
                description = (f"level {row['level']} | {row['task']} | {row['exit_type']}"
                               if args.panel != "familiar" else f"Tier {TIER[row['stage']]} | {row['stage']}")
                print(f"[FROZEN CHECKPOINT EVALUATION] episode {number} | {label} | {steps} lifetime decisions "
                      f"| SHA-256 {digest}\n  {row['id']} | {description} | engine seed {seed}",
                      flush=True)
                show_training_status(state)
                with exported_agent(repo, work / "agent.log", "cpu") as (agent, _):
                    result = play(row, seed, agent, playback=playback)
                print("[FROZEN CHECKPOINT EVALUATION] " + json.dumps(result, allow_nan=False), flush=True)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
