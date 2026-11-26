"""Prepare and run the bounded, matched exit-recovery PPO comparison."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "runs/limit-campaign-20261127"
DEFAULT_OUT = ROOT / "runs/exit-recovery-study"
WEIGHTS = "submission/weights/best_model.pt"
CONFIG = ARCHIVE / "config.json"
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "SDL_VIDEODRIVER": "x11",
       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
ARMS = ("control", "curriculum")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, record):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def pin(path):
    return {"path": str(path), "sha256": digest(path)}


def checked(record):
    path = Path(record["path"])
    if digest(path) != record["sha256"]:
        raise ValueError(f"Pinned input changed: {path}")
    return path


def usage(output):
    return sum(read(path)["invocation_steps"] for path in (output / "training").glob("*/summary.json"))


def prepare(output, protocol):
    from training.maps import load_verified_maps

    start = read(output / "provenance/starting-checkpoint.json")
    if not start["model_exact_export_match"] or not start["optimizer_restoration"]["finite_bit_identical_state"]:
        raise ValueError("Start checkpoint and Adam provenance must be verified")
    warm = checked(start["checkpoint"])
    checked(start["frozen_export"])
    reward = checked(start["training_only_reward"])
    archive_manifest = checked(start["previous_distribution"]["manifest"])
    map_root = output / "maps"
    qa = read(output / "map-qa.json")
    if not qa["approved"] or set(qa["manifests"]) != {"manifest.json", "final.json", "fixtures.json"}:
        raise ValueError("The complete map and fixture QA gate must pass")
    for name, expected in qa["manifests"].items():
        checked({"path": str(map_root / name), "sha256": expected})
    old = [row for row in load_verified_maps(archive_manifest.parent) if row["split"] == "train"]
    old_practice = [row for row in old if row.get("task") == "conditional"]
    shared = [row for row in old if row.get("task") != "conditional"]
    newer = load_verified_maps(map_root)
    new_practice = [row for row in newer if row["split"] == "train"]
    scored = [row for row in newer if row["split"] != "train"] + read(map_root / "final.json")["maps"]
    for key in ("geometry_sha256", "sha256", "family"):
        if {row[key] for row in old} & {row[key] for row in scored}:
            raise ValueError(f"Archived training overlaps validation or final on {key}")
    counts = Counter(row["exit_type"] for row in old_practice)
    if dict(counts, shared=len(shared)) != protocol["practice_weights"]:
        raise ValueError("Archived sampling distribution disagrees with the protocol")

    for subdir in ("training", "exports", "logs"):
        (output / subdir).mkdir(exist_ok=True)
    manifests = {}
    for name, practice in zip(ARMS, (old_practice, new_practice), strict=True):
        target = output / "training-data" / name
        target.mkdir(parents=True)
        training = []
        for row in practice + shared:
            destination = target / f"{row['sha256']}.wad"
            if destination.exists():
                raise ValueError(f"Duplicate WAD: {row['id']}")
            shutil.copyfile(row["path"], destination)
            training.append(dict(row, path=str(destination), recovery_practice=row.get("task") == "conditional"))
        write(target / "manifest.json", {"maps": training})
        load_verified_maps(target)
        PracticeSchedule(protocol, staged=name == "curriculum").catalog(training)
        manifests[name] = pin(target / "manifest.json")
    files = [Path(__file__), ROOT / "training/train.py", ROOT / "training/ppo.py",
             ROOT / "training/mechanics.py", ROOT / "training/rewards.py", ROOT / "training/skills.py",
             ROOT / "scripts/export_candidate.py"]
    files += sorted((ROOT / "env/doom_env").glob("*.py"))
    files += [ROOT / path for path in start["submission_files"] if path != WEIGHTS]
    for relative, expected in start["submission_files"].items():
        if relative != WEIGHTS:
            checked({"path": str(ROOT / relative), "sha256": expected})
    write(output / "pinned-inputs.json", dict(warm_start=pin(warm), reward=pin(reward), config=pin(CONFIG),
                                              protocol=pin(output / "protocol.json"), map_qa=pin(output / "map-qa.json"),
                                              training=manifests, files=[pin(path) for path in files]))


class PracticeSchedule:
    """Pair category and reset seed draws across arms with two RNG draws per reset."""

    def __init__(self, protocol, *, staged):
        self.start = protocol["warm_start_decisions"]
        self.boundaries = protocol["phases_start_added_decisions"] if staged else [0]
        self.limits = protocol["practice_weights"]
        self.staged = staged
        self.pools = None

    def catalog(self, maps):
        shared = [row for row in maps if not row["recovery_practice"]]
        if len(shared) != self.limits["shared"]:
            raise ValueError("Shared practice must have the archived map count")
        pools = []
        for level in range(len(self.boundaries)):
            current = [row for row in maps if row["recovery_practice"] and (not self.staged or row["level"] == level)]
            switch = [row for row in current if row["exit_type"] == "switch"]
            walkover = [row for row in current if row["exit_type"] == "walkover"]
            if not switch or not walkover or len(switch) * self.limits["walkover"] != len(walkover) * self.limits["switch"]:
                raise ValueError(f"Practice phase {level} has the wrong switch/walkover proportion")
            pools.append((switch, walkover, shared))
        self.pools = pools

    def choose(self, maps, rng, steps):
        if self.pools is None:
            self.catalog(maps)
        level = bisect_right(self.boundaries, steps - self.start) - 1
        if level < 0:
            raise ValueError("Practice started before the declared warm checkpoint")
        draw = rng.random() * sum(self.limits.values())
        index = 0 if draw < self.limits["switch"] else 1 if draw < self.limits["switch"] + self.limits["walkover"] else 2
        pool = self.pools[level][index]
        return pool[int(rng.random() * len(pool))]


def pinned_inputs(output):
    info = read(output / "pinned-inputs.json")
    for record in [info[name] for name in ("reward", "warm_start", "protocol", "config", "map_qa")] + info["files"]:
        checked(record)
    for record in info["training"].values():
        checked(record)
    return info


def learner(output, protocol, args):
    from training import train

    pins = pinned_inputs(output)
    spec = importlib.util.spec_from_file_location("archived_switch_reward", checked(pins["reward"]))
    reward = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reward)
    arm, seed, milestone = args.arm, args.seed, args.milestone
    if arm not in ARMS or seed not in protocol["training_seeds"] or milestone not in protocol["screen_decisions"]:
        raise ValueError("Unknown arm, seed, or milestone")
    middle = protocol["screen_decisions"][0]
    start = protocol["warm_start_decisions"]
    previous = output / "training" / f"{arm}-{seed}-{middle}" / f"step-{start + middle}.pt"
    if milestone == middle:
        initial = checked(pins["warm_start"])
    else:
        export = read(output / "exports" / f"{arm}-{seed}-{middle}" / "export.json")
        initial = checked({"path": str(previous), "sha256": export["source_checkpoint_sha256"]})
    run_dir = output / "training" / f"{arm}-{seed}-{milestone}"
    target = start + milestone
    schedule = PracticeSchedule(protocol, staged=arm == "curriculum")
    options = ["--config", CONFIG, "--map-root", Path(pins["training"][arm]["path"]).parent,
               "--resume", initial, "--seed", seed, "--out", run_dir, "--steps", target, "--checkpoint-steps", target,
               "--seconds", protocol["max_seconds_per_invocation"], "--stop-file", output / "STOP",
               "--headless" if args.headless else "--visualize"]
    if milestone == middle:
        options.append("--restart-rng")
    train.main(list(map(str, options)), select_training_map=schedule.choose, course_reward=reward.SwitchApproachReward)


def invoke(output, log_name, module, *args):
    command = [sys.executable, "-m", module, *map(str, args)]
    with (output / "logs" / f"{log_name}.log").open("x", encoding="utf-8") as log:
        log.write(json.dumps(command) + "\n")
        log.flush()
        subprocess.run(command, cwd=ROOT, env=ENV, check=True, stdout=log, stderr=subprocess.STDOUT)


def audit_learning(folder):
    from scripts.campaign_ledger import inspect

    receipt = inspect(folder)
    log = rows(folder / "learning.jsonl")
    if receipt["initial_replay_log_prob_error"] is None or receipt["initial_replay_log_prob_error"] > 1e-4:
        raise ValueError(f"PPO replay parity failed: {folder}")
    episodes = [episode for update in log for episode in update["episodes"]]
    for episode in episodes:
        pose = episode.get("switch_approach_shaping", {})
        if pose.get("paid", 0) < 0 or pose.get("paid", 0) > 1.500001:
            raise ValueError(f"Unexpected pose-reward payment: {folder}")
    return receipt, episodes


def validation_progress(folder):
    summary = read(folder / "summary.json")["overall"]
    if summary["errors"] or summary["invalid_actions"]:
        raise ValueError(f"Validation failed or produced invalid actions: {folder}")
    selected = [row for row in rows(folder / "episodes.jsonl")
                if row["exit_type"] == "switch" and row["switch"]["stages"]["switch_near"] != 0]
    if not selected:
        raise ValueError(f"Validation has no out-of-range switch episodes: {folder}")
    return dict(episodes=len(selected), exits=sum(row["exit"] for row in selected),
                range_entries=sum(row["switch"]["stages"]["switch_near"] is not None for row in selected))


def midpoint_safety(output, protocol):
    if not (output / "validation-12000.json").is_file():
        raise ValueError("Run the complete paired midpoint validation first")
    base = validation_progress(output / "validation/approach-baseline")
    specs = {row["path"]: row for row in read(output / "training-data/curriculum/manifest.json")["maps"]}
    results = {}
    for seed in protocol["training_seeds"]:
        level = protocol["screen_decisions"][0]
        control = validation_progress(output / "validation" / f"control-{seed}-{level}")
        name = f"curriculum-{seed}-{level}"
        curriculum = validation_progress(output / "validation" / name)
        if not base["episodes"] == control["episodes"] == curriculum["episodes"]:
            raise ValueError("Expected matched out-of-range validation counts")
        _, episodes = audit_learning(output / "training" / name)
        practice = [row for row in episodes if specs[row["map"]]["recovery_practice"] and row["level"] > 0
                    and specs[row["map"]]["exit_type"] == "switch"]
        training_exits = sum(row["exit"] for row in practice)
        rewarded = sum(row.get("switch_approach_shaping", {}).get("paid", 0) > 0 for row in practice)
        stalled = (not curriculum["exits"] and curriculum["range_entries"] <= max(base["range_entries"], control["range_entries"])
                   and not training_exits and rewarded >= 12)
        results[str(seed)] = dict(warm_start=base, control=control, curriculum=curriculum, training_practice_episodes=len(practice),
                                  training_switch_exits=training_exits, training_pose_reward_episodes=rewarded,
                                  reward_without_progress=stalled)
    approved = not any(item["reward_without_progress"] for item in results.values())
    write(output / "midpoint-safety.json", dict(approved=approved, seeds=results))
    print(json.dumps(dict(approved=approved, seeds=results)), flush=True)


def training_ledger(output, protocol):
    manifests = {arm: {row["path"]: row for row in read(output / "training-data" / arm / "manifest.json")["maps"]}
                 for arm in ARMS}
    runs = {}
    for folder in sorted((output / "training").iterdir()):
        arm, _, _ = folder.name.split("-")
        receipt, episodes = audit_learning(folder)
        spec = manifests[arm]
        stages = {}
        for episode in episodes:
            row = spec[episode["map"]]
            label = f"level-{row['level']}/{row['exit_type']}" if row["recovery_practice"] else f"shared/{row['stage']}"
            totals = stages.setdefault(label, Counter())
            totals.update(episodes=1, exits=int(episode["exit"]), reached=int(episode.get("switch_approach_shaping", {}).get("reached", False)))
        runs[folder.name] = dict(receipt, training_stages=stages)

    matching = {}
    for seed in protocol["training_seeds"]:
        for milestone in protocol["screen_decisions"]:
            labels = [f"{arm}-{seed}-{milestone}" for arm in ARMS]
            if not all(label in runs for label in labels):
                continue
            patterns = []
            for arm, label in zip(ARMS, labels, strict=True):
                specs = {row["id"]: row for row in manifests[arm].values()}
                workers = {}
                for item in rows(output / "training" / label / "map-assignments.jsonl"):
                    spec = specs[item["map"]]
                    category = spec["exit_type"] if spec["recovery_practice"] else spec["sha256"]
                    workers.setdefault(item["environment"], []).append((category, item["engine_seed"]))
                patterns.append(workers)
            control, new = patterns
            if control.keys() != new.keys():
                raise ValueError("The number of environment streams changed between arms")
            same = 0
            if milestone == protocol["screen_decisions"][0]:
                for worker, stream in control.items():
                    n = min(len(stream), len(new[worker]))
                    if stream[:n] != new[worker][:n]:
                        raise ValueError("Mismatched practice category or engine seed in paired stream")
                    same += n
            # A phase-two stream continues its own phase-one prefix, which has a policy-dependent length.
            matching[f"{seed}-{milestone}"] = dict(reset_counts=[sum(map(len, pattern.values())) for pattern in patterns],
                                                    exact_common_starts=same if same else None)
    totals = dict(decisions=sum(item["invocation_steps"] for item in runs.values()),
                  updates=sum(item["invocation_updates"] for item in runs.values()),
                  elapsed_seconds=sum(item["elapsed_seconds"] for item in runs.values()))
    write(output / "training-ledger.json", dict(**totals, runs=runs, paired_sampling=matching))
    print(json.dumps(totals), flush=True)


def wave(output, protocol, milestone, headless):
    pinned_inputs(output)
    middle = protocol["screen_decisions"][0]
    length = milestone if milestone == middle else milestone - middle
    start = protocol["warm_start_decisions"]
    if milestone != middle and not read(output / "midpoint-safety.json")["approved"]:
        raise ValueError("The midpoint safety gate did not authorize continuing")
    for i, seed in enumerate(protocol["training_seeds"]):
        for arm in (ARMS if i % 2 == 0 else ARMS[::-1]):
            name = f"{arm}-{seed}-{milestone}"
            folder = output / "training" / name
            if not folder.exists():
                if usage(output) + length > protocol["max_training_decisions"]:
                    raise ValueError("This run would exceed the decision budget")
                if (output / "STOP").exists():
                    raise ValueError("A stop was requested")
                print(f"{name} | {length:,} training decisions", flush=True)
                flags = ["--headless"] if headless else []
                invoke(output, f"train-{name}", "scripts.run_exit_recovery", "learner", "--out", output,
                       "--arm", arm, "--seed", seed, "--milestone", milestone, *flags)
            result, _ = audit_learning(folder)
            if result["invocation_steps"] != length or result["steps"] != start + milestone:
                raise ValueError(f"Incomplete {name}. Retain its checkpoint and inspect the stop reason.")
            checkpoint = folder / f"step-{start + milestone}.pt"
            destination = output / "exports" / name
            if destination.is_dir():
                if read(destination / "export.json")["source_checkpoint_sha256"] != digest(checkpoint):
                    raise ValueError(f"Existing export disagrees with {checkpoint}")
            else:
                invoke(output, f"export-{name}", "scripts.export_candidate", "--checkpoint", checkpoint, "--out", destination)
    print(f"Total completed PPO decisions {usage(output):,}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "midpoint", "safety", "finish", "ledger", "learner"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--milestone", type=int)
    args = parser.parse_args()
    output = args.out.resolve()
    protocol = read(output / "protocol.json")
    if args.phase == "prepare":
        prepare(output, protocol)
    elif args.phase == "learner":
        learner(output, protocol, args)
    elif args.phase == "safety":
        midpoint_safety(output, protocol)
    elif args.phase == "ledger":
        training_ledger(output, protocol)
    else:
        milestone = protocol["screen_decisions"][0 if args.phase == "midpoint" else -1]
        wave(output, protocol, milestone, args.headless)


if __name__ == "__main__":
    main()
