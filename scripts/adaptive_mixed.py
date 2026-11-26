"""Matched sequential and mixed on-policy PPO, using the verified warm checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
from random import Random
from tempfile import NamedTemporaryFile
import time

from scripts.run_exit_recovery import PracticeSchedule


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "runs/exit-recovery-study"
ARCHIVE = ROOT / "runs/limit-campaign-20261127"
OUT = ROOT / "runs/adaptive-campaign-20261127"
ARMS = ("sequential", "mixed")
LANES = ("approach", "approach", "align", "walkover", "keys", "combat", "exploration", "gauntlet")
SEEDS = (59, 83)
MIDPOINT, ENDPOINT = 12_000, 25_000
CAP = 100_000
WARM_STEPS = 314_496


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def records(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def pin(path):
    return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


def checked(record):
    path = Path(record["path"])
    if pin(path) != record:
        raise ValueError(f"Pinned campaign input changed: {path}")
    return path


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def classify(row):
    if not row["recovery_practice"]:
        return row["stage"]
    if row["exit_type"] == "walkover":
        return "walkover"
    return "align" if row["level"] == 0 else "approach"


class MixedSchedule:
    """Allocate fresh transitions by environment, independently of episode length."""

    def __init__(self, seed, lanes=LANES, *, warm_start=True):
        self.lanes = list(lanes)
        Random(f"mixed-lane-order:{seed}").shuffle(self.lanes)
        self.initial_streams = [Random(f"{seed}:{index}").getstate() for index in range(len(lanes))]
        self.warm_start = warm_start
        self.assignments = {}
        self.pools = None

    def catalog(self, maps):
        pools = {category: [] for category in self.lanes}
        for row in maps:
            pools[classify(row)].append(row)
        if any(not pool for pool in pools.values()):
            raise ValueError("All mixed practice categories need approved training maps")
        self.pools = pools

    def choose(self, maps, rng, _steps):
        if self.pools is None:
            self.catalog(maps)
        if rng not in self.assignments:
            # The trainer creates the per-environment RNGs and starts all environments in index order.
            # Later resets reuse their RNG object. Mixed continuation restores those streams in index order.
            index = len(self.assignments)
            if index >= len(self.lanes) or self.warm_start and rng.getstate() != self.initial_streams[index]:
                raise ValueError("Mixed practice requires all seeded environment lanes in index order")
            self.assignments[rng] = self.lanes[index]
        rng.random()
        pool = self.pools[self.assignments[rng]]
        return pool[int(rng.random() * len(pool))]


def audit_partition(train, development):
    for field in ("family", "geometry_sha256", "sha256"):
        if {row[field] for row in train} & {row[field] for row in development}:
            raise ValueError(f"Practice overlaps diagnostic validation by {field}")


def prepare(output):
    from training.maps import load_verified_maps

    provenance = read(PREVIOUS / "provenance/starting-checkpoint.json")
    if (not provenance["model_exact_export_match"] or provenance["snapshot_decisions"] != WARM_STEPS
            or not provenance["optimizer_restoration"]["finite_bit_identical_state"]):
        raise ValueError("The warm checkpoint identity receipt is incomplete")
    previous_pins = read(PREVIOUS / "pinned-inputs.json")
    for source in previous_pins["files"]:
        checked(source)
    inputs = {name: previous_pins[name] for name in ("warm_start", "reward", "config", "protocol", "map_qa")}
    inputs.update(training=previous_pins["training"]["curriculum"],
                  warm_export=provenance["frozen_export"],
                  published=provenance["original_checkpoints"]["published"],
                  frozen_exploration=provenance["original_checkpoints"]["frozen_exploration"],
                  development=pin(PREVIOUS / "maps/manifest.json"),
                  archived_qa=pin(ARCHIVE / "map-qa.json"))
    for source in inputs.values():
        checked(source)
    train = load_verified_maps(Path(inputs["training"]["path"]).parent)
    if len(train) != 112 or any(row["split"] != "train" for row in train):
        raise ValueError("Expected the 96 approved conditional maps and 16 shared training maps")
    development = [row for row in read(Path(inputs["development"]["path"]))["maps"] if row["split"] == "validation"]
    audit_partition(train, development)
    qa = read(Path(inputs["map_qa"]["path"]))
    if not qa["approved"] or qa["manifests"]["manifest.json"] != inputs["development"]["sha256"]:
        raise ValueError("Conditional maps lost their engine QA receipt")
    cfg = read(Path(inputs["config"]["path"]))
    if cfg["envs"] != len(LANES) or cfg["device"] != "cpu" or not cfg["paired_sampling"]:
        raise ValueError("Mixed practice requires the measured eight-serial-CPU matched config")
    old_protocol = read(Path(inputs["protocol"]["path"]))
    PracticeSchedule(old_protocol, staged=True).catalog(train)
    pools = Counter(map(classify, train))
    for seed in SEEDS:
        schedule = MixedSchedule(seed)
        schedule.catalog(train)
        streams = [Random(f"{seed}:{index}") for index in range(len(LANES))]
        for _ in range(6):
            draws = [classify(schedule.choose(train, stream, WARM_STEPS)) for stream in streams]
            if Counter(draws) != Counter(LANES):
                raise ValueError("The mixed warm-start preflight did not cover all categories")
            for stream in streams:
                stream.randrange(2**30)
    warmed_reward = load_reward(checked(inputs["reward"]))
    configurations = [warmed_reward(row, guided=True, gamma=cfg["gamma"], trigger_discipline=False) for row in train]
    if any(tracker._switch_geometry is None for row, tracker in zip(train, configurations, strict=True)
           if row["recovery_practice"] and row["exit_type"] == "switch"):
        raise ValueError("A switch map cannot receive the same approved approach reward")
    mix_counts = Counter(LANES)
    plan = dict(
        hypothesis="Mixed fresh on-policy practice retains skills while improving switch range and exits.",
        inputs=inputs, source_files=[*previous_pins["files"], pin(Path(__file__))],
        start_decisions=WARM_STEPS, arm_order=ARMS, seeds=SEEDS, milestones=[MIDPOINT, ENDPOINT],
        max_phase2_decisions=CAP, sequential_protocol=old_protocol,
        conditional_map_count=96, shared_map_count=16, pool_sizes=pools,
        training_config=cfg, frame_skip=4,
        mixed_transition_distribution={category: count / len(LANES) for category, count in mix_counts.items()},
        lane_order={str(seed): MixedSchedule(seed).lanes for seed in SEEDS},
        seed_stream="Per-environment Random seeded by matched training_seed:environment_index. Map and engine reset draws use the same count in both arms. Episodes reset at continuation.",
        final_panel_access="No final manifest or final WAD read during training or preflight.",
        development_gates={"required_in_both_training_seeds": True, "range_min": 24, "range_total": 54,
                           "range_min_gain_independent_layouts": 2, "align_min": 11, "align_total": 18,
                           "familiar_min": 27, "familiar_total": 120, "walkover_max_added_failures": 1,
                           "gauntlet_max_deaths": 21, "gauntlet_total": 30, "invalid_or_crash_max": 0,
                           "late_exit_min": 19, "late_exit_total": 54,
                           "late_exit_min_gain_independent_layouts": 2},
    )
    write(output / "mixed-plan.json", plan)
    print(json.dumps({"prepared": str(output / "mixed-plan.json"), "pool_sizes": pools,
                      "lane_order": plan["lane_order"]}), flush=True)


def load_reward(path):
    spec = importlib.util.spec_from_file_location("adaptive_archived_switch_reward", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SwitchApproachReward


def seconds_available(output):
    clock = read(output / "clock.json")
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != clock["boot_id"]:
        raise ValueError("The campaign's monotonic clock originates from another boot")
    return clock["training_cutoff_monotonic"] - time.monotonic()


def used_decisions(output):
    total = 0
    for manifest in (output / "training").glob("*/mixed-invocation.json"):
        invocation = read(manifest)
        folder = manifest.parent
        rows = records(folder / "learning.jsonl")
        total += sum(row["rollout_decisions"] for row in rows)
    return total


def audit(folder, plan, arm, label):
    from scripts.campaign_ledger import inspect

    receipt = inspect(folder)
    if receipt["initial_replay_log_prob_error"] is None or receipt["initial_replay_log_prob_error"] > 1e-4:
        raise ValueError("PPO replay log probabilities failed parity")
    maps = {row["id"]: row for row in read(Path(plan["inputs"]["training"]["path"]))["maps"]}
    envs = len(LANES)
    changes = iter(records(folder / "map-assignments.jsonl"))
    pending = next(changes, None)
    active = {}
    batches = []
    cumulative = Counter()
    for row in records(folder / "learning.jsonl"):
        counts = Counter()
        start = row["steps"] - row["rollout_decisions"]
        for decision in range(start, row["steps"], envs):
            while pending and pending["steps"] <= decision:
                active[pending["environment"]] = classify(maps[pending["map"]])
                pending = next(changes, None)
            if len(active) != envs:
                raise ValueError("Rollout audit cannot associate every transition with a map")
            counts.update(active.values())
        if sum(counts.values()) != row["rollout_decisions"]:
            raise ValueError("Rollout count disagrees with actual PPO decisions")
        if arm == "mixed" and counts != {category: int(weight * row["rollout_decisions"])
                                         for category, weight in plan["mixed_transition_distribution"].items()}:
            raise ValueError("A mixed PPO rollout did not have its scheduled transition allocation")
        for episode in row["episodes"]:
            paid = episode.get("switch_approach_shaping", {}).get("paid", 0.0)
            if not math.isfinite(paid) or not 0 <= paid <= 1.500001:
                raise ValueError("Invalid pose reward in completed training episode")
        cumulative.update(counts)
        batches.append({"steps": row["steps"], "transition_counts": counts})
    receipt.update(label=label, transition_counts=cumulative, batches=batches)
    write(folder / "mixed-receipt.json", receipt)
    return receipt


def learner(output, args):
    from scripts.export_candidate import export
    from training import train

    plan = read(output / "mixed-plan.json")
    for source in [*plan["inputs"].values(), *plan["source_files"]]:
        checked(source)
    if args.arm is None or args.seed is None or args.milestone is None:
        raise ValueError("A learner requires an arm, training seed and milestone")
    label = f"{args.arm}-{args.seed}-{args.milestone}"
    warm = args.milestone == MIDPOINT and args.resume is None
    target = WARM_STEPS + args.milestone
    if warm:
        source = checked(plan["inputs"]["warm_start"])
        start = WARM_STEPS
    elif args.resume is not None:
        import torch
        source = args.resume.resolve(strict=True)
        if not source.is_relative_to(output / "training"):
            raise ValueError("A recovery snapshot must come from this campaign")
        invocation = read(source.parent / "mixed-invocation.json")
        if invocation["label"] != label:
            raise ValueError("A recovery snapshot must resume the same experiment")
        start = torch.load(source, map_location="cpu", weights_only=True)["trained_steps"]
    else:
        middle = f"{args.arm}-{args.seed}-{MIDPOINT}"
        source = output / "training" / middle / f"step-{WARM_STEPS + MIDPOINT}.pt"
        if not source.is_file():
            source = Path(read(output / "exports" / middle / "export.json")["source_checkpoint"])
        start = WARM_STEPS + MIDPOINT
    if args.milestone == ENDPOINT:
        gate = read(output / "evaluation/midpoint-gate.json")
        if not gate.get("continue_matched_experiment"):
            raise ValueError("The pre-registered midpoint gate did not approve extension")
    if start >= target or used_decisions(output) + (target - start) > CAP:
        raise ValueError("The requested PPO invocation violates the decision budget")
    remaining = seconds_available(output)
    if remaining < 90 or (output / "STOP").exists():
        raise ValueError("Training cutoff is too close or a stop was requested")
    folder = output / "training" / (label if args.resume is None else f"{label}-resume-{start}")
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / "mixed-invocation.json", dict(label=label, start=start, target=target, source=pin(source),
                                                  plan=pin(output / "mixed-plan.json"), headless=not args.visualize))
    schedule = (PracticeSchedule(plan["sequential_protocol"], staged=True) if args.arm == "sequential"
                else MixedSchedule(args.seed, warm_start=warm))
    config = checked(plan["inputs"]["config"])
    map_root = Path(plan["inputs"]["training"]["path"]).parent
    options = ["--config", config, "--map-root", map_root, "--resume", source, "--seed", args.seed,
               "--out", folder, "--steps", target, "--checkpoint-steps", target,
               "--seconds", min(600, remaining - 30), "--stop-file", output / "STOP",
               "--visualize" if args.visualize else "--headless"]
    if args.visualize:
        options += ["--speed", args.speed]
    if warm:
        options.append("--restart-rng")
    before = os.environ.get("SDL_VIDEODRIVER")
    if args.visualize:
        os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    try:
        train.main(list(map(str, options)), select_training_map=schedule.choose,
                   course_reward=load_reward(checked(plan["inputs"]["reward"])))
    finally:
        if args.visualize and before is None:
            os.environ.pop("SDL_VIDEODRIVER", None)
    summary = read(folder / "summary.json")
    if summary["steps"] != target:
        recovery = folder / f"step-{summary['steps']}.pt"
        latest = folder / "latest.pt"
        if latest.is_file():
            import shutil
            with NamedTemporaryFile(dir=folder, prefix="recovery-", delete=False) as stream:
                temporary = Path(stream.name)
            try:
                shutil.copyfile(latest, temporary)
                temporary.replace(recovery)
            finally:
                temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Incomplete {label}: {summary['stop_reason']}. Recover from {recovery} if present.")
    audit(folder, plan, args.arm, label)
    export(folder / f"step-{target}.pt", output / "exports" / label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "learner"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--milestone", type=int, choices=(MIDPOINT, ENDPOINT))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.out.resolve())
    else:
        learner(args.out.resolve(), args)


if __name__ == "__main__":
    main()
