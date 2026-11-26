"""Compare frozen checkpoints using explicit local tier weights."""
import argparse
import json
from pathlib import Path

import torch

from training.evaluate import evaluate
from training.maps import load_verified_maps

TIER_WEIGHTS = dict(combat=1, exploration=2, keys=3, gauntlet=4)
METRICS = ("proxy_score", "exit", "survived", "kills", "damage", "keys", "doors", "locks", "unique_cells")


def tier_metrics(episodes):
    tiers = {}
    for stage in ("movement", *TIER_WEIGHTS):
        rows = [row for row in episodes if row["stage"] == stage]
        if rows:
            tiers[stage] = dict(episodes=len(rows), **{
                key: sum(row[key] for row in rows) / len(rows) for key in METRICS
            })
    return tiers


def local_score(tiers):
    return sum(weight * tiers[stage]["proxy_score"] for stage, weight in TIER_WEIGHTS.items()) / sum(TIER_WEIGHTS.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, type=Path)
    parser.add_argument("--memory-ablation", action="append", default=[], type=Path)
    parser.add_argument("--map-root", default="training/maps")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--policy-seed", type=int, help="fixed policy RNG seed, 0 for deployment parity")
    parser.add_argument("--ticks", type=int, default=2100)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Report already exists. Use a new path.")
    if args.split == "test" and args.memory_ablation:
        parser.error("Memory ablations use the validation split.")
    torch.set_num_threads(1)
    maps = [m for m in load_verified_maps(args.map_root) if m["split"] == args.split]
    if not set(TIER_WEIGHTS).issubset(m["stage"] for m in maps):
        parser.error("All four scored tiers are required.")
    entries = [(path, False) for path in args.checkpoint] + [(path, True) for path in args.memory_ablation]
    candidates = []
    for path, reset in entries:
        result = evaluate(path, maps, seeds=args.seeds, ticks=args.ticks,
                          reset_memory=reset, policy_seed=args.policy_seed)
        tiers = tier_metrics(result["episodes"])
        row = dict(**result, tiers=tiers, tier_weighted_proxy=local_score(tiers), reset_memory=reset)
        candidates.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "episodes"}), flush=True)
    report = dict(split=args.split, score="local proxy with project-defined tier weights",
                  tier_weights=TIER_WEIGHTS, policy_seed=args.policy_seed, candidates=candidates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
