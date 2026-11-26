"""Record actual training use, PPO diagnostics and parameter updates per experiment."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

import torch

from scripts.campaign_report import read


def inspect(folder):
    summary = json.loads((folder / "summary.json").read_text())
    logs = read(folder / "learning.jsonl")
    checkpoint = Path(summary["checkpoint"])
    current = torch.load(checkpoint, map_location="cpu", weights_only=True)
    start_file = Path(summary["resumed_from"])
    start = torch.load(start_file, map_location="cpu", weights_only=True)
    changes = {key: float((value - start["model"][key]).abs().max()) for key, value in current["model"].items()
               if value.dtype.is_floating_point and key in start["model"]}
    assert current["trained_steps"] - start["trained_steps"] == summary["invocation_steps"]
    assert current["updates"] - start["updates"] == summary["invocation_updates"] == len(logs)
    assert all(torch.isfinite(value).all() for value in current["model"].values())
    assert any(change > 0 for change in changes.values())
    stats = {}
    for field in ("entropy", "approx_kl", "clip_fraction", "grad_norm", "policy_loss", "value_loss", "explained_variance"):
        values = [row[field] for row in logs if row[field] is not None]
        if values:
            assert all(math.isfinite(value) for value in values)
            stats[field] = dict(start=values[0], end=values[-1], min=min(values), max=max(values))
    return dict(summary, checkpoint_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
                starting_checkpoint_sha256=sha256(start_file.read_bytes()).hexdigest(),
                sampled_episodes=sum(len(r["episodes"]) for r in logs),
                throughput=summary["invocation_steps"]/summary["elapsed_seconds"],
                updated_tensor_count=sum(value > 0 for value in changes.values()),
                initial_replay_log_prob_error=logs[0].get("replay_log_prob_max_error"),
                tensor_max_abs_deltas=changes, stats=stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    reports = {folder.name: inspect(folder) for folder in args.run}
    result = dict(runs=reports, decisions=sum(row["invocation_steps"] for row in reports.values()),
                  updates=sum(row["invocation_updates"] for row in reports.values()),
                  elapsed_seconds=sum(row["elapsed_seconds"] for row in reports.values()))
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: result[name] for name in ("decisions", "updates", "elapsed_seconds")}, indent=2))
