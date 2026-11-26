"""Export matched policy metrics and map-family bootstrap intervals."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from scripts.campaign_report import identity, read
from scripts.exit_switch_eval import cohort, grouped, summarize


PAIR_METRICS = {
    "exit": lambda row: bool(row["exit"]),
    "activation": lambda row: bool(row.get("switch", {}).get("activation")),
    "faced_in_range": lambda row: reached(row, "switch_faced_in_range"),
    "attempted": lambda row: reached(row, "switch_attempt"),
    "survived": lambda row: not bool(row.get("dead") or row.get("agent_error") or row.get("engine_error")),
}
METHOD = {
    "switch_activation": "Authoritative wrapper exit on a map whose sole exit is a switch.",
    "discovery": "Exit cell visited before a decision or a geometrically visible point on the target switch.",
    "switch_visible": "Pre-action geometry proxy using sampled points in a 90-degree view with line of sight. This measures a recognition opportunity.",
    "approach": "A transition into pre-action facing and the nominal 64-unit unobstructed USE range.",
    "attempt": "A rising USE edge at a pre-action approach opportunity. Movement and turning within the four-tic step can change the true intercept.",
    "timing": "Wrapper time at 35 tics per second. A reset of the engine clock at exit is estimated from the final four-tic interval.",
    "ammo": "Net ammo decreases summed across ammo types. Pickups and terminal inventory changes can mask spending.",
    "latency": "Round trip through the one-CPU exported DoomAgent process, including transport and preprocessing.",
    "bootstrap": "Equal-weight paired differences across map families. Average all seeds within each map, then all maps within each family. Resample families 20,000 times.",
}


def reached(row, stage):
    return row.get("switch", {}).get("stages", {}).get(stage) is not None


def matching(reference, candidate):
    left = {identity(row): row for row in reference}
    right = {identity(row): row for row in candidate}
    if left.keys() != right.keys() or len(left) != len(reference) or len(right) != len(candidate):
        raise ValueError("Expected one episode per identical WAD, internal map and engine seed")
    pairs = [(left[key], right[key]) for key in sorted(left)]
    for a, b in pairs:
        if a["family"] != b["family"] or cohort(a) != cohort(b) or a["policy_seed"] != b["policy_seed"]:
            raise ValueError("Paired episodes disagree on family, cohort or policy RNG seed")
    return pairs


def family_differences(pairs, metric):
    layout = defaultdict(list)
    for a, b in pairs:
        layout[a["family"], identity(a)[:2]].append(int(metric(b)) - int(metric(a)))
    families = defaultdict(list)
    for (family, _), values in layout.items():
        families[family].append(float(np.mean(values)))
    return {family: float(np.mean(values)) for family, values in sorted(families.items())}


def bootstrap(pairs, metric, *, samples=20000):
    family = family_differences(pairs, metric)
    if not family:
        return None
    delta = np.array(list(family.values()))
    low = high = None
    if len(delta) >= 2:
        rng = np.random.default_rng(20261126)
        resampled = rng.choice(delta, size=(samples, len(delta)), replace=True).mean(axis=1)
        low, high = map(float, np.quantile(resampled, [.025, .975]))
    return dict(delta=float(delta.mean()), low=low, high=high, episodes=len(pairs), families=len(family),
                per_family=family,
                leaving_out_family={name: float(np.delete(delta, i).mean()) for i, name in enumerate(family) if len(delta) >= 2})


def compare(reference, candidate):
    pairs = matching(reference, candidate)
    cohorts = defaultdict(list)
    for a, b in pairs:
        cohorts[cohort(a)].append((a, b))
    return {name: {metric: bootstrap(group, measure) for metric, measure in PAIR_METRICS.items()}
            for name, group in sorted(cohorts.items())}


def representatives(rows):
    result = {}
    for row in rows:
        phase = row.get("switch", {}).get("first_unreached_stage")
        label = "exit" if row["exit"] else phase or row["outcome"]
        key = f"{cohort(row)}/{label}"
        if row.get("trace") and key not in result:
            result[key] = dict(map=row["map"], seed=row["engine_seed"], trace=row["trace"],
                               stages=row["switch"]["stages"], outcome=row["outcome"])
    return result


def policy_report(folder, rows):
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8")) if folder.is_dir() else {}
    return dict(summary=summary, all=summarize(rows), by_family=grouped(rows, lambda row: row["family"]),
                by_cohort=grouped(rows, cohort), trajectories=representatives(rows))


def write_csv(path, policies):
    keys = ("episodes", "exits", "switch_activations", "approaches", "activation_attempts", "distractor_attempts",
            "held_use_at_opportunity", "keys", "locks", "completed_key_lock_exit", "deaths", "tick_caps",
            "invalid_actions", "mean_exit_seconds", "mean_damage", "ammo_spent_lower_bound", "mean_roundtrip_latency_ms")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, ("policy", "scope", "group", *keys))
        writer.writeheader()
        for name, report in policies.items():
            for scope, groups in (("cohort", report["by_cohort"]), ("family", report["by_family"])):
                for group, stats in groups.items():
                    writer.writerow(dict(policy=name, scope=scope, group=group, **{key: stats.get(key) for key in keys}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--pair", action="append", default=[], metavar="REFERENCE=CANDIDATE")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sources = {name: Path(path) for name, path in (value.split("=", 1) for value in args.policy)}
    if len(sources) != len(args.policy):
        parser.error("Policy names must be distinct")
    rows = {name: read(path / "episodes.jsonl" if path.is_dir() else path) for name, path in sources.items()}
    policies = {name: policy_report(sources[name], records) for name, records in rows.items()}
    paired = {}
    for pair in args.pair:
        left, right = pair.split("=", 1)
        paired[f"{right}_vs_{left}"] = compare(rows[left], rows[right])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(method=METHOD, policies=policies, paired=paired), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(args.out.with_suffix(".csv"), policies)


if __name__ == "__main__":
    main()
