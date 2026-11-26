"""Summarize paired WAD-and-seed results and whole-map uncertainty."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def identity(row):
    return row["wad_sha256"], row["sector_map"], row["engine_seed"]


def summarize(rows):
    observed = [row for row in rows if not row.get("engine_error")]
    decisions = sum(row.get("decisions", 0) for row in observed)

    def count(key):
        return sum(bool(row.get(key)) for row in observed)

    def average(key, source=observed):
        values = [row[key] for row in source if row.get(key) is not None]
        return float(np.mean(values)) if values else None

    def work(key):
        values = [row["work"][key] for row in observed if row.get("work", {}).get(key) is not None]
        return sum(values) if values else None

    exits = [row for row in observed if row["exit"]]
    next_use = [lock["since_key"] for row in observed for lock in row.get("unlocks", []) if lock["since_key"] is not None]
    return dict(episodes=len(rows), maps=len({identity(r)[:2] for r in rows}),
                families=len({r["family"] for r in rows}), seeds=sorted({r["engine_seed"] for r in rows}),
                exits=len(exits), deaths=count("dead"), tick_caps=sum(row["outcome"] == "tick_cap" for row in observed),
                invalid_actions=count("invalid_actions"), crashes=count("crashes"), inference_timeouts=count("inference_timeouts"),
                engine_errors=len(rows) - len(observed), decision_count=decisions,
                exit_seconds_mean=average("duration_seconds", exits), estimated_exits=sum(r["duration_estimated"] for r in exits),
                damaged_mean=average("damage"), damage_at_exit_mean=average("damage", exits),
                health_at_exit_mean=average("health", exits), armor_mean=average("armor"),
                kills=sum(row.get("kills", 0) for row in observed), kill_episodes=count("kills"),
                kill_and_exit=sum(bool(row["kills"]) for row in exits), hits=sum(row.get("hits",0) for row in observed),
                damage_dealt=sum(row.get("damage_dealt",0.) for row in observed), shots=None,
                rounds_spent_lower_bound=work("rounds_spent_lower_bound"),
                keys_collected=sum(row.get("keys",0) for row in observed), key_episodes=count("keys"),
                keys_seen_episodes=sum(bool(row["work"].get("key_visible_steps")) for row in observed),
                locks_unlocked=sum(row.get("locks",0) for row in observed), unlock_episodes=count("locks"),
                key_and_unlock=sum(bool(r["keys"] and r["locks"]) for r in observed),
                key_and_exit=sum(bool(r["keys"]) for r in exits), key_unlock_and_exit=sum(bool(r["keys"] and r["locks"]) for r in exits),
                mean_key_to_unlock_tics=float(np.mean(next_use)) if next_use else None,
                doors_opened=sum(row.get("doors",0) for row in observed), doors_traversed=sum(row.get("doors_traversed",0) for row in observed),
                unique_areas_mean=average("unique_cells"), longest_progress_drought_mean=average("longest_no_progress"),
                episodes_with_stalls=sum(row.get("longest_no_progress", 0) >= 50 for row in observed),
                dead_end_entries=work("dead_end_entries"), dead_end_departures=work("dead_end_departures"),
                rotation_only_fraction=(work("spin_window_steps") or 0)/max(decisions, 1),
                blocked_fraction=(work("blocked_move") or 0)/max(decisions, 1),
                loaded_weapon_edges=work("weapon_press_while_loaded"),
                inference_ms=average("inference_ms"), worst_episode_p99_ms=max((r["p99_ms"] for r in observed if r.get("p99_ms") is not None),default=None))


def paired(rows, baseline, metric):
    candidate = {identity(row): row for row in rows}
    reference = {identity(row): row for row in baseline}
    if len(candidate) != len(rows) or len(reference) != len(baseline) or candidate.keys() != reference.keys():
        raise ValueError("Expected exactly the same unique WAD, internal map and engine seed pairs")
    differences, family = defaultdict(list), {}
    for trial, row in candidate.items():
        control = reference[trial]
        if row.get("engine_error") or control.get("engine_error"):
            continue
        differences[trial[:2]].append(metric(row) - metric(control))
        family[trial[:2]] = row["family"]
    if not differences:
        return None
    values = np.array([np.mean(trials) for trials in differences.values()])
    rng = np.random.default_rng(36131)
    samples = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    families = np.array([family[key] for key in differences])
    return dict(delta=float(values.mean()), maps=len(values), pairs=sum(map(len, differences.values())),
                low=float(np.quantile(samples, .025)), high=float(np.quantile(samples, .975)),
                layout_families=len(set(families)),
                leaving_out_family={name: float(values[families != name].mean())
                                    for name in sorted(set(families)) if not np.all(families == name)})


def panel(baseline, candidates):
    metrics = dict(exits=lambda row: int(row["exit"]), unlocks=lambda row: int(bool(row["locks"])),
                   kill_and_exit=lambda row: int(row["exit"] and bool(row["kills"])),
                   damage=lambda row: row["damage"], stalls=lambda row: int(row["longest_no_progress"] >= 50))

    def groups(rows):
        return {str(tier): [row for row in rows if row["tier"] == tier] for tier in sorted({row["tier"] for row in rows})}

    def policy(rows):
        by_map = defaultdict(list)
        for row in rows:
            by_map[row["map"]].append(row)
        return dict(overall=summarize(rows), tiers={tier: summarize(group) for tier, group in groups(rows).items()},
                    per_map={name: summarize(group) for name, group in by_map.items()})

    reference = groups(baseline)
    result = {"reference": policy(baseline)}
    for label, rows in candidates.items():
        item = policy(rows)
        item["paired_vs_reference"] = {"all": {key: paired(rows, baseline, measure) for key, measure in metrics.items()}}
        for tier, tier_rows in groups(rows).items():
            item["paired_vs_reference"][tier] = {key: paired(tier_rows, reference[tier], measure)
                                                  for key, measure in metrics.items()}
        hard = [row for row in rows if row["tier"] >= 3]
        if hard:
            hard_reference = [row for row in baseline if row["tier"] >= 3]
            item["paired_vs_reference"]["hard"] = {key: paired(hard, hard_reference, measure)
                                                   for key, measure in metrics.items()}
        result[label] = item
    return result


def write_curves(output, training):
    columns = ("run", "updates", "decisions", "seconds", "stage", "episodes", "exits", "kills", "keys", "locks", "mean_damage",
               "mean_coverage", "longest_stall_mean", "drill_success", "entropy", "clip_fraction", "approx_kl", "grad_norm")
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        for name, path in training.items():
            rows = read(path)
            first = rows[0]["steps"] - rows[0].get("rollout_decisions", 1024)
            for start in range(0, len(rows), 8):
                window = rows[start:start+8]
                ending = window[-1]
                group = defaultdict(list)
                for entry in window:
                    for episode in entry["episodes"]:
                        group[episode["stage"]].append(episode)
                for stage, episodes in group.items():
                    def mean(key):
                        found = [episode[key] for episode in episodes if key in episode]
                        return sum(found)/len(found) if found else None
                    writer.writerow(dict(run=name, updates=ending["updates"], decisions=ending["steps"]-first,
                                         seconds=ending["seconds"], stage=stage, episodes=len(episodes),
                                         exits=sum(o["exit"] for o in episodes), kills=sum(o["kills"] for o in episodes),
                                         keys=sum(o["keys"] for o in episodes), locks=sum(o["locks"] for o in episodes),
                                         mean_damage=mean("damage"), mean_coverage=mean("coverage"),
                                         longest_stall_mean=mean("longest_no_progress"), drill_success=mean("drill_success"),
                                         **{key: ending[key] for key in ("entropy", "clip_fraction", "approx_kl", "grad_norm")}))


def named(pairs):
    return {name: Path(path) for name, path in (pair.split("=",1) for pair in pairs)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[], metavar="NAME=EPISODES.jsonl")
    parser.add_argument("--training", action="append", default=[], metavar="NAME=LEARNING.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = panel(read(args.reference), {name: read(path) for name, path in named(args.candidate).items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.training:
        write_curves(args.out.with_suffix(".curves.csv"), named(args.training))
