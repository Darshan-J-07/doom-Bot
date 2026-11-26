"""Evaluate the post-lock progression funnel through the exported neural agent."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from math import floor
from pathlib import Path

from scripts.adaptive_mixed import checked, read, records, write
from scripts.exit_switch_eval import evaluate, load_rows
from training.continuity_post_lock_maps import far_side


STAGES = ("door_crossing", "exit_area_discovery", "interaction_range", "alignment", "fresh_use", "confirmed_exit")


def funnel(spec, episode):
    """Condition switch signals on crossing, or on a certified far-side spawn."""
    stages = dict.fromkeys(STAGES)
    door = tuple(spec["open_door_cell"])
    far = far_side(spec)
    spawn_is_far = spec["post_lock_spawn"] == "far-side"
    tile = spec["tile"]

    def tile_at(x, y):
        return floor(x / tile[0]), floor(y / tile[1])

    previous = tile_at(*spec["expected_start"])
    entered_from_near = False
    with gzip.open(episode["trace"], "rt", encoding="utf-8") as trace:
        for line in trace:
            frame = json.loads(line)
            tic = frame["decision_tic_before"]
            sample = frame["switch_before_action"]
            if spawn_is_far or stages["door_crossing"] is not None:
                if (sample["area"] or sample["visible"]) and stages["exit_area_discovery"] is None:
                    stages["exit_area_discovery"] = tic
                if sample["near"] and stages["exit_area_discovery"] is not None and stages["interaction_range"] is None:
                    stages["interaction_range"] = tic
                if sample["facing"] and stages["interaction_range"] is not None:
                    if stages["alignment"] is None:
                        stages["alignment"] = tic
                    if frame["use_rising"] and stages["fresh_use"] is None:
                        stages["fresh_use"] = tic
            after = tile_at(frame["x"], frame["y"])
            if after == door and previous != door:
                entered_from_near = previous not in far
            elif previous == door and after != door:
                if after in far and entered_from_near and stages["door_crossing"] is None:
                    stages["door_crossing"] = frame["tic"]
                entered_from_near = False
            previous = after
    if episode["exit"]:
        stages["confirmed_exit"] = episode["ticks"]
    return dict(layout=spec["id"], family=spec["family"], engine_seed=episode["engine_seed"],
                exit_type=spec["exit_type"], spawn_side=spec["post_lock_spawn"], stage_tics=stages,
                survived=not episode["dead"], invalid_actions=episode["invalid_actions"],
                agent_error=episode["agent_error"], trace=episode["trace"])


def report(episodes):
    stages = Counter()
    for row in episodes:
        stages.update(name for name, tic in row["stage_tics"].items() if tic is not None)
    return dict(episodes=len(episodes), independent_layouts=len({row["layout"] for row in episodes}),
                stage_counts={name: stages[name] for name in STAGES})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Engine seeds must be distinct.")
    qa = read(args.map_root / "qa.json")
    manifest = checked(qa["manifest"])
    rows = load_rows(manifest, "validation", tasks=("conditional",))
    verified = {row["id"] for row in qa["witnesses"] if row["split"] == "validation"}
    if {row["id"] for row in rows} != verified:
        raise ValueError("Development maps do not match their engine witness receipt.")
    by_id = {row["id"]: row for row in rows}
    summary = evaluate(rows, args.repo, args.seeds, args.out, trace_every_seed=True,
                       manifest_sha256=qa["manifest"]["sha256"])
    episodes = [funnel(by_id[episode["map"]], episode) for episode in records(args.out / "episodes.jsonl")]
    cohorts = {}
    for spawn in sorted({row["spawn_side"] for row in episodes}):
        for mode in sorted({row["exit_type"] for row in episodes}):
            cohort = [row for row in episodes if (row["spawn_side"], row["exit_type"]) == (spawn, mode)]
            if cohort:
                cohorts[f"{spawn}/{mode}"] = report(cohort)
    write(args.out / "post-lock-funnel.json", dict(
        **report(episodes), cohorts=cohorts, stage_order=STAGES, by_episode=episodes,
        training_scope="Conditional post-lock only. No full keycard progress credited.",
        staged_at_spawn="A far-side spawn gives no crossing credit. A later observed recrossing after backtracking is counted.",
        measurement="Geometry samples precede four-tic actions. An engine-confirmed exit may occur before a precursor is sampled.",
    ))
    raise SystemExit(1 if summary["invalid_actions"] or summary["errors"] else 0)


if __name__ == "__main__":
    main()
