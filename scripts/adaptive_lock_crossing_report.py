"""Compare the single-factor key-lane probe and verify relevant lock traversals."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from math import floor
from pathlib import Path

from scripts import campaign_eval
from scripts.adaptive_campaign_eval import APPROACH, archived, compare, frozen_scope, panels, protocol
from scripts.adaptive_mixed import MIDPOINT, OUT, SEEDS, checked, pin, read, records, write


def near(left, right):
    return left is not None and sum(abs(a - b) for a, b in zip(left, right, strict=True)) == 1


def crossing(frames, door, tile, unlock_tic):
    prior, entered_from = None, None
    for frame in frames:
        position = (floor(frame["x"] / tile[0]), floor(frame["y"] / tile[1]))
        if frame["tic"] >= unlock_tic:
            if position == door and near(prior, door):
                entered_from = prior
            elif prior == door and position != door:
                if entered_from is not None and position != entered_from and near(position, door):
                    return dict(tic=frame["tic"], entry_neighbor=entered_from, exit_neighbor=position)
                entered_from = None
        prior = position
    return None


def observed_progress(episodes, maps):
    outcomes = []
    for episode in episodes:
        stage = episode["stage"]
        if stage not in ("keys", "gauntlet"):
            continue
        spec = maps[episode["wad_sha256"], episode["sector_map"]]
        traces = []
        trace = episode.get("trace")
        cells = spec.get("drill", {}).get("cells")
        unavailable = bool(episode["unlocks"] and (not trace or not Path(trace).is_file() or not cells))
        if episode["unlocks"] and not unavailable:
            tiles = sorted(((x, y), glyph) for x, y, glyph in cells)
            with gzip.open(trace, "rt", encoding="utf-8") as stream:
                frames = [json.loads(line) for line in stream]
            for event in episode["unlocks"]:
                door, glyph = tiles[event["sector"]]
                if glyph not in "RBY" or event.get("since_key") is None:
                    continue
                observed = crossing(frames, door, spec["tile"], event["tic"])
                if observed is not None:
                    traces.append(dict(sector=event["sector"], unlock_tic=event["tic"],
                                       key_tic=event["tic"] - event["since_key"], **observed))
        outcomes.append(dict(map=episode["map"], family=episode["family"], tier=stage,
                             engine_seed=episode["engine_seed"], key=bool(episode["keys"]),
                             unlocked=bool(episode["locks"]), traversed=None if unavailable else bool(traces),
                             exit=episode["exit"], dead=episode["dead"], crossings=traces))
    by_tier = {}
    for stage in ("keys", "gauntlet"):
        tier = [row for row in outcomes if row["tier"] == stage]
        by_tier[stage] = dict(episodes=len(tier), independent_layouts=len({row["map"] for row in tier}),
                             traversals=sum(row["traversed"] is True for row in tier),
                             unmeasured_crossings=sum(row["traversed"] is None for row in tier),
                             crossing_layouts=sorted({row["map"] for row in tier if row["traversed"]}),
                             exits=sum(row["exit"] for row in tier),
                             key_unlock_cross_exit=sum(row["traversed"] is True and row["exit"] for row in tier),
                             deaths=sum(row["dead"] for row in tier))
    return dict(tiers=by_tier, episodes=outcomes)


def replay_warm(output):
    cohorts = panels(protocol(output))
    pins, _, familiar = archived(cohorts)
    _, specs, _ = cohorts["familiar"]
    lookup = {(row["sha256"], row["map_name"]): row for row in specs}
    pending = [row for row in familiar[APPROACH] if row["unlocks"] and not row.get("trace")]
    target = output / "warm-lock-replays"
    target.mkdir(exist_ok=False)
    new = []
    parent = pins[APPROACH]
    with campaign_eval.exported_agent(Path(parent["repo"]), target / "agent.log", "cpu") as (agent, _):
        for old in pending:
            spec = lookup[old["wad_sha256"], old["sector_map"]]
            seed = old["engine_seed"]
            path = target / f"{spec['id']}-seed-{seed}.jsonl.gz"
            replay = campaign_eval.play(spec, seed, agent, trace_path=path)
            for field in ("exit", "dead", "keys", "locks", "unlocks", "decisions", "action_sha256"):
                if replay[field] != old[field]:
                    raise ValueError(f"Warm replay changed archived {field}: {spec['id']} seed {seed}")
            new.append(dict(wad_sha256=old["wad_sha256"], sector_map=old["sector_map"], engine_seed=seed,
                            map=spec["id"], action_sha256=old["action_sha256"], trace=pin(path)))
    write(target / "index.json", dict(policy=parent, episodes=new, all_action_hashes_and_outcomes_match=True))
    print(f"Verified {len(new)} warm replays against archived action hashes and outcomes", flush=True)


def run(output):
    cohorts = panels(protocol(output))
    pins, _, familiar = archived(cohorts)
    _, specs, _ = cohorts["familiar"]
    lookup = {(row["sha256"], row["map_name"]): row for row in specs}
    reference = familiar[APPROACH]
    supplement = output / "warm-lock-replays/index.json"
    if supplement.is_file():
        replays = read(supplement)
        if not replays["all_action_hashes_and_outcomes_match"] or replays["policy"] != pins[APPROACH]:
            raise ValueError("Warm replay receipt disagrees with the development baseline")
        identity = lambda row: (row["wad_sha256"], row["sector_map"], row["engine_seed"], row["action_sha256"])
        observed = {identity(row): str(checked(row["trace"])) for row in replays["episodes"]}
        reference = [dict(row, trace=observed.get(identity(row), row.get("trace"))) for row in reference]
    result = {"frozen_start": observed_progress(reference, lookup), "seeds": {}}
    for seed in SEEDS:
        treatment = f"key-lock-crossing-{seed}-{MIDPOINT}"
        control = f"mixed-{seed}-{MIDPOINT}"
        outputs = {}
        for label in (control, treatment):
            first, switch = frozen_scope(output, "switch", label, cohorts["switch"])
            second, standard = frozen_scope(output, "familiar", label, cohorts["familiar"])
            if first != second:
                raise ValueError("The evaluated policies changed between scopes")
            outputs[label] = first, switch, standard
        first, switch, standard = outputs[control]
        candidate, crossed_switch, crossed_standard = outputs[treatment]
        path = output / "training" / treatment / "learning.jsonl"
        drills = [episode for update in records(path) for episode in update["episodes"] if "closed_lock_tics" in episode]
        counts = Counter({event: 0 for event in ("key", "unlocked", "entered_door", "crossed", "exit")})
        for episode in drills:
            counts.update(event for event, tic in episode["closed_lock_tics"].items() if tic is not None)
        result["seeds"][str(seed)] = dict(control=first, treatment=candidate,
                                          paired=compare(switch, crossed_switch, standard, crossed_standard),
                                          control_progression=observed_progress(standard, lookup),
                                          treatment_progression=observed_progress(crossed_standard, lookup),
                                          practice_episodes=len(drills), practice_progress=counts,
                                          practice_records=drills)
    path = output / "evaluation" / "paired-key-lock-crossing-vs-mixed.json"
    write(path, dict(plan=pin(output / "lock-crossing-plan.json"),
                     acquired_baseline_traces=pin(supplement) if supplement.is_file() else None,
                     definition="Sampled four-tic door sector entry after observed key and unlock, followed by exit to a different adjacent tile.",
                     **result))
    print(path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--replay-warm", action="store_true", help="Fill missing warm lock traces from exact artifact")
    args = parser.parse_args()
    operation = replay_warm if args.replay_warm else run
    operation(args.out.resolve())
