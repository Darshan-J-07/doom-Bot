"""Evaluate matched exit-approach curricula through the released DoomAgent."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import time

from doom_env.spaces import FRAME_SKIP
from scripts import campaign_eval, exit_switch_eval as switch_eval
from scripts.campaign_report import identity, read
from scripts.exit_switch_report import bootstrap, compare, matching, reached, representatives


ROOT = Path(__file__).resolve().parents[1]
LIMIT = ROOT / "runs/limit-campaign-20261127"
SKILL = ROOT / "runs/skill-campaign"
WEIGHT = "submission/weights/best_model.pt"
APPROACH = LIMIT / "exports/approach-41-12000"
EXPLORATION = SKILL / "frozen/repo-checkout"
PUBLISHED = SKILL / "baseline-source"
BASELINES = {
    "approach-baseline": (APPROACH, "e18cfbfd323ed3df5f8d499bbd0e19174ea00bb0f3f3f042953a74a496aefee8"),
    "exploration-baseline": (EXPLORATION, "396a64abf6f358a6f298ef8d675c05a022b94482e9847bb9333e9a02876668dd"),
    "published": (PUBLISHED, "fe14608eaa2728ebe2389b7940144713ab6c0ce39da28a1aba24f576005a463f"),
}
STALL_DECISIONS = 50
METHOD = {
    "stages": "First pre-action area/view, nominal 64-unit geometric switch range, unobstructed facing and fresh targeted USE. Exit is engine-confirmed.",
    "closest_approach": "Minimum pre-action distance from the player to the switch segment in map units.",
    "stall": "GameplayObserver no-progress intervals of at least 50 decisions. Progress is a new cell, key, opened door, damage dealt or exit.",
    "entry_gate": "Switch episodes that start outside range on both sides of the pair. Layout gains are counted by geometry digest, averaging repeated engine seeds.",
    "confidence": "Paired bootstrap resamples map families and keeps all engine seeds for each map together.",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def pin(repo, expected=None):
    repo = repo.resolve()
    weight = campaign_eval.digest(repo / WEIGHT)
    if expected and weight != expected:
        raise ValueError(f"Frozen checkpoint changed: {repo}")
    receipt_path = repo / "export.json"
    if receipt_path.is_file():
        for name, checksum in read_json(receipt_path)["files"].items():
            if campaign_eval.digest(repo / name) != checksum:
                raise ValueError(f"Export receipt mismatch: {repo / name}")
    files = {str(path.relative_to(repo)): campaign_eval.digest(path)
             for path in sorted((repo / "submission").rglob("*.py"))}
    return dict(repo=str(repo), weights_sha256=weight, source_sha256=files)


def require_qa(output, manifest, qa_path):
    qa = read_json(qa_path or output / "map-qa.json")
    if not qa["approved"] or qa["manifests"].get(manifest.name) != campaign_eval.digest(manifest):
        raise ValueError(f"Manifest requires successful engine fixture QA: {manifest}")


def stall_intervals(trace, episode):
    waits, run, doors = [], 0, 0
    initial_keys = {int(key) for key, tic in episode["key_pickup_tics"].items() if tic == 0}
    keys = initial_keys
    count = 0
    with gzip.open(trace, "rt", encoding="utf-8") as stream:
        for count, line in enumerate(stream, start=1):
            frame = json.loads(line)
            held = set(frame["keys"])
            progress = (frame["new_area"] or held - keys or frame["doors"] > doors or frame["dealt"] > 0
                        or frame.get("true_exit", False))
            if progress:
                waits.append(run)
                run = 0
            else:
                run += 1
            keys, doors = held, frame["doors"]
    waits.append(run)
    if count == 0 or count != episode["decisions"]:
        raise ValueError(f"Incomplete decision trace: {trace}")
    # Campaign traces omit the exit flag; the observer still records true exit as progress.
    if episode["exit"] and frame.get("true_exit") is None and run:
        waits[-1] -= 1
    if max(waits) != episode["longest_no_progress"]:
        raise ValueError(f"Stall trace disagrees with the engine observer: {trace}")
    return sum(wait for wait in waits if wait >= STALL_DECISIONS) * FRAME_SKIP


def earliest_chain(episode):
    tics = episode.get("switch", {}).get("stages", {})
    end = tics.get("switch_activated")
    unlocks = [item for item in episode["unlocks"] if item.get("since_key") is not None]
    if end is None or not unlocks:
        return None
    unlock = min(unlocks, key=lambda item: item["tic"])
    key = unlock["tic"] - unlock["since_key"]
    if 0 < key <= unlock["tic"] <= end:
        return dict(key=key, unlock=unlock["tic"], switch_activation=end, exit=end)
    return None


def layout_id(row):
    return row.get("geometry_sha256") or identity(row)[:2]


def summarize(episodes):
    total = switch_eval.summarize(episodes)
    layouts = {layout_id(row) for row in episodes}
    switch_rows = [row for row in episodes if row["exit_type"] == "switch"]
    from_outside = [row for row in switch_rows if row["switch"]["stages"]["switch_near"] != 0]
    total.update(layouts=len(layouts), exits_from_outside_range=sum(row["exit"] for row in from_outside),
                 started_outside_range=len(from_outside), entered_range_from_outside=sum(reached(row, "switch_near") for row in from_outside),
                 first_stage_unreached=dict(sorted(group_counts(
                     row["switch"]["first_unreached_stage"] or "completed" for row in switch_rows).items())),
                 mean_closest_switch_units=switch_eval.mean(row["switch"]["minimum_distance"] for row in switch_rows
                                                           if row["switch"]["minimum_distance"] is not None),
                 mean_longest_stall_seconds=switch_eval.mean(row["longest_no_progress"] * FRAME_SKIP / 35 for row in episodes),
                 mean_total_stalled_seconds=switch_eval.mean(row["stalled_ticks"] / 35 for row in episodes if "stalled_ticks" in row),
                 ordered_key_unlock_switch_exits=sum(earliest_chain(row) is not None for row in switch_rows))
    total["first_tic_means"] = {stage: switch_eval.mean(row["switch"]["stages"][stage] for row in switch_rows
                                                       if reached(row, stage)) for stage in switch_eval.STAGES}
    return total


def group_counts(values):
    counts = defaultdict(int)
    for value in values:
        counts[value] += 1
    return counts


def groups(rows, key):
    parts = defaultdict(list)
    for row in rows:
        parts[key(row)].append(row)
    return {name: summarize(group) for name, group in sorted(parts.items())}


def panel_report(raw):
    return dict(overall=summarize(raw), by_family=groups(raw, lambda row: row["family"]),
                by_tier=groups(raw, lambda row: row["stage"]), by_cohort=groups(raw, switch_eval.cohort),
                by_recovery_stage=groups(raw, lambda row: f"{row['level']}/{switch_eval.cohort(row)}"),
                representative_traces=representatives([row for row in raw if row["exit_type"] != "original"]))


def decorated(raw, row, seed, trace, regression):
    if regression:
        raw = dict(raw, latency_decisions=raw.get("decisions", 0), agent_seconds=raw.get("measured_agent_seconds", 0.),
                   latency_p95_ms=None)
    return dict(raw, map=row["id"], path=row["path"], sector_map=row.get("map_name", "MAP01"),
                wad_sha256=row["sha256"], geometry_sha256=row.get("geometry_sha256"), stage=row["stage"],
                family=row["family"], level=row.get("level", 0), task=row["stage"] if regression else row["task"],
                exit_type="original" if regression else row["exit_type"], engine_seed=seed, policy_seed=0,
                post_lock=bool(row.get("post_lock")), trace=str(trace) if trace else None)


def run_panel(specs, pinned, seeds, output, manifest, *, regression=False):
    if pin(Path(pinned["repo"]), pinned["weights_sha256"]) != pinned:
        raise ValueError(f"DoomAgent source changed: {pinned['repo']}")
    expected = {(row["sha256"], row.get("map_name", "MAP01"), seed) for row in specs for seed in seeds}
    if len(expected) != len(specs) * len(seeds):
        raise ValueError("Duplicate paired WAD, map and engine seed")
    output_file = output / "episodes.jsonl"
    summary_file = output / "summary.json"
    source = campaign_eval.digest(manifest)
    if summary_file.is_file():
        receipt = read_json(summary_file)
        rows = read(output_file)
        if (receipt["policy"] != pinned or receipt["manifest_sha256"] != source or
                {identity(row) for row in rows} != expected or len(rows) != len(expected)):
            raise ValueError(f"Existing result does not match the requested panel: {output}")
        return rows
    output.mkdir(parents=True)
    rows = []
    print(f"{output.parent.name} | {output.name} | {len(specs)} layouts × {len(seeds)} seeds", flush=True)
    with campaign_eval.exported_agent(Path(pinned["repo"]), output / "agent.log", "cpu") as (agent, init_seconds):
        for index, spec in enumerate(specs):
            for seed in seeds:
                trace = output / f"map-{index:03d}-seed-{seed}.jsonl.gz"
                raw = (campaign_eval.play(spec, seed, agent, trace_path=trace) if regression
                       else switch_eval.play(spec, seed, agent, trace=trace))
                episode = decorated(raw, spec, seed, trace, regression)
                if raw.get("agent_error") or raw.get("engine_error") or raw.get("invalid_actions"):
                    campaign_eval.append_jsonl(output_file, episode)
                    raise RuntimeError(f"Evaluation failed on {spec['id']} seed {seed}: {raw.get('agent_error') or raw.get('engine_error')}")
                episode["stalled_ticks"] = stall_intervals(trace, episode)
                if not regression:
                    episode["ordered_key_unlock_switch_exit"] = earliest_chain(episode)
                campaign_eval.append_jsonl(output_file, episode)
                rows.append(episode)
        peak_rss = switch_eval.high_water_rss(switch_eval.agent_pid(agent))
    if pin(Path(pinned["repo"]), pinned["weights_sha256"]) != pinned or campaign_eval.digest(manifest) != source:
        raise ValueError("Evaluation input changed during execution")
    save(summary_file, dict(policy=pinned, manifest_sha256=source, engine_seeds=seeds, init_seconds=init_seconds,
                            peak_agent_rss_mib=peak_rss, method=METHOD, **panel_report(rows)))
    return rows


def scores(pairs, metric):
    layouts = defaultdict(list)
    candidate, reference = 0, 0
    for before, after in pairs:
        a, b = bool(metric(before)), bool(metric(after))
        layouts[layout_id(before)].append(int(b) - int(a))
        reference += a
        candidate += b
    if not layouts:
        raise ValueError("A paired gate requires at least one layout")
    return dict(candidate=candidate, reference=reference, episodes=len(pairs), layouts=len(layouts),
                gain_pp=100 * (candidate - reference) / len(pairs), improved_layouts=sum(sum(d) > 0 for d in layouts.values()),
                regressed_layouts=sum(sum(d) < 0 for d in layouts.values()), family_interval=bootstrap(pairs, metric))


def comparison(reference, candidate, thresholds):
    pairs = matching(reference, candidate)
    outside, walkovers = [], []
    for before, after in pairs:
        if before["exit_type"] == "walkover":
            walkovers.append((before, after))
            continue
        starts_near = before["switch"]["stages"]["switch_near"] == 0
        if starts_near != (after["switch"]["stages"]["switch_near"] == 0):
            raise ValueError("Two matched episodes disagree on whether the start was within USE range")
        if not starts_near:
            outside.append((before, after))
    exits = scores(outside, lambda row: row["exit"])
    reaches = scores(outside, lambda row: reached(row, "switch_near"))
    walk = scores(walkovers, lambda row: row["exit"])
    minimum = thresholds["min_improved_layouts"]
    passed = (exits["gain_pp"] >= thresholds["min_exit_gain_pp"] and exits["improved_layouts"] >= minimum and
              reaches["gain_pp"] >= thresholds["min_range_gain_pp"] and reaches["improved_layouts"] >= minimum and
              walk["gain_pp"] >= -thresholds["walkover_max_loss_pp"])
    return dict(exits=exits, range_entry=reaches, walkovers=walk, passed=passed)


def validation_decision(policies, protocol, milestones):
    seeds = protocol["training_seeds"]
    thresholds = protocol.get("promotion", protocol)
    candidates = []
    for milestone in milestones:
        seed_metrics = {}
        for seed in seeds:
            new = policies[f"curriculum-{seed}-{milestone}"]
            seed_metrics[str(seed)] = {
                "vs_frozen": comparison(policies["approach-baseline"], new, thresholds),
                "vs_control": comparison(policies[f"control-{seed}-{milestone}"], new, thresholds),
            }
        comparisons = [entry for pair in seed_metrics.values() for entry in pair.values()]
        worst_exit_gain = min(entry["exits"]["gain_pp"] for entry in comparisons)
        worst_range_gain = min(entry["range_entry"]["gain_pp"] for entry in comparisons)
        passed = all(entry["passed"] for entry in comparisons)
        rank = (passed, worst_exit_gain, worst_range_gain,
                sum(entry["exits"]["gain_pp"] for entry in comparisons), -milestone)
        candidates.append((rank, dict(added_decisions=milestone, seeds=seed_metrics, passed=passed)))
    selected = max(candidates, key=lambda row: row[0])[1]
    return selected, [entry for _, entry in candidates]


def retain_best(policies, milestones, training_seeds, thresholds):
    labels = [f"curriculum-{seed}-{budget}" for budget in milestones for seed in training_seeds]

    def rank(label):
        scores = comparison(policies["approach-baseline"], policies[label], thresholds)
        return tuple(scores[key]["gain_pp"] for key in ("exits", "range_entry", "walkovers"))

    return max(labels, key=rank)


def regression_archive(specs, seeds, pinned, paths):
    expected = {(row["sha256"], row.get("map_name", "MAP01"), seed): row for row in specs for seed in seeds}
    exact = []
    for path in paths:
        receipt = read_json(path / "summary.json")
        if receipt["checkpoint_sha256"] != pinned["weights_sha256"] or receipt["errors"]:
            raise ValueError(f"Archived baseline weight or outcome mismatch: {path}")
        for raw in read(path / "episodes.jsonl"):
            row = expected.get(identity(raw))
            if row is None:
                continue
            if (raw["policy_seed"] != 0 or raw["stage"] != row["stage"] or raw["family"] != row["family"] or
                    raw["tick_cap"] != row["ticks"] or raw.get("invalid_actions") or raw.get("agent_error") or raw.get("engine_error")):
                raise ValueError(f"Archived row does not use exact protocol: {raw['map']}")
            exact.append(decorated(raw, row, raw["engine_seed"], raw.get("trace"), True))
    if len(exact) != len(expected) or {identity(row) for row in exact} != set(expected):
        raise ValueError("Archived regression baselines do not have the required WAD and seed pairs")
    return exact


def assess(output, selection, protocol, qa_path):
    if not selection["validation_passed"]:
        raise ValueError("Validation gate failed; fresh assessment remains unopened")
    final = output / "maps/final.json"
    if campaign_eval.digest(final) != selection["final_manifest_sha256"]:
        raise ValueError("Frozen final manifest changed")
    require_qa(output, final, qa_path)
    rows = switch_eval.load_rows(final, "test")
    labels = ["approach-baseline", "exploration-baseline", *selection["selected_curriculum"]]
    policies = {label: run_panel(rows, selection["policies"][label], protocol["assessment_engine_seeds"],
                                 output / "assessment" / label, final) for label in labels}
    paired = {}
    for label in selection["selected_curriculum"]:
        for control in ("approach-baseline", "exploration-baseline"):
            paired[f"{label}_vs_{control}"] = compare(policies[control], policies[label])
    save(output / "assessment-report.json", dict(method=METHOD, policies={key: panel_report(raw) for key, raw in policies.items()}, paired=paired))


def regression(output, selection, protocol, manifest):
    rows = campaign_eval.read_rows(manifest, "test")
    seeds = protocol.get("regression_engine_seeds", [0, 1, 2, 3, 4])
    archives = {
        "approach-baseline": [LIMIT / part / "approach-41-12000" for part in ("regression", "regression-extra")],
        "exploration-baseline": [SKILL / "final-candidate"],
        "published": [SKILL / "final-reference"],
    }
    policies = {label: regression_archive(rows, seeds, selection["policies"][label], paths) for label, paths in archives.items()}
    for label in selection["selected_curriculum"] if selection["validation_passed"] else [selection["retained_curriculum"]]:
        policies[label] = run_panel(rows, selection["policies"][label], seeds, output / "regression" / label, manifest, regression=True)
    paired = {f"{label}_vs_{control}": compare(policies[control], policies[label])
              for label in policies.keys() - archives.keys() for control in archives}
    save(output / "regression-report.json", dict(method=METHOD, engine_seeds=seeds, manifest_sha256=campaign_eval.digest(manifest),
                                                archived_baselines={key: list(map(str, paths)) for key, paths in archives.items()},
                                                policies={key: panel_report(raw) for key, raw in policies.items()}, paired=paired))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("baseline", "screen", "assess", "regression"))
    parser.add_argument("--out", type=Path, default=ROOT / "runs/exit-recovery-study")
    parser.add_argument("--wait", type=int, default=0, help="seconds to wait for pending policy exports")
    parser.add_argument("--milestones", type=int, nargs="+", help="validation-only subset of preregistered milestones")
    parser.add_argument("--stop-reason", help="close at completed partial milestones after a recorded fail-fast stop")
    parser.add_argument("--qa", type=Path, help="alternative fixture QA receipt")
    parser.add_argument("--regression-manifest", type=Path, default=LIMIT / "original-120-panel.json")
    args = parser.parse_args()
    output = args.out.resolve()
    protocol = read_json(output / "protocol.json")
    if args.phase == "assess":
        assess(output, read_json(output / "selection.json"), protocol, args.qa)
        return
    if args.phase == "regression":
        regression(output, read_json(output / "selection.json"), protocol, args.regression_manifest)
        return
    manifest = output / "maps/manifest.json"
    require_qa(output, manifest, args.qa)
    rows = switch_eval.load_rows(manifest, "validation")
    seeds = protocol["validation_engine_seeds"]
    pins = {label: pin(*artifact) for label, artifact in BASELINES.items()}
    raw = {"approach-baseline": run_panel(rows, pins["approach-baseline"], seeds,
                                          output / "validation/approach-baseline", manifest)}
    if args.phase == "baseline":
        return
    expected = sorted(protocol["screen_decisions"])
    milestones = sorted(set(args.milestones or expected))
    if not set(milestones) <= set(expected):
        parser.error("milestones must be preregistered in the study protocol")
    if args.stop_reason and milestones == expected:
        parser.error("a fail-fast closure requires a partial milestone set")
    pending = [f"{arm}-{seed}-{budget}" for budget in milestones for seed in protocol["training_seeds"]
               for arm in ("control", "curriculum")]
    deadline = time.monotonic() + args.wait
    while pending:
        for label in pending.copy():
            repo = output / "exports" / label
            if (repo / "export.json").is_file():
                pins[label] = pin(repo)
                raw[label] = run_panel(rows, pins[label], seeds, output / "validation" / label, manifest)
                pending.remove(label)
        if pending:
            if time.monotonic() >= deadline:
                raise FileNotFoundError(f"Validation exports are pending: {pending}")
            time.sleep(3)
    selected, comparisons = validation_decision(raw, protocol, milestones)
    payload = dict(method=METHOD, candidate_milestones=comparisons, best_common_milestone=selected,
                   policies={key: panel_report(records) for key, records in raw.items()})
    if milestones != expected and not args.stop_reason:
        save(output / ("validation-" + "-".join(map(str, milestones)) + ".json"), payload)
        print(json.dumps(dict(midpoint=selected["added_decisions"], pass_at_tested_budget=selected["passed"])), flush=True)
        return
    if args.stop_reason:
        payload["stop_reason"] = args.stop_reason
    save(output / "validation-report.json", payload)
    milestone = selected["added_decisions"]
    selected_labels = [f"curriculum-{seed}-{milestone}" for seed in protocol["training_seeds"]]
    passed = selected["passed"] and not args.stop_reason
    retained = retain_best(raw, [milestone] if passed else milestones, protocol["training_seeds"],
                           protocol.get("promotion", protocol))
    final_manifest = output / "maps/final.json"
    freeze_labels = [*BASELINES, *selected_labels, retained, *(f"control-{seed}-{milestone}" for seed in protocol["training_seeds"])]
    save(output / "selection.json", dict(validation_passed=passed, added_decisions=milestone, stop_reason=args.stop_reason,
         selected_curriculum=selected_labels, retained_curriculum=retained,
         final_manifest_sha256=campaign_eval.digest(final_manifest) if passed else None,
         policies={key: pins[key] for key in freeze_labels}))
    print(json.dumps(dict(validation_passed=passed, added_decisions=milestone, retained_curriculum=retained)), flush=True)


if __name__ == "__main__":
    main()
