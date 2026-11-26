"""Run pinned development evaluations and fail-fast gates for the adaptive campaign."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from scripts import campaign_eval, campaign_report, exit_recovery_eval as recovery
from scripts.exit_switch_eval import load_rows
from scripts.exit_switch_report import matching, reached


ROOT = recovery.ROOT
DEFAULT = ROOT / "runs/adaptive-campaign-20261127"
PROTOCOL_PATH = ROOT / "configs/adaptive-campaign-eval.json"
RECOVERY = ROOT / "runs/exit-recovery-study"
APPROACH = "approach-baseline"
FINAL_MANIFEST = RECOVERY / "maps/final.json"
FINAL_MANIFEST_SHA256 = "c51e72efb2223ab716a03145bd2ba4955201c22e38559b53d892b5593ba74ffc"
FINAL_SEEDS = [11, 23, 37]
ARCHIVES = {
    APPROACH: [recovery.LIMIT / part / "approach-41-12000" for part in ("regression", "regression-extra")],
    "exploration-baseline": [recovery.SKILL / "final-candidate"],
    "published": [recovery.SKILL / "final-reference"],
}


def protocol(output):
    destination = output / "evaluation-protocol.json"
    settings = recovery.read_json(destination)
    if settings["version"] != 1:
        raise ValueError("Unsupported evaluation protocol version")
    return settings


def panels(settings):
    found = {}
    for name, spec in settings["development"].items():
        manifest = ROOT / spec["manifest"]
        if campaign_eval.digest(manifest) != spec["sha256"]:
            raise ValueError(f"Preregistered manifest changed: {manifest}")
        if name == "switch":
            recovery.require_qa(RECOVERY, manifest, None)
            rows = load_rows(manifest, spec["split"])
        else:
            rows = campaign_eval.read_rows(manifest, spec["split"])
        if len(rows) != spec["layout_count"] or len({recovery.layout_id(row) for row in rows}) != len(rows):
            raise ValueError(f"Expected distinct preregistered development layouts on {name}")
        found[name] = manifest, rows, spec["engine_seeds"]
    return found


def expected_pairs(panel):
    _, rows, seeds = panel
    return {(row["sha256"], row.get("map_name", "MAP01"), seed): row for row in rows for seed in seeds}


def check_pairs(records, panel):
    expected = expected_pairs(panel)
    observed = {campaign_report.identity(row): row for row in records}
    if len(observed) != len(records) or observed.keys() != expected.keys():
        raise ValueError("Expected one episode per pinned layout and engine seed")
    for key, row in observed.items():
        spec = expected[key]
        if (row["policy_seed"] != 0 or row["family"] != spec["family"] or row["stage"] != spec["stage"]
                or row.get("tick_cap", spec["ticks"]) != spec["ticks"] or row["ticks"] > spec["ticks"]
                or row["geometry_sha256"] != spec["geometry_sha256"]):
            raise ValueError(f"Episode does not match preregistered development metadata: {key}")


def archived(panels_by_name):
    pins = {name: recovery.pin(*reference) for name, reference in recovery.BASELINES.items()}
    switch_folder = RECOVERY / "validation" / APPROACH
    switch_manifest, _, switch_seeds = panels_by_name["switch"]
    receipt = recovery.read_json(switch_folder / "summary.json")
    if (receipt["policy"] != pins[APPROACH] or receipt["manifest_sha256"] != campaign_eval.digest(switch_manifest)
            or receipt["engine_seeds"] != switch_seeds or receipt["overall"]["errors"]):
        raise ValueError("Archived switch baseline does not match the current development protocol")
    switch = campaign_report.read(switch_folder / "episodes.jsonl")
    check_pairs(switch, panels_by_name["switch"])
    _, familiar_rows, familiar_seeds = panels_by_name["familiar"]
    for name, sources in ARCHIVES.items():
        for path in sources:
            source_receipt = recovery.read_json(path / "summary.json")
            if source_receipt["repo"] != pins[name]["repo"] or source_receipt["device"] != "cpu":
                raise ValueError(f"Archived familiar results identify a different repo or device: {path}")
    familiar = {name: recovery.regression_archive(familiar_rows, familiar_seeds, pins[name], sources)
                for name, sources in ARCHIVES.items()}
    for records in familiar.values():
        check_pairs(records, panels_by_name["familiar"])
    return pins, switch, familiar


def is_bad(row):
    return any(row.get(key) for key in ("agent_error", "engine_error", "invalid_actions", "crashes", "inference_timeouts"))


def exit_metric(row):
    return bool(row["exit"])


def range_metric(row):
    return reached(row, "switch_near")


def death_metric(row):
    return bool(row["dead"])


def evaluate_scope(output, name, label, repo, panel):
    manifest, rows, seeds = panel
    pinned = recovery.pin(repo)
    records = recovery.run_panel(rows, pinned, seeds, output / "evaluation" / name / label,
                                 manifest, regression=name == "familiar")
    check_pairs(records, panel)
    return records


def frozen_scope(output, name, label, panel):
    folder = output / "evaluation" / name / label
    receipt = recovery.read_json(folder / "summary.json")
    pin = receipt["policy"]
    if (recovery.pin(Path(pin["repo"]), pin["weights_sha256"]) != pin
            or receipt["manifest_sha256"] != campaign_eval.digest(panel[0]) or receipt["engine_seeds"] != panel[2]):
        raise ValueError(f"Evaluation artifact or protocol changed for {label} on {name}")
    records = campaign_report.read(folder / "episodes.jsonl")
    check_pairs(records, panel)
    return pin, records


def await_scopes(output, labels, wait_seconds):
    deadline = time.monotonic() + wait_seconds
    pending = [output / "evaluation" / scope / label / "summary.json"
               for label in labels for scope in ("switch", "familiar")]
    while pending:
        for path in pending.copy():
            try:
                recovery.read_json(path)
                pending.remove(path)
            except (FileNotFoundError, json.JSONDecodeError):
                continue
        if pending:
            if time.monotonic() >= deadline:
                raise FileNotFoundError(f"Development scopes pending: {', '.join(str(path) for path in pending)}")
            time.sleep(2)


def blocks(reference_switch, switch, reference_familiar, familiar):
    chunks = defaultdict(list)
    for previous, updated in matching(reference_switch, switch):
        if previous["exit_type"] == "walkover":
            group = "walkover"
        elif previous["switch"]["stages"]["switch_near"] == 0:
            group = "alignment"
        else:
            group = "switch_outside"
        if (previous["switch"]["stages"]["switch_near"] == 0) != (updated["switch"]["stages"]["switch_near"] == 0):
            raise ValueError("Paired switch episodes disagree on initial range")
        chunks[group].append((previous, updated))
    for previous, updated in matching(reference_familiar, familiar):
        chunks["familiar"].append((previous, updated))
        chunks[f"familiar_{previous['stage']}"].append((previous, updated))
    return chunks


def compare(reference_switch, switch, reference_familiar, familiar):
    cohorts = blocks(reference_switch, switch, reference_familiar, familiar)
    contrasts = {name: {"exit": recovery.scores(pairs, exit_metric), "death": recovery.scores(pairs, death_metric)}
                 for name, pairs in sorted(cohorts.items())}
    contrasts["switch_outside"]["range_entry"] = recovery.scores(cohorts["switch_outside"], range_metric)
    return contrasts


def acquired_key(row):
    return any(tic > 0 for tic in row["key_pickup_tics"].values())


def unlocked_after_key(row):
    return any(event["since_key"] is not None and 0 < event["tic"] - event["since_key"] <= event["tic"] <= row["ticks"]
               for event in row["unlocks"])


def tier_progression(reference, candidate):
    by_tier = defaultdict(list)
    for left, right in matching(reference, candidate):
        by_tier[left["stage"]].append((left, right))
    result = {}
    for tier, pairs in sorted(by_tier.items()):
        def totals(index):
            values = [row[index] for row in pairs]
            return dict(mean_damage=sum(row["damage"] for row in values) / len(values),
                        ammunition_decrease_lower_bound=sum(row["work"]["rounds_spent_lower_bound"] for row in values),
                        kills=sum(row["kills"] for row in values),
                        attack_without_visible_enemy=sum(row["work"]["attack_without_visible_enemy"] for row in values))

        result[tier] = dict(key_pickup=recovery.scores(pairs, acquired_key),
                            unlock_after_matching_key=recovery.scores(pairs, unlocked_after_key),
                            ordered_key_unlock_exit=recovery.scores(pairs, lambda row: row["exit"] and unlocked_after_key(row)),
                            survived=recovery.scores(pairs, lambda row: not row["dead"] and not is_bad(row)),
                            reference_resources=totals(0), candidate_resources=totals(1))
    return result


def contrast(output, pairs, name):
    settings = protocol(output)
    cohorts = panels(settings)
    _, switch_reference, familiar_references = archived(cohorts)
    records = {APPROACH: (switch_reference, familiar_references[APPROACH])}
    for label in {label for pair in pairs for label in pair} - records.keys():
        switch_pin, switch = frozen_scope(output, "switch", label, cohorts["switch"])
        familiar_pin, familiar = frozen_scope(output, "familiar", label, cohorts["familiar"])
        if switch_pin != familiar_pin:
            raise ValueError(f"Development scopes disagree on exact policy for {label}")
        records[label] = (switch, familiar)
    result = {}
    for reference, candidate in pairs:
        left_switch, left_familiar = records[reference]
        right_switch, right_familiar = records[candidate]
        result[f"{candidate}_vs_{reference}"] = dict(
            outcomes=compare(left_switch, right_switch, left_familiar, right_familiar),
            familiar_progression=tier_progression(left_familiar, right_familiar))
    destination = output / "evaluation" / f"contrast-{name}.json"
    recovery.save(destination, dict(comparisons=result,
                                   unlock="A door opening with a matching previously acquired key. The final exit uses engine completion.",
                                   ammunition="A sum of observed inventory decreases. Pickups can hide spending and actual shot counts are unknown."))
    print(f"Completed {len(pairs)} paired contrasts at {destination}", flush=True)


def gate_metrics(contrast, switch, familiar, thresholds, *, late):
    switch_range = contrast["switch_outside"]["range_entry"]
    switch_exit = contrast["switch_outside"]["exit"]
    limits = {
        "range_entries": (switch_range["candidate"], thresholds["switch_range_entries"]),
        "range_layout_gains": (switch_range["improved_layouts"], thresholds["switch_range_improved_layouts"]),
        "alignment_exits": (contrast["alignment"]["exit"]["candidate"], thresholds["alignment_exits"]),
        "walkover_exits": (contrast["walkover"]["exit"]["candidate"], thresholds["walkover_exits"]),
        "familiar_exits": (contrast["familiar"]["exit"]["candidate"], thresholds["familiar_exits"]),
    }
    if late:
        limits["true_switch_exits"] = (switch_exit["candidate"], thresholds["late_switch_exits"])
        limits["true_switch_exit_layout_gains"] = (switch_exit["improved_layouts"], thresholds["late_switch_exit_improved_layouts"])
    checks = {name: dict(observed=actual, minimum=minimum, passed=actual >= minimum)
              for name, (actual, minimum) in limits.items()}
    deaths = contrast["familiar_gauntlet"]["death"]["candidate"]
    max_deaths = thresholds["gauntlet_deaths_max"]
    checks["gauntlet_deaths"] = dict(observed=deaths, maximum=max_deaths, passed=deaths <= max_deaths)
    errors = sum(is_bad(row) for row in (*switch, *familiar))
    checks["invalid_actions_or_errors"] = dict(observed=errors, maximum=0, passed=errors == 0)
    return dict(passed=all(check["passed"] for check in checks.values()), checks=checks)


def local_score(row, weight, spec):
    if is_bad(row):
        return None
    terms = dict(exit=int(row["exit"]), key=row["keys"], door=row["doors"], lock=row["locks"], kill=row["kills"],
                 unique_cell=row["unique_cells"], health=row["health"], damage=row["damage"], death=int(row["dead"]),
                 exit_fraction_of_tic_cap_remaining=int(row["exit"]) * max(0, 1 - row["ticks"] / spec["ticks"]))
    return sum(weight[name] * value for name, value in terms.items())


def report(settings, panels_by_name, switch, familiar):
    proxy = {}
    for name, rows in (("switch", switch), ("familiar", familiar)):
        sequence = expected_pairs(panels_by_name[name])
        values = [local_score(row, settings["local_proxy"], sequence[campaign_report.identity(row)]) for row in rows]
        valid = [value for value in values if value is not None]
        proxy[name] = dict(mean=sum(valid) / len(valid) if valid else None, episodes=len(valid))
    return dict(switch=recovery.panel_report(switch), familiar=recovery.panel_report(familiar),
                per_map=dict(switch=recovery.groups(switch, lambda row: row["map"]),
                             familiar=recovery.groups(familiar, lambda row: row["map"])),
                familiar_combat_and_progression={stage: campaign_report.summarize([row for row in familiar if row["stage"] == stage])
                                                 for stage in campaign_eval.TIER}, local_proxy=proxy)


def check_archived_totals(switch, familiar):
    base = compare(switch, switch, familiar, familiar)
    declared = [("switch_outside", "range_entry", 21, 54), ("switch_outside", "exit", 18, 54),
                ("alignment", "exit", 12, 18), ("walkover", "exit", 9, 24),
                ("familiar", "exit", 30, 120), ("familiar_gauntlet", "death", 20, 30)]
    for cohort, metric, count, total in declared:
        if (base[cohort][metric]["reference"], base[cohort][metric]["episodes"]) != (count, total):
            raise ValueError(f"Preregistered baseline does not reproduce: {cohort}/{metric}")


def earlier_switch_panel(baseline, developmental, development_panel):
    root = recovery.LIMIT / "assessment/approach-41-12000"
    manifest = recovery.LIMIT / "maps/manifest.json"
    receipt = recovery.read_json(root / "summary.json")
    if (receipt["checkpoint_sha256"] != baseline["weights_sha256"] or receipt["repo"] != baseline["repo"]
            or receipt["manifest_sha256"] != campaign_eval.digest(manifest) or receipt["errors"]):
        raise ValueError("Earlier fresh switch evidence identifies a different artifact or manifest")
    old_specs = {row["sha256"]: row for row in load_rows(manifest, "test", tasks=("conditional",))
                 if row["exit_type"] == "switch"}
    old_rows = [row for row in campaign_report.read(root / "episodes.jsonl")
                if row["task"] == "conditional" and row["exit_type"] == "switch"]
    old_pairs = {campaign_report.identity(row) for row in old_rows}
    specified = {(row["sha256"], row["map_name"], seed) for row in old_specs.values() for seed in receipt["seeds"]}
    if (old_pairs != specified or len(old_rows) != 54 or any(is_bad(row) or row["policy_seed"] != 0 for row in old_rows)
            or any(row["exit"] or reached(row, "switch_near") for row in old_rows)):
        raise ValueError("Earlier fresh switch result does not reproduce 0/54 on the archived protocol")
    out_of_range = [row for row in developmental if row["exit_type"] == "switch"
                    and row["switch"]["stages"]["switch_near"] != 0]
    current_layouts = {row["geometry_sha256"] for row in out_of_range}
    earlier_layouts = {row["geometry_sha256"] for row in old_specs.values()}
    current_wads = {row["wad_sha256"] for row in out_of_range}
    if current_layouts & earlier_layouts or current_wads & old_specs.keys():
        raise ValueError("Earlier fresh and current diagnostic switch cohorts share WADs or geometry")
    return dict(weights_sha256=baseline["weights_sha256"], earlier=dict(manifest=str(manifest),
                manifest_sha256=receipt["manifest_sha256"], independent_layouts=len(earlier_layouts),
                engine_seeds=receipt["seeds"], range_entries=0, exits=0, episodes=len(old_rows)),
                current=dict(manifest=str(development_panel[0]), manifest_sha256=campaign_eval.digest(development_panel[0]),
                independent_layouts=len(current_layouts), engine_seeds=development_panel[2],
                range_entries=sum(reached(row, "switch_near") for row in out_of_range),
                exits=sum(row["exit"] for row in out_of_range), episodes=len(out_of_range)), shared_geometry=0, shared_wads=0)


def initialize(output):
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "evaluation-protocol.json"
    if not destination.is_file():
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if campaign_eval.digest(destination) != campaign_eval.digest(PROTOCOL_PATH):
        raise ValueError("Campaign evaluation protocol already exists with different settings")
    settings = protocol(output)
    cohorts = panels(settings)
    pins, switch, familiar = archived(cohorts)
    check_archived_totals(switch, familiar[APPROACH])
    evidence = {name: campaign_report.panel(records, {})["reference"] for name, records in familiar.items()}
    payload = dict(protocol_sha256=campaign_eval.digest(destination), frozen=pins, archives={"switch": str(RECOVERY / "validation" / APPROACH),
                   "familiar": {name: list(map(str, source)) for name, source in ARCHIVES.items()}},
                   approach=report(settings, cohorts, switch, familiar[APPROACH]), familiar_baselines=evidence)
    receipt = output / "development-baselines.json"
    if not receipt.exists():
        recovery.save(receipt, payload)
    distinction = output / "historical-switch-cohorts.json"
    if not distinction.exists():
        recovery.save(distinction, earlier_switch_panel(pins[APPROACH], switch, cohorts["switch"]))
    practice_receipt = output / "phase2-partitions.json"
    plan_path = output / "mixed-plan.json"
    if plan_path.is_file() and not practice_receipt.exists():
        from training.maps import load_verified_maps

        source = recovery.read_json(plan_path)["inputs"]["training"]
        training_manifest = Path(source["path"])
        if campaign_eval.digest(training_manifest) != source["sha256"]:
            raise ValueError("Mixed training manifest changed")
        train = load_verified_maps(training_manifest.parent)
        audit = {}
        for name, (manifest, rows, seeds) in cohorts.items():
            shared = {field: sorted({row[field] for row in train} & {row[field] for row in rows})
                      for field in ("sha256", "geometry_sha256", "family")}
            if any(shared.values()):
                raise ValueError(f"Mixed training overlaps {name}")
            audit[name] = dict(manifest=str(manifest), manifest_sha256=campaign_eval.digest(manifest),
                               layouts=len(rows), engine_seeds=seeds, train_overlap=shared)
        recovery.save(practice_receipt, dict(training=source, training_layouts=len(train), evaluations=audit,
                                              reserved_panel_access="No reserved manifest or WAD opened by this audit"))
    print(f"Pinned {len(switch)} switch and {len(familiar[APPROACH])} familiar episodes for {APPROACH}", flush=True)


def publish_policy(output, label, pinned, switch, familiar, switch_reference, familiar_reference):
    if not (output / "clock.json").is_file() or not (output / "status.json").is_file():
        return
    from scripts.adaptive_supervisor import record

    paired = compare(switch_reference, switch, familiar_reference, familiar)
    values = {
        "true_switch_exits": paired["switch_outside"]["exit"],
        "switch_range_entries": paired["switch_outside"]["range_entry"],
        "near_switch_exits": paired["alignment"]["exit"],
        "walkover_exits": paired["walkover"]["exit"],
        "familiar_exits": paired["familiar"]["exit"],
        "gauntlet_deaths": paired["familiar_gauntlet"]["death"],
    }
    snapshot = {key: {name: metric[name] for name in ("candidate", "reference", "episodes", "layouts")}
                for key, metric in values.items()}
    results = ", ".join(f"{key.replace('_', ' ')} {value['candidate']}/{value['episodes']}"
                        for key, value in snapshot.items())
    export = recovery.read_json(Path(pinned["repo"]) / "export.json")
    record(output, message=f"Completed paired development evaluation for {label}. {results}.",
           status={"latest_validation": dict(policy=label, trained_steps=export["trained_steps"],
                                             weights_sha256=pinned["weights_sha256"], **snapshot)})


def sync(output, labels):
    settings = protocol(output)
    cohorts = panels(settings)
    _, switch_reference, familiar_references = archived(cohorts)
    for label in labels:
        switch_pin, switch = frozen_scope(output, "switch", label, cohorts["switch"])
        familiar_pin, familiar = frozen_scope(output, "familiar", label, cohorts["familiar"])
        if switch_pin != familiar_pin:
            raise ValueError(f"Development scopes disagree on exact policy for {label}")
        destination = output / "evaluation" / f"{label}.json"
        if not destination.is_file():
            recovery.save(destination, report(settings, cohorts, switch, familiar))
        publish_policy(output, label, switch_pin, switch, familiar, switch_reference, familiar_references[APPROACH])


def run(output, artifacts, wait, scope, *, publish_status=True):
    settings = protocol(output)
    cohorts = panels(settings)
    receipt = recovery.read_json(output / "development-baselines.json")
    if receipt["protocol_sha256"] != campaign_eval.digest(output / "evaluation-protocol.json"):
        raise ValueError("Preregistered evaluation configuration changed")
    _, switch_reference, familiar_references = archived(cohorts)
    for label, repo in artifacts:
        deadline = time.monotonic() + wait
        while not (repo / "export.json").is_file():
            if time.monotonic() >= deadline:
                raise FileNotFoundError(f"Frozen export is pending: {repo}")
            time.sleep(2)
        started = time.monotonic()
        if scope:
            records = evaluate_scope(output, scope, label, repo, cohorts[scope])
            print(f"{label} | {scope} {sum(row['exit'] for row in records)}/{len(records)} | "
                  f"{time.monotonic() - started:.1f} seconds", flush=True)
            continue
        switch = evaluate_scope(output, "switch", label, repo, cohorts["switch"])
        familiar = evaluate_scope(output, "familiar", label, repo, cohorts["familiar"])
        result = report(settings, cohorts, switch, familiar)
        result["invocation_seconds"] = time.monotonic() - started
        destination = output / "evaluation" / f"{label}.json"
        if not destination.is_file():
            recovery.save(destination, result)
        if publish_status:
            publish_policy(output, label, recovery.pin(repo), switch, familiar, switch_reference, familiar_references[APPROACH])
        print(f"{label} | switch {result['switch']['overall']['exits']}/{len(switch)} | "
              f"familiar {result['familiar']['overall']['exits']}/{len(familiar)} | {result['invocation_seconds']:.1f} seconds", flush=True)


def decide(output, labels, *, milestone=None, late=False, name=None, publish_status=True):
    settings = protocol(output)
    cohorts = panels(settings)
    pins, reference_switch, familiar_baselines = archived(cohorts)
    reference_familiar = familiar_baselines[APPROACH]
    check_archived_totals(reference_switch, reference_familiar)
    records = {}
    for label in labels:
        switch_identity, switch = frozen_scope(output, "switch", label, cohorts["switch"])
        familiar_identity, familiar = frozen_scope(output, "familiar", label, cohorts["familiar"])
        if switch_identity != familiar_identity:
            raise ValueError(f"Evaluation panels used different policy artifacts for {label}")
        pins[label], records[label] = switch_identity, (switch, familiar)
    results = {}
    for label, (switch, familiar) in records.items():
        reference = compare(reference_switch, switch, reference_familiar, familiar)
        export = recovery.read_json(Path(pins[label]["repo"]) / "export.json")
        training_config = export.get("training_config") or {}
        seed = training_config.get("seed", export.get("training_seed"))
        numbered = Path(export["source_checkpoint"])
        if campaign_eval.digest(numbered) != export["source_checkpoint_sha256"]:
            raise ValueError(f"Numbered checkpoint identity changed for {label}")
        if milestone is not None and (seed not in settings["training_seeds"] or not label.endswith(f"-{seed}-{milestone}")
                                      or export["trained_steps"] != 314496 + milestone):
            raise ValueError(f"Matched export seed or decision milestone does not match its label: {label}")
        results[label] = dict(identity=pins[label], training_seed=seed, source_checkpoint=str(numbered),
                              source_checkpoint_sha256=export["source_checkpoint_sha256"], trained_steps=export["trained_steps"],
                              updates=export["updates"], vs_approach=reference,
                              gate=gate_metrics(reference, switch, familiar, settings["thresholds"], late=late),
                              development=report(settings, cohorts, switch, familiar))
    if milestone:
        eligible = [f"mixed-{seed}-{milestone}" for seed in settings["training_seeds"]]
        for seed in settings["training_seeds"]:
            mixed, sequential = f"mixed-{seed}-{milestone}", f"sequential-{seed}-{milestone}"
            left_switch, left_familiar = records[sequential]
            right_switch, right_familiar = records[mixed]
            results[mixed]["vs_sequential"] = compare(left_switch, right_switch, left_familiar, right_familiar)
    else:
        eligible = labels
    qualified = {label: results[label]["gate"]["passed"] for label in eligible}
    training_seeds = {results[label]["training_seed"] for label in eligible} - {None}
    passed = len(training_seeds) >= 2 and all(qualified.values())
    destination = output / "evaluation" / (f"gate-{milestone}.json" if milestone else f"gate-{name}.json")
    if not destination.exists():
        recovery.save(destination, dict(passed=passed, late=late, decisions=milestone, seed_replication_required=2,
                                       qualified=qualified, independent_training_seeds=sorted(training_seeds), results=results,
                                       protocol_sha256=campaign_eval.digest(output / "evaluation-protocol.json")))
    if milestone == settings["midpoint_decisions"]:
        receipt = output / "evaluation/midpoint-gate.json"
        body = dict(continue_matched_experiment=passed, qualified=qualified,
                    gate_report=str(destination), gate_report_sha256=campaign_eval.digest(destination))
        if receipt.exists():
            if recovery.read_json(receipt) != body:
                raise ValueError("Saved midpoint gate does not match the current evidence")
        else:
            temporary = receipt.with_suffix(".json.tmp")
            recovery.save(temporary, body)
            temporary.replace(receipt)
    if publish_status and (output / "clock.json").is_file() and (output / "status.json").is_file():
        from scripts.adaptive_supervisor import record

        verdict = "passed" if passed else "failed"
        record(output, message=f"Development gate {milestone or name} {verdict}. Qualification by policy {qualified}.",
               status={"latest_validation_gate": dict(name=milestone or name, passed=passed, path=str(destination))})
    print(json.dumps(dict(passed=passed, late=late, qualified=qualified, evidence=str(destination))), flush=True)


def final_rank(row):
    tiers = row["development"]["familiar_combat_and_progression"]
    paired = row["vs_approach"]
    return (tiers["gauntlet"]["exits"], tiers["keys"]["key_unlock_and_exit"], tiers["keys"]["unlock_episodes"],
            paired["switch_outside"]["exit"]["candidate"], paired["switch_outside"]["range_entry"]["candidate"],
            paired["familiar"]["exit"]["candidate"], -tiers["gauntlet"]["deaths"])


def verify_gate(output, path):
    receipt = recovery.read_json(path)
    if (not receipt["passed"] or not receipt["late"] or len(receipt["independent_training_seeds"]) < 2
            or receipt["protocol_sha256"] != campaign_eval.digest(output / "evaluation-protocol.json")):
        raise ValueError("Final panel is sealed until a preregistered late gate passes in two training seeds")
    return receipt


def freeze(output, gate):
    receipt = verify_gate(output, gate)
    labels = [label for label, passed in receipt["qualified"].items() if passed]
    label = max(labels, key=lambda item: final_rank(receipt["results"][item]))
    best = receipt["results"][label]["identity"]
    if recovery.pin(Path(best["repo"]), best["weights_sha256"]) != best:
        raise ValueError("Freeze requires the exact checked DoomAgent artifact")
    export = recovery.read_json(Path(best["repo"]) / "export.json")
    checkpoint = Path(export["source_checkpoint"])
    checkpoint_sha256 = export["source_checkpoint_sha256"]
    if campaign_eval.digest(checkpoint) != checkpoint_sha256:
        raise ValueError("The candidate numbered checkpoint changed since export")
    policies = {name: recovery.pin(*artifact) for name, artifact in recovery.BASELINES.items()}
    policies[label] = best
    selection = dict(selected=label, policies=policies, gate=str(gate), gate_sha256=campaign_eval.digest(gate),
                     selected_checkpoint=str(checkpoint), selected_checkpoint_sha256=checkpoint_sha256,
                     final_manifest_sha256=FINAL_MANIFEST_SHA256, engine_seeds=FINAL_SEEDS,
                     rank=["gauntlet exits", "key unlock and exit", "unlock episodes", "switch exits",
                           "range entries", "familiar exits", "fewer gauntlet deaths"])
    recovery.save(output / "final-selection.json", selection)
    print(f"Frozen {label}. The final panel is eligible.", flush=True)


def final_panel(output, labels):
    selection_path = output / "final-selection.json"
    if not selection_path.is_file():
        raise ValueError("The 48-layout final panel requires a candidate frozen from a passing late gate")
    selection = recovery.read_json(selection_path)
    gate = Path(selection["gate"])
    if selection["gate_sha256"] != campaign_eval.digest(gate):
        raise ValueError("Qualification evidence changed. The final panel stays sealed.")
    verify_gate(output, gate)
    if campaign_eval.digest(selection["selected_checkpoint"]) != selection["selected_checkpoint_sha256"]:
        raise ValueError("Frozen numbered checkpoint changed. The final panel stays sealed.")
    if selection["final_manifest_sha256"] != campaign_eval.digest(FINAL_MANIFEST):
        raise ValueError("Reserved final manifest changed")
    recovery.require_qa(RECOVERY, FINAL_MANIFEST, None)
    layout_rows = load_rows(FINAL_MANIFEST, "test")
    if len(layout_rows) != 48 or len({recovery.layout_id(row) for row in layout_rows}) != 48:
        raise ValueError("Expected all 48 independent final layouts")
    pins = selection["policies"]
    episodes = {}
    cohort = (FINAL_MANIFEST, layout_rows, selection["engine_seeds"])
    for label in labels or pins:
        repo = Path(pins[label]["repo"])
        if recovery.pin(repo, pins[label]["weights_sha256"]) != pins[label]:
            raise ValueError(f"Frozen final source or weights changed: {label}")
        episodes[label] = evaluate_scope(output, "final", label, repo, cohort)
    if labels:
        return
    from scripts.exit_switch_report import compare as final_compare

    chosen = selection["selected"]
    paired = {name: final_compare(records, episodes[chosen]) for name, records in episodes.items() if name != chosen}
    reports = {name: recovery.panel_report(records) for name, records in episodes.items()}
    recovery.save(output / "final-assessment.json", dict(selection=selection, panels=reports, paired=paired,
                                                          independent_layouts=len(layout_rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "sync", "gate", "contrast", "freeze", "assess"))
    parser.add_argument("--out", default=DEFAULT, type=Path)
    parser.add_argument("--artifact", action="append", default=[], metavar="LABEL=REPO")
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--wait", type=int, default=0)
    parser.add_argument("--scope", choices=("switch", "familiar"), help="Evaluate one scope in an isolated process")
    parser.add_argument("--milestone", type=int, choices=(12000, 25000))
    parser.add_argument("--gate-name")
    parser.add_argument("--pair", action="append", default=[], metavar="REFERENCE=CANDIDATE")
    parser.add_argument("--gate-file", type=Path)
    parser.add_argument("--late", action="store_true")
    args = parser.parse_args()
    output = args.out.resolve()
    if args.phase == "prepare":
        initialize(output)
    elif args.phase == "run":
        artifacts = [value.split("=", 1) for value in args.artifact]
        artifacts = [(label, Path(path).resolve()) for label, path in artifacts]
        artifacts.extend((label, output / "exports" / label) for label in args.label)
        if not artifacts and args.milestone:
            for seed in protocol(output)["training_seeds"]:
                for arm in ("sequential", "mixed"):
                    label = f"{arm}-{seed}-{args.milestone}"
                    artifacts.append((label, output / "exports" / label))
        if not artifacts or any(not label or Path(label).name != label or label in {".", ".."} for label, _ in artifacts):
            parser.error("Specify exports using safe unique labels")
        if len({label for label, _ in artifacts}) != len(artifacts):
            parser.error("Specify unique export labels")
        run(output, artifacts, args.wait, args.scope)
    elif args.phase == "sync":
        if not args.label:
            parser.error("Specify at least one completed export using --label")
        await_scopes(output, args.label, args.wait)
        sync(output, args.label)
    elif args.phase == "gate":
        labels = args.label
        late = args.late or args.milestone == 25000
        if args.milestone:
            labels = [f"{arm}-{seed}-{args.milestone}" for seed in protocol(output)["training_seeds"]
                      for arm in ("sequential", "mixed")]
        elif not (labels and args.gate_name):
            parser.error("Use --milestone or specify a gate name and labels")
        if args.gate_name and Path(args.gate_name).name != args.gate_name:
            parser.error("Use a gate name with no path components")
        await_scopes(output, labels, args.wait)
        decide(output, labels, milestone=args.milestone, late=late, name=args.gate_name)
    elif args.phase == "contrast":
        pairs = [value.split("=", 1) for value in args.pair]
        if (not args.gate_name or Path(args.gate_name).name != args.gate_name or not pairs
                or not all(len(pair) == 2 and all(label and Path(label).name == label for label in pair) for pair in pairs)):
            parser.error("Specify a contrast name through --gate-name and paired labels through --pair")
        contrast(output, pairs, args.gate_name)
    elif args.phase == "freeze":
        if not args.gate_file:
            parser.error("Use --gate-file to select the qualification evidence")
        freeze(output, args.gate_file.resolve())
    else:
        final_panel(output, args.label)


if __name__ == "__main__":
    main()
