"""Complete the autonomous campaign audit after 10,800 saved learning seconds."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import suppress
import csv
from datetime import datetime, timezone
from html import escape
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import psutil

from scripts import adaptive_campaign_eval as evaluation
from scripts import adaptive_reporting as reporting
from scripts.adaptive_lock_crossing_report import observed_progress
from scripts.continuity_campaign import DEFAULT, PYTHON_ENV, ROOT, TARGET, module, note, read
from training.learning_meter import METADATA_KEY, NS_PER_SECOND, write_json


def verify_learning(run, status):
    """Reconcile the controller ledger with the checkpoints that earned credit."""
    if status.get("training_pid") or status.get("evaluation_pid"):
        raise ValueError("Complete training and the development queue before finalization")
    if status["active_training_seconds"] < TARGET:
        raise ValueError(f"Finalization requires {TARGET:,} saved active learning seconds")
    import torch

    credited, meter_ids = {}, set()
    for name, groups in (("ppo", status["blocks"]), ("supervised", status["supervised_blocks"])):
        for block in groups:
            elapsed = block.get("active_training_ns", 0)
            if not isinstance(elapsed, int) or elapsed < 0:
                raise ValueError(f"Invalid active time in {block['label']}")
            if not elapsed:
                continue
            checkpoint = Path(block.get("numbered_checkpoint") or block["checkpoint"]).resolve(strict=True)
            identity = block.get("meter_id", block["label"])
            if (not checkpoint.is_relative_to(run / "training") or block["label"] in credited or identity in meter_ids):
                raise ValueError(f"Invalid or duplicate learning credit for {block['label']}")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            counters = (payload["trained_steps"], payload["updates"])
            if name == "ppo":
                meter = payload["args"][METADATA_KEY]
                saved = meter["active_rollout_ns"] + meter["active_optimization_ns"]
                decisions = counters[0] - meter["start_steps"]
                updates = counters[1] - meter["start_updates"]
                if (meter["meter_id"] != identity or
                        counters != (meter["checkpoint_steps"], meter["checkpoint_updates"]) or
                        decisions != block["added_decisions"] or updates != block["added_updates"] or
                        not (decisions > 0 and updates > 0)):
                    raise ValueError(f"PPO checkpoint counters disagree with the ledger for {block['label']}")
            else:
                meter = payload["curriculum"]["continuity_bc"]["summary"]
                saved = meter["active_training_ns"]
                decisions, updates = 0, meter["added_supervised_updates"]
                if (meter["meter_id"] != identity or updates != block["supervised_updates"]
                        or updates < 1 or counters != (meter["checkpoint_steps"], meter["checkpoint_updates"])):
                    raise ValueError(f"Supervised checkpoint counters disagree with the ledger for {block['label']}")
            if saved != elapsed:
                raise ValueError(f"Checkpoint-backed active time disagrees for {block['label']}")
            meter_ids.add(identity)
            credited[block["label"]] = dict(kind=name, phase=block["phase"], active_training_ns=saved,
                                           decisions=decisions, updates=updates, checkpoint=reporting.identity(checkpoint))
    active = sum(item["active_training_ns"] for item in credited.values())
    if active < TARGET * NS_PER_SECOND:
        raise ValueError(f"Finalization requires {TARGET:,} saved active learning seconds, found {active / NS_PER_SECOND:.3f}")
    if not math.isclose(active / NS_PER_SECOND, status["active_training_seconds"], rel_tol=0., abs_tol=1e-6):
        raise ValueError("The controller's active duration disagrees with verified checkpoints")
    return credited


def development_evidence(run, status):
    settings = evaluation.protocol(run)
    baseline_receipt = read(run / "development-baselines.json")
    if baseline_receipt["protocol_sha256"] != evaluation.campaign_eval.digest(run / "evaluation-protocol.json"):
        raise ValueError("The preregistered development protocol changed during the campaign")
    panels = evaluation.panels(settings)
    pins, switch_reference, familiar_references = evaluation.archived(panels)
    trials, unavailable = {}, {}
    groups = {evaluation.APPROACH: {"switch": switch_reference}}
    for label, rows in familiar_references.items():
        groups.setdefault(label, {})["familiar"] = rows
    for item in status["evaluation_queue"]:
        label = item["label"]
        if not label or Path(label).name != label or label in {".", ".."}:
            raise ValueError("Unexpected label in the development queue")
        if item["status"] != "complete":
            unavailable[label] = f"Development attempt status is {item['status']}"
            continue
        try:
            left, switch = evaluation.frozen_scope(run, "switch", label, panels["switch"])
            right, familiar = evaluation.frozen_scope(run, "familiar", label, panels["familiar"])
            export = read(run / "exports" / label / "export.json")
            source = Path(export["source_checkpoint"]).resolve(strict=True)
            seed = (export.get("training_config") or {}).get("seed", export.get("training_seed"))
            if (left != right or Path(left["repo"]) != run / "exports" / label
                    or source != Path(item["checkpoint"]).resolve(strict=True)
                    or evaluation.campaign_eval.digest(source) != export["source_checkpoint_sha256"]
                    or seed != item["seed"]):
                raise ValueError("Paired panels, source checkpoint, or training seed differ")
            comparison = evaluation.compare(switch_reference, switch, familiar_references[evaluation.APPROACH], familiar)
        except (OSError, KeyError, ValueError) as error:
            unavailable[label] = str(error)
            continue
        trials[label] = dict(identity=left, training_seed=seed, source_checkpoint=str(source),
                             source_checkpoint_sha256=export["source_checkpoint_sha256"], vs_approach=comparison,
                             gate=evaluation.gate_metrics(comparison, switch, familiar, settings["thresholds"], late=True),
                             development=read(run / "evaluation" / f"{label}.json"))
        groups[label] = dict(switch=switch, familiar=familiar)
    return settings, panels, pins, groups, trials, unavailable


def prior_teacher(run, panels, groups, unavailable):
    label = "bc-ppo-59-12000"
    expected = read(run / "provenance" / "initial-identities.json")["lock_opening_export"]
    archive = (ROOT / expected["path"]).resolve().parents[4]
    try:
        left, switch = evaluation.frozen_scope(archive, "switch", label, panels["switch"])
        right, familiar = evaluation.frozen_scope(archive, "familiar", label, panels["familiar"])
        if left != right or left["weights_sha256"] != expected["sha256"]:
            raise ValueError("The retained imitation branch identity changed")
    except (OSError, KeyError, ValueError) as error:
        unavailable[label] = str(error)
        return {}
    groups[label] = dict(switch=switch, familiar=familiar)
    return {label: dict(identity=left, training_seed=59, development=read(archive / "evaluation" / f"{label}.json"))}


def strict_pairs(run, status, settings, trials):
    """Check the recorded pair and recompute its gates from the pinned episodes."""
    pairs, issues = {}, {}
    blocks = {block["label"]: block for block in status["blocks"]}
    for name, entry in status["gates"].items():
        if not entry["passed"]:
            continue
        try:
            path = Path(entry["path"]).resolve(strict=True)
            if not path.is_relative_to(run / "evaluation"):
                raise ValueError("The late gate is not part of this campaign")
            receipt = evaluation.verify_gate(run, path)
            labels = set(entry["labels"])
            members = {x["label"] for x in status["evaluation_queue"]
                       if x["pair"] == name and x["status"] == "complete"}
            seeds = {trials[label]["training_seed"] for label in labels}
            matched = {(blocks[label]["profile"], blocks[label]["parent"], blocks[label]["phase"]) for label in labels}
            if (labels != set(receipt["qualified"]) or labels != set(receipt["results"]) or labels != members
                    or seeds != set(settings["training_seeds"]) or len(seeds) < 2
                    or len(labels) != len(seeds) or seeds != set(receipt["independent_training_seeds"])
                    or len(matched) != 1):
                raise ValueError("The late gate does not cover the independent matched seeds")
            for label in labels:
                actual, recorded = trials[label], receipt["results"][label]
                if (receipt["qualified"][label] is not True or not actual["gate"]["passed"]
                        or recorded["gate"] != actual["gate"] or recorded["identity"] != actual["identity"]
                        or recorded["training_seed"] != actual["training_seed"]
                        or recorded["source_checkpoint_sha256"] != actual["source_checkpoint_sha256"]
                        or evaluation.final_rank(recorded) != evaluation.final_rank(actual)):
                    raise ValueError(f"Late gate evidence differs for {label}")
            pairs.update({label: path for label in labels})
        except (OSError, KeyError, ValueError) as error:
            issues[name] = str(error)
    return pairs, issues


def command(run, attempt, label, argv, seconds, cancelled):
    """Run a bounded finalization job and preserve its exact invocation."""
    if cancelled() or (run / "STOP").exists():
        raise InterruptedError("Finalization paused by operator request")
    start = time.monotonic()
    path = attempt / f"{label}.log"
    with path.open("x", encoding="utf-8") as log:
        log.write("$ " + shlex.join(argv) + "\n")
        log.flush()
        process = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   env={**os.environ, **PYTHON_ENV})
        reason = None
        try:
            while process.poll() is None:
                if (run / "STOP").exists() or cancelled():
                    reason = "operator pause"
                    break
                if time.monotonic() - start >= seconds:
                    reason = "wall limit"
                    break
                time.sleep(.5)
        finally:
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM if reason is None else signal.SIGKILL)
    receipt = dict(command=argv, log=str(path), wall_seconds=time.monotonic() - start, wall_limit_seconds=seconds,
                   return_code=process.returncode, stop_reason=reason, passed=reason is None and process.returncode == 0)
    write_json(attempt / f"{label}.json", receipt)
    if reason == "operator pause":
        raise InterruptedError("Finalization paused by operator request")
    return receipt


def run_audit(run, attempt, repo, cancelled):
    output = attempt / "artifact"
    receipt = command(run, attempt, "compliance", module("scripts.adaptive_compliance", "verify", "--repo", repo,
                      "--out", output, "--devices", "cpu", "cuda", "--repeats", 5, "--container-repeats", 2),
                      660, cancelled)
    data = reporting.optional(output / "compliance.json")
    result = dict(artifact=receipt, exact_audit=data, host_validator={},
                  passed=bool(receipt["passed"] and data and data["passed"]))
    if not result["passed"]:
        local = output / "repo-checkout"
        if not local.is_dir():
            local = repo
        for device in ("cpu", "cuda"):
            result["host_validator"][device] = command(run, attempt, f"validator-host-{device}",
                [sys.executable, str(ROOT / "validate.py"), "--repo", str(local), "--device", device], 170, cancelled)
    return result


def run_reserved(run, attempt, selected, late_gate, audit_passed, cancelled):
    if not late_gate:
        return dict(status="sealed", reason="No recomputed two-seed late gate passed")
    if not audit_passed:
        return dict(status="sealed", reason="Exact artifact audit or project tests did not pass")
    selection_file = run / "final-selection.json"
    if not selection_file.exists():
        evaluation.freeze(run, late_gate)
    frozen = read(selection_file)
    if (frozen["selected"] != selected or Path(frozen["gate"]) != late_gate
            or frozen["gate_sha256"] != evaluation.campaign_eval.digest(late_gate)):
        raise ValueError("The reserved assessment has a different frozen candidate")
    if (run / "final-assessment.json").is_file():
        return dict(status="complete", report=read(run / "final-assessment.json"))
    for label in frozen["policies"]:
        folder = run / "evaluation" / "final" / label
        if folder.is_dir() and not (folder / "summary.json").is_file():
            folder.rename(folder.with_name(f"{label}-interrupted-{time.time_ns()}"))
    write_json(attempt / "reserved-access.json", dict(candidate=selected, gate=str(late_gate)))
    result = command(run, attempt, "reserved-assessment", module("scripts.adaptive_campaign_eval", "assess", "--out", run),
                     1800, cancelled)
    report = reporting.optional(run / "final-assessment.json")
    return dict(status="complete" if result["passed"] and report else "incomplete",
                attempt=result, report=report)


def log_rows(path):
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append(dict(incomplete_json_line=index))
    return rows


def learning_curves(run):
    ppo, bc = {}, {}
    for folder in sorted((run / "training").iterdir()):
        if (folder / "active-learning.jsonl").is_file():
            stats = log_rows(folder / "learning.jsonl")
            entries = {}
            for row in stats:
                if "updates" not in row:
                    continue
                compact = {k: v for k, v in row.items() if k != "episodes"}
                examples = row.get("episodes", [])
                compact.update(episodes=len(examples), exits=sum(bool(ep.get("exit")) for ep in examples),
                               deaths=sum(bool(ep.get("dead")) for ep in examples))
                entries[(row["steps"], row["updates"])] = compact
            receipts = log_rows(folder / "active-learning.jsonl")
            for row in receipts:
                if "checkpoint_steps" in row:
                    row.update(evidence=entries.get((row["checkpoint_steps"], row["checkpoint_updates"])))
            ppo[folder.name] = dict(committed=receipts, summary=reporting.optional(folder / "summary.json"))
        if (folder / "imitation-learning.jsonl").is_file():
            bc[folder.name] = dict(updates=log_rows(folder / "imitation-learning.jsonl"),
                                   summary=reporting.optional(folder / "summary.json"))
    return dict(ppo=ppo, supervised=bc)


def progression_evidence(run, panels, groups, unavailable):
    result, post_lock, stages = {}, {}, {}
    specs = {(spec["sha256"], spec["map_name"]): spec for spec in panels["familiar"][1]}
    for label, cohorts in groups.items():
        if "familiar" in cohorts:
            try:
                result[label] = observed_progress(cohorts["familiar"], specs)
            except (OSError, KeyError, ValueError, IndexError) as error:
                unavailable[f"{label} lock-crossing diagnostic"] = str(error)
    for path in sorted((run / "post-lock-diagnostics").glob("*/post-lock-funnel.json")):
        label = path.parent.name
        data = read(path)
        events = {(r["layout"], r["engine_seed"]): r["stage_tics"] for r in data.pop("by_episode")}
        post_lock[label] = dict(data, source=str(path))
        groups.setdefault(label, {})["post-lock"] = log_rows(path.parent / "episodes.jsonl")
        stages[label] = events
    return result, post_lock, stages


def episode_csv(path, groups, seeds, crossed, post_lock):
    columns = ("policy", "training_seed", "scope", "archived", "map", "family", "stage", "geometry_sha256",
               "wad_sha256", "engine_seed", "policy_seed", "outcome", "exit", "dead", "ticks", "health", "damage",
               "damage_dealt", "keys", "doors", "locks", "doors_traversed", "relevant_lock_crossed",
               "ordered_key_unlock_cross_exit", "kills", "unique_cells", "longest_no_progress",
               "rounds_spent_lower_bound", "attack_without_visible_enemy", "inference_ms", "p99_ms",
               "invalid_actions", "inference_timeouts", "crashes", "agent_error", "engine_error",
               "area_discovered", "switch_near", "switch_faced_in_range", "switch_attempt", "switch_activated",
               "post_lock_door_crossing", "post_lock_exit_area_discovery", "post_lock_interaction_range",
               "post_lock_alignment", "post_lock_fresh_use", "post_lock_confirmed_exit", "trace")
    count = 0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for label, panels in groups.items():
            passage = {(r["map"], r["engine_seed"]): r for r in crossed.get(label, {}).get("episodes", [])}
            for scope, episodes in panels.items():
                for data in episodes:
                    if "incomplete_json_line" in data:
                        continue
                    row = {field: data.get(field) for field in columns}
                    row.update(policy=label, training_seed=seeds.get(label), scope=scope,
                               archived=label in {evaluation.APPROACH, "published", "exploration-baseline", "bc-ppo-59-12000"}
                               and scope != "reserved")
                    for key in ("rounds_spent_lower_bound", "attack_without_visible_enemy"):
                        row[key] = data.get("work", {}).get(key)
                    row.update({key: value for key, value in data.get("switch", {}).get("stages", {}).items()
                                if key in columns and key != "exit"})
                    identity = (data.get("map"), data.get("engine_seed"))
                    if scope == "familiar" and identity in passage:
                        crossing = passage[identity]
                        row["relevant_lock_crossed"] = crossing["traversed"]
                        row["ordered_key_unlock_cross_exit"] = (None if crossing["traversed"] is None
                                                               else bool(crossing["traversed"] and crossing["exit"]))
                    if scope == "post-lock":
                        row.update({f"post_lock_{key}": value for key, value in post_lock.get(label, {}).get(identity, {}).items()})
                    writer.writerow(row)
                    count += 1
    return count


def elapsed(clock):
    wall = (datetime.now(timezone.utc) - datetime.fromisoformat(clock["start_utc"])).total_seconds()
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    return dict(wall_seconds=wall, elapsed_monotonic_seconds=(time.monotonic() - clock["start_monotonic"]
                if boot == clock["boot_id"] else None))


def reproduction(path, run, status, jobs):
    monitor = [sys.executable, "-m", "scripts.continuity_campaign"]
    view = module("scripts.view_adaptive_campaign", "--run", run, "--panel", "post-lock", "--speed", 1)
    warm = status["blocks"][0]["parent"]
    visible = module("scripts.continuity_post_lock", "learner", "--out", run / "native-live-probe",
                     "--resume", warm, "--seed", 59, "--active-seconds", 30, "--seconds", 900,
                     "--decisions", 10000, "--visualize", "--speed", 1)
    lines = ["# Reproduction and live gameplay", "",
             "Choose an unused output directory for a repeat learner so earlier evidence remains available.", "",
             "```bash", shlex.join(["cd", str(ROOT)]),
             "watch -n 2 " + shlex.quote(shlex.join([*monitor, "status", "--out", str(run)])),
             shlex.join(["tail", "-f", str(run / "progress.log")]), "```", "",
             "Run the frozen checkpoint viewer during headless training.", "", "```bash",
             "SDL_VIDEODRIVER=x11 " + shlex.join(view), "```", "",
             "Start the actual native PPO probe in a display-capable terminal. It writes to its own directory.",
             "", "```bash", shlex.join(visible), "```", "",
             "Stop and resume the supervised campaign with preserved counters.", "", "```bash",
             shlex.join([*monitor, "stop", "--out", str(run)]),
             shlex.join([*monitor, "resume", "--out", str(run)]), "```", ""]
    invocations = [args for block in status["blocks"] for args in block.get("commands", [])]
    bootstrap = status["blocks"][0]
    if bootstrap.get("unit"):
        source = read(Path(bootstrap["output"]) / "continuity-invocation.json")
        options = source["options"]
        invocations.insert(0, module("scripts.continuity_post_lock", "learner", "--out", bootstrap["output"],
            "--resume", bootstrap["parent"], "--seed", bootstrap["seed"],
            "--active-seconds", bootstrap["active_limit_seconds"], "--seconds", options[options.index("--seconds") + 1],
            "--decisions", int(options[options.index("--steps") + 1]) - source["start_steps"],
            "--rollout", options[options.index("--rollout") + 1], "--meter-id", bootstrap["meter_id"],
            "--stop-file", run / "STOP"))
    for heading, commands in (
        ("Recorded learner invocations", invocations),
        ("Final verification invocations", [receipt["command"] for receipt in jobs]),
    ):
        lines.extend((f"## {heading}", "", "```bash", *(shlex.join(cmd) for cmd in commands), "```", ""))
    path.write_text("\n".join(lines), encoding="utf-8")


def report_html(path, data):
    timing = data["timing"]
    picked = data["selection"]
    row_html = []
    for label, entry in data["development"].items():
        stats = entry["development"]
        switch = stats["switch"]["overall"]
        familiar = stats["familiar"]
        combat = stats["familiar_combat_and_progression"]
        crossing = data["relevant_lock_crossing"].get(label, {}).get("tiers", {})
        fields = [label, entry["training_seed"], switch["exits_from_outside_range"], switch["entered_range_from_outside"],
                  stats["switch"]["by_recovery_stage"].get("0/conditional_switch", {}).get("exits"),
                  stats["switch"]["by_cohort"].get("conditional_walkover", {}).get("exits"),
                  familiar["overall"]["exits"], *(combat[tier]["exits"] for tier in ("combat", "exploration", "keys", "gauntlet")),
                  combat["keys"]["unlock_episodes"], crossing.get("gauntlet", {}).get("traversals"),
                  combat["gauntlet"]["deaths"]]
        row_html.append("<tr>" + "".join(f"<td>{escape(str(x) if x is not None else '—')}</td>" for x in fields) + "</tr>")
    heads = ("Policy", "Training seed", "Far switch exits", "Range entries", "Alignment", "Walkovers", "Familiar exits",
             "Tier 1", "Tier 2", "Tier 3", "Tier 4", "Tier 3 unlocks", "Gauntlet lock crossings", "Gauntlet deaths")
    links = ("report.json", "episodes.csv", "compliance.json", "experiment_manifest.json", "checkpoint_manifest.json",
             "learning_curves.json", "reproduction_commands.md", "failure_analysis.md")
    report = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DOOM NEXUS three-hour continuity campaign</title>
<style>
body {{font-family:system-ui,sans-serif;line-height:1.6;margin:2rem auto;padding:0 1rem;max-width:94rem}}
.results {{overflow:auto}} table {{border-collapse:collapse}} th,td {{padding:.4rem;text-align:right;border-bottom:1px solid #8885}}
th:first-child,td:first-child {{text-align:left}} :root {{color-scheme:light dark}}
</style></head><body><main>
<h1>DOOM NEXUS continuity campaign</h1>
<p><strong>{escape(picked["classification"])}</strong> · {escape(picked["label"])}</p>
<p>{timing["active_training_seconds"]:,.3f} active learning seconds across {timing["ppo_updates"]:,} PPO
updates and {timing["supervised_updates"]:,} supervised updates. PPO consumed {timing["ppo_decisions"]:,} decisions.
Elapsed wall time was {timing["wall_seconds"]:,.1f} seconds. The frozen development evaluator spent
{timing["development_evaluation_seconds"]:,.1f} seconds and was excluded from learning time.</p>
<p>Reserved assessment {escape(data["reserved_assessment"]["status"])}.
Published SHA-256 {escape(data["originals"]["final"]["published"]["sha256"])}.</p>
<h2>Paired development gameplay</h2>
<p>Switch and familiar panels were paired by WAD and engine seed. The numerical endpoints, layout-specific variation,
post-lock funnel, combat efficiency, inference latency and comparisons against the frozen start are in the JSON evidence.</p>
<div class="results"><table><thead><tr>{"".join(f"<th>{head}</th>" for head in heads)}</tr></thead>
<tbody>{"".join(row_html)}</tbody></table></div>
<h2>Evidence</h2><ul>{"".join(f'<li><a href="{link}">{link}</a></li>' for link in links)}</ul>
</main></body></html>"""
    path.write_text(report, encoding="utf-8")


def finalize(run, output, cancelled):
    marker = run / "finalized.json"
    if marker.is_file():
        result = read(marker)
        for item in result["files"].values():
            if reporting.identity(Path(item["path"])) != item:
                raise ValueError("The completed campaign report changed")
        print(json.dumps(result), flush=True)
        return
    status = read(run / "status.json")
    clock = read(run / "clock.json")
    credited = verify_learning(run, status)
    originals = reporting.audit_originals(run)
    settings, panels, pins, groups, candidates, unavailable = development_evidence(run, status)
    qualified, gate_issues = strict_pairs(run, status, settings, candidates)
    ranked = sorted(candidates, key=lambda label: evaluation.final_rank(candidates[label]), reverse=True)
    selected = next((label for label in ranked if label in qualified), ranked[0] if ranked else evaluation.APPROACH)
    before = candidates[selected]["identity"] if selected in candidates else pins[selected]
    attempt = run / "finalize-attempts" / str(time.time_ns())
    attempt.mkdir(parents=True)
    write_json(attempt / "selection.json", dict(label=selected, candidates_in_predeclared_rank_order=ranked,
               late_gate=str(qualified[selected]) if selected in qualified else None, identity=before))
    note(run, f"Started bounded final audits for {selected} after the verified learning target.")
    audit = run_audit(run, attempt, Path(before["repo"]), cancelled)
    tests = command(run, attempt, "tests", module("pytest", "-q", "--junitxml", attempt / "pytest.xml"), 480, cancelled)
    tests_path = attempt / "pytest.xml"
    if tests_path.is_file():
        tests["suites"] = [suite.attrib for suite in ET.parse(tests_path).getroot().iter("testsuite")]
        tests["xml"] = reporting.identity(tests_path)
    reserved = run_reserved(run, attempt, selected, qualified.get(selected), audit["passed"] and tests["passed"], cancelled)
    originals = reporting.audit_originals(run)
    if evaluation.recovery.pin(Path(before["repo"]), before["weights_sha256"]) != before:
        raise ValueError("The audited policy artifact changed")
    if selected not in qualified or not audit["passed"] or not tests["passed"]:
        classification = "REJECTED FOR PROMOTION"
    elif reserved["status"] == "complete":
        classification = "VERIFIED DEVELOPMENT CANDIDATE WITH RESERVED RESULTS"
    else:
        classification = "VERIFIED DEVELOPMENT CANDIDATE WITH INCOMPLETE RESERVED ASSESSMENT"
    picked = dict(label=selected, classification=classification, identity=before, ranked_development_candidates=ranked,
                  qualified_two_seed_gate=str(qualified[selected]) if selected in qualified else None,
                  published_replacement=False)
    baselines = read(run / "development-baselines.json")
    reference = baselines["approach"]
    additional_reference = prior_teacher(run, panels, groups, unavailable)
    candidates = {evaluation.APPROACH: dict(identity=pins[evaluation.APPROACH], training_seed=None, development=reference),
                  **additional_reference, **candidates}
    crossed, post_lock, stages = progression_evidence(run, panels, groups, unavailable)
    if reserved["status"] in {"complete", "incomplete"}:
        for directory in sorted((run / "evaluation" / "final").glob("*/episodes.jsonl")):
            groups.setdefault(directory.parent.name, {})["reserved"] = log_rows(directory)
    timing = {**elapsed(clock), "active_training_seconds": sum(v["active_training_ns"] for v in credited.values()) / NS_PER_SECOND,
              "ppo_training_seconds": sum(v["active_training_ns"] for v in credited.values() if v["kind"] == "ppo") / NS_PER_SECOND,
              "supervised_training_seconds": sum(v["active_training_ns"] for v in credited.values() if v["kind"] != "ppo") / NS_PER_SECOND,
              "ppo_decisions": sum(v["decisions"] for v in credited.values()),
              "ppo_updates": sum(v["updates"] for v in credited.values() if v["kind"] == "ppo"),
              "supervised_updates": sum(v["updates"] for v in credited.values() if v["kind"] != "ppo"),
              "development_evaluation_seconds": status["evaluation_seconds"]}
    timing["nontraining_wall_seconds"] = max(0., timing["wall_seconds"] - timing["active_training_seconds"])
    resources = dict(final_available_ram_mib=psutil.virtual_memory().available / 2**20,
                     final_available_disk_gib=shutil.disk_usage(run).free / 2**30)
    output.mkdir(parents=True, exist_ok=True)
    row_count = episode_csv(output / "episodes.csv", groups, {label: item["training_seed"] for label, item in candidates.items()},
                            crossed, stages)
    receipts = [read(path) for path in sorted(attempt.glob("*.json"))]
    jobs = [row for row in receipts if "command" in row]
    reproduction(output / "reproduction_commands.md", run, status, jobs)
    write_json(output / "compliance.json", dict(audit=audit, tests=tests, originals=originals))
    checkpoint_manifest = reporting.checkpoint_manifest(run)
    checkpoint_manifest["credited_checkpoints"] = credited
    write_json(output / "checkpoint_manifest.json", checkpoint_manifest)
    write_json(output / "learning_curves.json", learning_curves(run))
    write_json(output / "experiment_manifest.json", dict(state=status, preregistered_protocol=settings,
               block_invocations={str(p.relative_to(run)): read(p) for p in sorted((run / "training").glob("*/*invocation.json"))},
               profiles=reporting.optional(run / "continuity-profile.json"), credited=credited, final_jobs=jobs))
    limits = [f"{label}: {reason}." for label, reason in {**unavailable, **gate_issues}.items()]
    if not audit["passed"]:
        limits.append("The exact artifact audit lacked a complete offline, read-only CPU and CUDA pass.")
    if not tests["passed"]:
        limits.append("The complete project test run did not pass within its bound.")
    (output / "failure_analysis.md").write_text("\n".join(("# Failure analysis", "",
        "The campaign clock counts only learning intervals reconciled with saved optimizer updates.", "",
        "Development gates reject an individual checkpoint. Later blocks continue their learning budget.", "",
        *(f"- {line}" for line in limits), "",
        "Conditional post-lock play is recorded separately from a full key, unlock, traversal and exit chain.", "",
        "The familiar and switch panels have been used for model selection. Only an eligible frozen artifact "
        "can be assessed on the reserved panel.", "")), encoding="utf-8")
    data = dict(selection=picked, timing=timing, clock=clock, development=candidates, gates=status["gates"],
                archived_standard_level_baselines=baselines["familiar_baselines"],
                gate_issues=gate_issues, post_lock=post_lock, relevant_lock_crossing=crossed,
                reserved_assessment=reserved, originals=originals, audit=audit, tests=tests,
                development_failures=unavailable, hardware=reporting.optional(run / "provenance" / "hardware.json"),
                deadline_audit=reporting.optional(run / "provenance" / "deadline-audit.json"),
                last_training_resources=status.get("resources"), final_resources=resources, episode_rows=row_count)
    timing.update(elapsed(clock))
    timing["nontraining_wall_seconds"] = max(0., timing["wall_seconds"] - timing["active_training_seconds"])
    write_json(output / "report.json", data)
    report_html(output / "report.html", data)
    files = {name: reporting.identity(output / name) for name in
             ("report.html", "report.json", "episodes.csv", "compliance.json", "experiment_manifest.json",
              "checkpoint_manifest.json", "learning_curves.json", "reproduction_commands.md", "failure_analysis.md")}
    final = dict(selection=picked, timing=timing, receipt=str(attempt), files=files,
                 reserved_assessment=reserved["status"], compliance_passed=audit["passed"], tests_passed=tests["passed"])
    write_json(marker, final)
    print(json.dumps(final), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--reports", type=Path)
    args = parser.parse_args()
    run = args.out.resolve(strict=True)
    output = (args.reports or ROOT / "reports" / run.name).resolve()
    cancelled = False

    def pause(*_):
        nonlocal cancelled
        cancelled = True

    signal.signal(signal.SIGTERM, pause)
    signal.signal(signal.SIGINT, pause)
    finalize(run, output, lambda: cancelled)


if __name__ == "__main__":
    main()
