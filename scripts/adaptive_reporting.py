"""Build the campaign's offline report from persisted run evidence."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import file_digest
from html import escape
import json
import os
from pathlib import Path
import shlex
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "runs/adaptive-campaign-20261127"
REPORT = ROOT / "reports/adaptive-campaign-20261127"
WEIGHT = "submission/weights/best_model.pt"


def read(path):
    return json.loads(path.read_text())


def optional(path):
    return read(path) if path.is_file() else None


def put(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def identity(path):
    with path.open("rb") as stream:
        return dict(path=str(path.resolve()), bytes=path.stat().st_size,
                    sha256=file_digest(stream, "sha256").hexdigest())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def checkpoint_manifest(run):
    numbered = sorted({*run.glob("training/*/step-*.pt"), *run.glob("imitation/*/bc-*.pt"),
                       *run.glob("imitation/control-*.pt"), *run.glob("memory/*.pt")})
    exports = {}
    for receipt in sorted(run.glob("exports/*/export.json")):
        data = read(receipt)
        artifact = identity(receipt.parent / WEIGHT)
        source = identity(Path(data["source_checkpoint"]))
        if artifact["sha256"] != data["files"][WEIGHT] or source["sha256"] != data["source_checkpoint_sha256"]:
            raise ValueError(f"Export or numbered source changed since freezing {receipt.parent.name}")
        exports[receipt.parent.name] = dict(artifact=artifact, source=source, receipt=identity(receipt),
                                            model_config=data["model_config"], training_config=data["training_config"],
                                            trained_steps=data["trained_steps"], updates=data["updates"],
                                            submission_files=data["files"])
    probes = [*run.glob("hardware*/*/step-*.pt"), *run.glob("viewer/*/step-*.pt")]
    return dict(numbered=[identity(path) for path in numbered], exports=exports,
                runtime_probes=[identity(path) for path in sorted(probes)])


def curves(run):
    records = {}
    for log in sorted(run.glob("training/*/learning.jsonl")):
        data = lines(log)
        if not data:
            continue
        source_steps = data[0]["steps"] - data[0]["rollout_decisions"]
        samples = []
        for row in data:
            episodes = row.get("episodes", [])
            keys = ("steps", "updates", "seconds", "rollout_decisions", "decisions_per_second", "learning_rate",
                    "loss", "policy_loss", "value_loss", "entropy", "explained_variance", "grad_norm",
                    "approx_kl", "clip_fraction", "anchor_kl", "curriculum")
            sample = {key: row[key] for key in keys if key in row}
            sample.update(added_decisions=row["steps"] - source_steps, completed_episodes=len(episodes),
                          episode_exits=sum(bool(ep.get("exit")) for ep in episodes),
                          episode_deaths=sum(bool(ep.get("dead")) for ep in episodes),
                          episode_reward_sum=sum(ep.get("shaped_return", ep.get("reward", 0.)) for ep in episodes))
            samples.append(sample)
        records[log.parent.name] = dict(summary=optional(log.parent / "summary.json"), samples=samples,
                                        learning_file=identity(log))
    return records


def episode_sources(run, baseline):
    for path in sorted(run.glob("evaluation/*/*/episodes.jsonl")):
        yield path.parent.name, path.parent.parent.name, path, False
    if baseline:
        yield "approach-baseline", "switch", Path(baseline["archives"]["switch"]) / "episodes.jsonl", True
        for label, folders in baseline["archives"]["familiar"].items():
            for folder in folders:
                yield label, "familiar", Path(folder) / "episodes.jsonl", True
    if (run / "final-selection.json").is_file():
        for path in sorted(run.glob("evaluation/final/*/episodes.jsonl")):
            yield path.parent.name, "final", path, False


def episode_csv(run, output, baseline):
    total = 0
    fields = ("policy", "cohort", "archived", "map", "wad_sha256", "geometry_sha256", "family", "stage",
              "engine_seed", "policy_seed", "result", "exit", "dead", "keys", "doors", "locks", "kills", "damage",
              "health", "ammo_spent_lower_bound", "attack_without_visible_enemy", "damage_dealt", "longest_no_progress",
              "ticks", "area_discovered", "switch_near", "switch_faced_in_range", "switch_attempt", "switch_activated",
              "minimum_distance", "invalid_actions", "agent_error", "engine_error", "trace")
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label, cohort, source, archived in episode_sources(run, baseline):
            for result in lines(source):
                row = {key: result.get(key) for key in fields}
                switch = result.get("switch", {})
                row.update({key: value for key, value in switch.get("stages", {}).items() if key in fields and key != "exit"})
                row.update(policy=label, cohort=cohort, archived=archived, result=result.get("outcome"),
                           minimum_distance=switch.get("minimum_distance"),
                           ammo_spent_lower_bound=result.get("work", {}).get("rounds_spent_lower_bound"),
                           attack_without_visible_enemy=result.get("work", {}).get("attack_without_visible_enemy"))
                writer.writerow(row)
                total += 1
    return total


def audit_originals(run):
    initial = read(run / "provenance/initial-identities.json")
    current = {label: identity(ROOT / item["path"]) for label, item in initial.items()}
    equal = {label: initial[label]["sha256"] == row["sha256"] for label, row in current.items()}
    if not all(equal.values()):
        raise ValueError("An original artifact changed during the campaign")
    return dict(initial=initial, final=current, unchanged=equal)


def test_results(run):
    path = run / "tests-final.xml"
    if not path.is_file():
        path = run / "tests.xml"
    if not path.is_file():
        return None
    return dict(artifact=identity(path), suites=[s.attrib for s in ET.parse(path).getroot().iter("testsuite")])


def report_html(output, data):
    local = lambda path: escape(os.path.relpath(path, output.parent), quote=True)
    details = escape(data["decision"]["summary"])
    verdict = escape(data["decision"]["classification"])
    disposition = escape(data["decision"].get("final_assessment", ""))
    next_experiment = escape(data["decision"].get("next_experiment", ""))
    rows = []
    for name, baseline in data["baselines"]["familiar_baselines"].items():
        if name == "approach-baseline":
            continue
        tiers = baseline["tiers"]
        rows.append("<tr>"+"".join(f"<td>{escape(str(value))}</td>" for value in (
            name+" (archived)", "—", "—", "—", "—", baseline["overall"]["exits"],
            *(tiers[str(tier)]["exits"] for tier in range(1, 5)),
            tiers["4"]["deaths"]))+"</tr>")
    for label, candidate in data["validation"].items():
        familiar = candidate["familiar"]
        switch = candidate["switch"]
        range_count = switch["overall"].get("entered_range_from_outside")
        rows.append("<tr>"+ "".join(f"<td>{escape(str(value))}</td>" for value in (
            label, switch["overall"].get("exits_from_outside_range"), range_count,
            switch["by_recovery_stage"].get("0/conditional_switch", {}).get("exits"),
            switch["by_cohort"].get("conditional_walkover", {}).get("exits"), familiar["overall"]["exits"],
            *(familiar["by_tier"][tier]["exits"] for tier in ("combat", "exploration", "keys", "gauntlet")),
            familiar["by_tier"]["gauntlet"]["deaths"])) + "</tr>")
    profiles = [item for item in (data.get("hardware_profiles"), data.get("hardware_followup")) if item]
    probes = [job for item in profiles for job in item["jobs"] if job.get("completed")]
    profiled = ", ".join(f"{job['label']} {job['decisions_per_second']:.1f}/s" for job in probes)
    metrics = data["training_totals"]
    studies = []
    for name, experiment in data["decision"].get("experiments", {}).items():
        studies.append(f"<tr><td>{escape(name)}</td><td>{escape(experiment['status'])}</td>"
                       f"<td>{escape(experiment['result'])}</td></tr>")
    links = (
        ("Full machine-readable report", "report.json"), ("Paired episodes", "episodes.csv"),
        ("All experiment settings", "experiment_manifest.json"), ("Checkpoint hashes", "checkpoint_manifest.json"),
        ("Learning curves", "learning_curves.json"), ("Compliance and timing", "compliance.json"),
        ("Failure analysis", "failure_analysis.md"), ("Reproduction commands", "reproduction_commands.md"),
    )
    output.write_text(f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DOOM NEXUS adaptive training campaign</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui,sans-serif; line-height:1.55; }}
body {{ max-width: 75rem; margin: auto; padding: 2rem 1.5rem; }}
main {{ overflow-x:auto; }} th,td {{ border-bottom:1px solid #7777; padding:.45rem .6rem; text-align:right; }}
th:first-child,td:first-child {{ text-align:left; }} a {{ color:LinkText; }} table {{ width:100%; border-collapse:collapse; }}
</style></head>
<body><main><h1>DOOM NEXUS adaptive training campaign</h1>
<p><strong>{verdict}</strong></p><p>{details}</p><p>{disposition}</p>
<p>{metrics['decisions']:,} PPO decisions, {metrics['updates']:,} PPO updates and
{metrics['seconds']:.1f} seconds of measured training. Campaign elapsed {data['wall_seconds']:.1f} seconds.</p>
<p>{escape(profiled)}</p>
<h2>Paired development performance</h2>
<p>Switch exits and range use 54 episodes across 18 layouts. Familiar maps use 120 episodes across
24 layouts. Each familiar tier has 30 episodes.</p>
<table><thead><tr><th>Policy</th><th>Switch exits</th><th>Range entries</th>
<th>Alignment exits /18</th><th>Walkover exits /24</th><th>Familiar exits</th>
<th>Combat</th><th>Exploration</th><th>Keys</th><th>Gauntlet</th><th>Gauntlet deaths</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Experiment outcomes</h2>
<table><thead><tr><th>Experiment</th><th>Status</th><th>Measured result</th></tr></thead>
<tbody>{''.join(studies)}</tbody></table>
<h2>Next evidence-supported experiment</h2><p>{next_experiment}</p>
<h2>Evidence</h2><ul>{''.join(f'<li><a href="{path}">{title}</a></li>' for title, path in links)}
<li><a href="{local(Path(data['run'])/'progress.log')}">Campaign event log</a></li></ul>
</main></body></html>""")


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--run", type=Path, default=CAMPAIGN)
    args.add_argument("--out", type=Path, default=REPORT)
    args.add_argument("--decision", type=Path, required=True)
    opts = args.parse_args()
    run, report = opts.run.resolve(), opts.out.resolve()
    report.mkdir(parents=True, exist_ok=True)
    checks = audit_originals(run)
    learning = curves(run)
    supervised = {p.parent.name: read(p) for p in sorted(run.glob("imitation/*/learning.json"))}
    put(report / "learning_curves.json", {"ppo": learning, "supervised": supervised})
    put(report / "checkpoint_manifest.json", checkpoint_manifest(run))
    plans = [*run.glob("*-plan.json"), run / "evaluation-protocol.json", *run.glob("imitation/*protocol.json")]
    experiment_files = {str(path.relative_to(run)): {"identity": identity(path), "details": read(path)}
                        for path in sorted(plans)}
    supervisor = read(run / "status.json")
    put(report / "experiment_manifest.json", dict(protocols=experiment_files, jobs=supervisor.get("jobs", {})))
    compliance = {str(path.relative_to(run)): read(path) for path in sorted((run / "compliance").glob("**/compliance.json"))}
    for filename in ("latency.json", "verification.json", "cold-start.json", "recheck.json", "runtime-versions.json"):
        for path in sorted((run / "compliance").glob(f"**/{filename}")):
            compliance[str(path.relative_to(run))] = read(path)
    put(report / "compliance.json", dict(originals=checks, tests=test_results(run), receipts=compliance,
                                        decision=read(opts.decision).get("compliance")))
    baselines = read(run / "development-baselines.json")
    validation = {"approach-baseline": baselines["approach"]}
    validation.update({path.stem: read(path) for path in sorted((run / "evaluation").glob("*.json"))
                       if (run / "exports" / path.stem / "export.json").is_file()})
    decision = read(opts.decision)
    summaries = [curve["summary"] for curve in learning.values() if curve["summary"]]
    totals = dict(decisions=sum(row["invocation_steps"] for row in summaries),
                  updates=sum(row["invocation_updates"] for row in summaries),
                  seconds=sum(row["elapsed_seconds"] for row in summaries))
    totals["decisions_per_second"] = totals["decisions"] / totals["seconds"] if totals["seconds"] else None
    profiles = [optional(run / path / "jobs.json") for path in ("hardware", "hardware-followup")]
    profile_jobs = [row["summary"] for item in profiles if item for row in item if row.get("completed")]
    totals["profiling_decisions"] = sum(row["invocation_steps"] for row in profile_jobs)
    totals["profiling_updates"] = sum(row["invocation_updates"] for row in profile_jobs)
    totals["profiling_seconds"] = sum(row["elapsed_seconds"] for row in profile_jobs)
    totals["native_ppo_preview"] = optional(run / "viewer/live-smoke/summary.json")
    totals["supervised_seconds"] = sum(row["elapsed_seconds"] for row in supervised.values())
    totals["supervised_updates"] = sum(row["records"][-1]["update"] for row in supervised.values())
    source_files = sorted([*ROOT.glob("scripts/adaptive_*.py"), ROOT / "scripts/view_adaptive_campaign.py",
                           ROOT / "training/anchored_ppo.py"])
    clock = read(run / "clock.json")
    finished = supervisor.get("phase") == "complete"
    now = (datetime.fromisoformat(supervisor["last_update_utc"]) if finished else datetime.now(timezone.utc))
    monotonic = supervisor["elapsed_seconds"] if finished else time.monotonic() - clock["start_monotonic"]
    data = dict(run=str(run), clock=clock, report_cutoff_utc=now.isoformat(),
                wall_seconds=(now - datetime.fromisoformat(clock["start_utc"])).total_seconds(),
                elapsed_monotonic_seconds=monotonic,
                training_totals=totals, decision=decision, validation=validation, baselines=baselines,
                hardware=optional(run / "hardware/hardware.json"), hardware_profiles=optional(run / "hardware/selection.json"),
                hardware_followup=optional(run / "hardware-followup/selection.json"),
                imitation_data=optional(run / "imitation/data/collection.json"),
                source_identities=[identity(item) for item in source_files if item.is_file()],
                source_commit=(run / "provenance/git-head.txt").read_text().strip(),
                final_assessment=optional(run / "final-assessment.json"),
                gates={path.stem: read(path) for path in sorted((run / "evaluation").glob("*gate*.json"))},
                episode_rows=episode_csv(run, report / "episodes.csv", baselines))
    put(report / "report.json", data)
    report_html(report / "report.html", data)
    print(f"Built {report / 'report.html'} with {data['episode_rows']:,} episodes.")


if __name__ == "__main__":
    main()
