"""Check fresh-process latency and package one exact frozen campaign export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

from doom_env import limits, make_env, remote
from scripts import campaign_eval, exit_recovery_eval as recovery
from scripts.exit_switch_eval import agent_pid, high_water_rss
from validate import RamSampler, VramSampler


ROOT = recovery.ROOT
DEMO = ROOT / "maps/two_rooms.wad"
HISTORY = ROOT / "runs/exit-switch-study/compliance-seed31/structured-31-25000"
WINDOWS_DOCKER = Path("/mnt/c/Users/Darshan/AppData/Local/Programs/DockerDesktop/resources/bin/docker.exe")
VERSION_PROBE = '''
from importlib.metadata import version
import json, platform
from doom_env import CONTRACT_VERSION
print(json.dumps(dict(python=platform.python_version(), operating_system=platform.platform(),
                      torch=version("torch"), vizdoom=version("vizdoom"), gymnasium=version("gymnasium"),
                      numpy=version("numpy"), contract=CONTRACT_VERSION)))
'''


def distribution(values):
    if not values:
        return dict(samples=0, mean=None, median=None, p95=None, p99=None, maximum=None)
    median, p95, p99 = np.percentile(values, [50, 95, 99]).tolist()
    return dict(samples=len(values), mean=float(np.mean(values)), median=median,
                p95=p95, p99=p99, maximum=max(values))


def gpu_identity():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                 "--format=csv,noheader,nounits"], text=True, timeout=5, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def cold_run(repo, device, log, *, seed, cap):
    started = time.perf_counter()
    total_gpu_before = VramSampler._gpu_used() if device == "cuda" else None
    timings, initial, reset, reason, fault, ready_to_first = [], None, None, None, None, None
    peak_rss, peak_gpu = None, None
    if not 0 <= cap <= 525:
        raise ValueError("Cold-start decision cap must be between 0 and 525")
    with make_env(DEMO, "MAP01", episode_ticks=2100, render_buffers=True) as env:
        observation, _ = env.reset(seed=seed)
        try:
            with campaign_eval.exported_agent(repo, log, device) as (connection, initial):
                ready_at = time.perf_counter()
                pid = agent_pid(connection)
                ram = RamSampler(pid) if pid is not None else None
                vram = VramSampler(pid) if pid is not None and device == "cuda" else None
                if ram:
                    ram.start()
                if vram:
                    vram.baseline = total_gpu_before
                    vram.start()
                try:
                    before_reset = time.perf_counter()
                    connection.reset(limits.RESET_TIMEOUT_S)
                    reset = time.perf_counter() - before_reset
                    budget = limits.agent_budget_s(2100)
                    spent = 0.0
                    for _ in range(cap or 525):
                        allowance = min(budget - spent, limits.ACT_TIMEOUT_S if timings else limits.FIRST_ACT_TIMEOUT_S)
                        if allowance <= 0:
                            raise remote.AgentFailure("timeout", "per-map compute budget exhausted", "episode")
                        before_action = time.perf_counter()
                        if not timings:
                            ready_to_first = (before_action - ready_at) * 1000.0
                        action = connection.act(observation, allowance)
                        seconds = time.perf_counter() - before_action
                        spent += seconds
                        timings.append(seconds * 1000.0)
                        observation, _, ended, truncated, info = env.step(action)
                        if ended or truncated:
                            reason = "exit" if info["exit_reached"] else "death" if ended else "tick_cap"
                            break
                    if reason is None:
                        reason = "diagnostic_step_cap"
                finally:
                    if ram:
                        ram.stop()
                        ram.join()
                        from_hwm = high_water_rss(pid)
                        peak_rss = max(ram.peak_mb, from_hwm or 0) or None
                    if vram:
                        vram.stop()
                        vram.join()
                        final_gpu = VramSampler._gpu_used()
                        difference = max(0, final_gpu - total_gpu_before) if None not in (final_gpu, total_gpu_before) else 0
                        peak_gpu = dict(per_process_mib=vram.peak or None,
                                        additional_whole_gpu_mib=max(difference, vram.peak_total) or None)
        except remote.AgentFailure as error:
            fault = dict(kind=error.kind, where=error.where, message=error.message)
    return dict(device=device, seed=seed, init_seconds=initial, reset_seconds=reset, first_action_ms=timings[0] if timings else None,
                ready_to_first_call_ms=ready_to_first,
                decisions=len(timings), outcome=reason, total_act_seconds=sum(timings) / 1000.0,
                step_latencies_ms=timings, agent_peak_rss_mib=peak_rss, gpu_peak=peak_gpu,
                error=fault, wall_seconds=time.perf_counter() - started)


def earlier_cuda():
    conditions = {"timed_out_original": HISTORY / "validator/container-cuda.log",
                  "passed_same_export_retry": HISTORY / "validator-retry1/container-cuda.log",
                  "warm_baseline_earlier": recovery.LIMIT / "compliance/approach-41-12000/validator/container-cuda.log"}
    result = {}
    for event, path in conditions.items():
        if not path.is_file():
            continue
        report = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(encoding="utf-8"))
        measurements = [line.strip() for line in report.splitlines() if any(item in line for item in
                        ("Initialization:", "Step latency", "timeout during act", "PASSED", "FAILED"))]
        result[event] = dict(log=str(path), sha256=campaign_eval.digest(path), observed=measurements)
    return result


def summarize(records):
    def measurements(key):
        return distribution([record[key] for record in records if record[key] is not None])

    return dict(init_seconds=measurements("init_seconds"), reset_seconds=measurements("reset_seconds"),
                ready_to_first_call_ms=measurements("ready_to_first_call_ms"),
                first_action_ms=measurements("first_action_ms"), decision_ms=distribution(
                    [latency for row in records for latency in row["step_latencies_ms"]]),
                subsequent_decision_ms=distribution([latency for row in records for latency in row["step_latencies_ms"][1:]]),
                errors=[row["error"] for row in records if row["error"]],
                max_agent_rss_mib=measurements("agent_peak_rss_mib")["maximum"],
                max_agent_gpu_mib=max((row["gpu_peak"]["per_process_mib"] or 0 for row in records if row["gpu_peak"]), default=0) or None,
                max_additional_whole_gpu_mib=max((row["gpu_peak"]["additional_whole_gpu_mib"] or 0
                                                for row in records if row["gpu_peak"]), default=0) or None)


def latency(repo, output, devices, repeats, cap):
    output.mkdir(parents=True, exist_ok=False)
    identity = recovery.pin(repo)
    gpu = gpu_identity() if "cuda" in devices else None
    result = dict(artifact=identity, gpu_nvidia_smi=gpu, platform="local evaluation transport", probe_map=str(DEMO),
                  probe_map_sha256=campaign_eval.digest(DEMO), max_decisions=cap or 525, earlier_cuda_history=earlier_cuda(),
                  measurement="A new one-CPU DoomAgent subprocess per trial. Action round trip includes preprocessing and host-visible output. CUDA tensor-to-host output synchronizes action availability.",
                  measurements={})
    for device in devices:
        if device == "cuda" and not gpu:
            result["measurements"][device] = dict(status="unverified", reason="No working nvidia-smi CUDA inventory")
            continue
        records = []
        for trial in range(repeats):
            record = cold_run(repo, device, output / f"{device}-{trial:02d}.log", seed=0, cap=cap)
            records.append(record)
            campaign_eval.append_jsonl(output / "cold-start-trials.jsonl", dict(trial=trial, **record))
            print(f"{device} attempt {trial + 1}/{repeats} | init {record['init_seconds']} s | "
                  f"first action {record['first_action_ms']} ms | {record['error'] or record['outcome']}", flush=True)
        aggregate = summarize(records)
        aggregate["status"] = "failed" if aggregate["errors"] else "passed"
        result["measurements"][device] = aggregate
    if recovery.pin(repo) != identity:
        raise ValueError("Cold-start artifact changed during measurement")
    recovery.save(output / "cold-start.json", result)
    return result


def invoke(output, args):
    command = list(map(str, args))
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(command) + "\n")
        stream.flush()
        return subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode


def validator_observations(path):
    if not path.is_file():
        return None
    report = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(encoding="utf-8"))
    fields = ("Initialization:", "Step latency", "Episode on", "Peak RAM", "Peak GPU memory", "timeout during", "PASSED", "FAILED")
    kept = [line.strip() for line in report.splitlines() if any(field in line for field in fields)]
    first = re.search(r"Step latency .+?first ([0-9.]+) ms", report)
    initialization = re.search(r"Initialization: ([0-9.]+)s", report)
    return dict(log=str(path), sha256=campaign_eval.digest(path), initialization_s=float(initialization.group(1)) if initialization else None,
                first_action_ms=float(first.group(1)) if first else None, observations=kept)


def validator(checkout, output, trial):
    container = output / f"container-{trial:02d}"
    if WINDOWS_DOCKER.is_file():
        code = invoke(container.with_suffix(".log"), ["bash", ROOT / "scripts/verify_campaign_container.sh", checkout, container])
        exits = {device: int(path.read_text()) if (path := container / f"container-{device}.exit").is_file() else None
                 for device in ("cpu", "cuda")}
        mode = "local organizer-recipe image via Docker Desktop"
    elif shutil.which("docker"):
        container.mkdir()
        base = [sys.executable, ROOT / "validate.py", "--repo", checkout, "--map", DEMO, "--docker"]
        exits = {"cpu": invoke(container / "container-cpu.log", [*base, "--no-gpu"]),
                 "cuda": invoke(container / "container-cuda.log", base)}
        code, mode = max(exits.values()), "unchanged validate.py Docker mode"
    else:
        code, exits, mode = None, {"cpu": None, "cuda": None}, "no Docker executable available"
    result = dict(exit_code=code, devices=exits, mode=mode, directory=str(container),
                  telemetry={device: validator_observations(container / f"container-{device}.log") for device in exits})
    recovery.save(output / f"validator-attempt-{trial:02d}.json", result)
    return result


def runtime_inventory(attempts):
    def collect(command):
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=25)
            if run.returncode == 0:
                return dict(status="measured", **json.loads(run.stdout.strip().splitlines()[-1]))
            return dict(status="unverified", exit_code=run.returncode, stderr=run.stderr.strip())
        except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as exc:
            return dict(status="unverified", reason=str(exc))

    inventory = dict(local=collect([sys.executable, "-c", VERSION_PROBE]), containers={})
    docker_client = str(WINDOWS_DOCKER) if WINDOWS_DOCKER.is_file() else shutil.which("docker")
    files = [Path(attempt["directory"]) / "container-image.txt" for attempt in attempts]
    images = {file.read_text().strip() for file in files if file.is_file()}
    if docker_client:
        for image in sorted(images):
            command = [docker_client, "run", "--rm", "--pull", "never", "--network", "none", "--read-only",
                       "--env", "PYTHONDONTWRITEBYTECODE=1", "--entrypoint", "/opt/venv/bin/python", image,
                       "-c", VERSION_PROBE]
            inventory["containers"][image] = collect(command)
    return inventory


def verify(repo, output, devices, repeats, cap, container_repeats):
    output.mkdir(parents=True, exist_ok=False)
    identity = recovery.pin(repo)
    export = recovery.read_json(repo / "export.json")
    checkpoint = Path(export["source_checkpoint"])
    if campaign_eval.digest(checkpoint) != export["source_checkpoint_sha256"]:
        raise ValueError("Numbered training checkpoint changed")
    package = invoke(output / "package.log", [sys.executable, "-m", "scripts.package_campaign",
                                             "--artifact", repo, "--out", output / "repo"])
    result = dict(artifact=identity, source_checkpoint=str(checkpoint), source_checkpoint_sha256=export["source_checkpoint_sha256"],
                  package_exit_code=package, passed=False)
    if package == 0:
        checkout = output / "repo-checkout"
        checked_identity = recovery.pin(checkout, identity["weights_sha256"])
        if checked_identity["source_sha256"] != identity["source_sha256"]:
            raise ValueError("Packaged policy source differs from the frozen export")
        package_receipt = recovery.read_json(output / "package.json")
        cold = latency(checkout, output / "cold-start", devices, repeats, cap)
        parity = invoke(output / "parity.log", [sys.executable, "-m", "scripts.audit_exit_vision", "--checkpoint", checkpoint,
                                               "--repo", checkout, "--wad", DEMO, "--out", output / "parity.json"])
        attempts = [validator(checkout, output, trial) for trial in range(container_repeats)]
        if recovery.pin(checkout, identity["weights_sha256"]) != checked_identity:
            raise ValueError("Packaged source or weights changed during validation")

        def status(codes):
            if any(code not in (None, 0) for code in codes):
                return "failed"
            return "unverified" if any(code is None for code in codes) else "passed"

        sandbox = status([trial["exit_code"] for trial in attempts])
        bytes_on_disk = package_receipt["packed_bytes"]
        result.update(cold_start=cold, parity_exit_code=parity, official_validator_sha256=campaign_eval.digest(ROOT / "validate.py"),
                      container_attempts=attempts, official_container_devices={d: status([trial["devices"][d] for trial in attempts])
                                                                            for d in ("cpu", "cuda")},
                      runtimes=runtime_inventory(attempts),
                      offline_and_read_only_container=sandbox, checkout_commit=package_receipt["commit"],
                      packed_bytes_including_git=bytes_on_disk, repository_size_limit_bytes=limits.MAX_REPO_BYTES,
                      passed=parity == 0 and sandbox == "passed" and bytes_on_disk <= limits.MAX_REPO_BYTES
                      and all(m["status"] == "passed" for m in cold["measurements"].values()))
    if recovery.pin(repo) != identity or campaign_eval.digest(checkpoint) != export["source_checkpoint_sha256"]:
        raise ValueError("Exact artifact changed during compliance")
    recovery.save(output / "compliance.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("latency", "verify"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--container-repeats", type=int, default=2, help="fresh official container attempts per device")
    parser.add_argument("--max-decisions", type=int, default=0, help="0 runs the whole 2,100-tic episode")
    args = parser.parse_args()
    if args.repeats < 2 or args.container_repeats < 1 or len(set(args.devices)) != len(args.devices):
        parser.error("Use at least two local repetitions, one container repetition and unique devices")
    if not 0 <= args.max_decisions <= 525:
        parser.error("Decision cap must be between 0 and 525")
    if args.phase == "verify":
        result = verify(args.repo.resolve(), args.out.resolve(), args.devices, args.repeats, args.max_decisions, args.container_repeats)
        return int(not result["passed"])
    result = latency(args.repo.resolve(), args.out.resolve(), args.devices, args.repeats, args.max_decisions)
    return int(any(item["status"] != "passed" for item in result["measurements"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
