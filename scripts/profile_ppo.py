"""Bounded, sequential resource measurements of real PPO workloads."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

CONFIGS = (
    ("cpu-1", "cpu", 1, 64, 16, False),
    ("cpu-2", "cpu", 2, 64, 16, False),
    ("cpu-4", "cpu", 4, 64, 16, False),
    ("cpu-4-workers", "cpu", 4, 64, 16, True),
    ("cpu-4-long", "cpu", 4, 128, 32, False),
    ("cuda-4", "cuda", 4, 64, 16, False),
    ("cpu-4-large-batch", "cpu", 4, 64, 16, False),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=512)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for label, device, envs, rollout, sequence, parallel in CONFIGS:
        available = psutil.virtual_memory().available
        if envs == 4 and available < 3 * 1024**3:
            rows.append(dict(label=label, skipped="less than 3 GiB available before launch"))
            continue
        target = args.out / label
        command = [sys.executable, "-m", "training.train", "--model", "mlp",
                   "--stage", "mixed", "--envs", str(envs), "--rollout", str(rollout),
                   "--sequence", str(sequence), "--epochs", "2", "--device", device,
                   "--steps", str(args.steps), "--seconds", "45", "--out", str(target)]
        batch = 8 if label == "cpu-4-large-batch" else 4
        if batch != 4:
            command.extend(("--sequences-per-batch", str(batch)))
        if parallel:
            command.append("--parallel")
        peak_rss = 0
        lowest_available = available
        start = time.monotonic()
        with target.with_suffix(".log").open("w") as log, subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT
        ) as process:
            root = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    family = [root, *root.children(recursive=True)]
                    total = 0
                    for child in family:
                        try:
                            total += child.memory_info().rss
                        except psutil.NoSuchProcess:
                            continue
                    peak_rss = max(peak_rss, total)
                    lowest_available = min(lowest_available, psutil.virtual_memory().available)
                    if lowest_available < 768 * 1024**2:
                        process.terminate()
                        for child in family[1:]:
                            try:
                                child.terminate()
                            except psutil.NoSuchProcess:
                                continue
                except psutil.NoSuchProcess:
                    break
                time.sleep(.15)
        summary = target / "summary.json"
        data = json.loads(summary.read_text()) if summary.is_file() else {}
        row = dict(data, label=label, device=device, envs=envs, rollout=rollout,
                   sequence=sequence, sequences_per_batch=batch, parallel=parallel, exit_code=process.returncode,
                   seconds=time.monotonic() - start, peak_tree_rss_bytes=peak_rss,
                   min_available_bytes=lowest_available, completed=(
                       process.returncode == 0 and data.get("steps") == args.steps))
        if row["completed"]:
            row["decisions_per_second"] = data["invocation_steps"] / data["elapsed_seconds"]
        rows.append(row)
        (args.out / "jobs.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
