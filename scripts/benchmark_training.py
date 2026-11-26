"""Compare complete real PPO runs, including rollout collection and optimization."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


def choose_device(rows, repeats):
    rates={}
    for device in ("cpu","cuda"):
        jobs=[r for r in rows if r["device"]==device]
        if len(jobs)==repeats and all(r.get("completed") for r in jobs):
            rates[device]=statistics.median(r["decisions_per_second"] for r in jobs)
    if not rates: raise RuntimeError("No device completed the required real PPO benchmark jobs")
    return max(rates,key=rates.get), rates


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",required=True)
    ap.add_argument("--steps",type=int,default=256)
    ap.add_argument("--seconds",type=int,default=180)
    ap.add_argument("--repeats",type=int,default=2)
    args=ap.parse_args()
    if min(args.steps,args.seconds,args.repeats)<=0: ap.error("Benchmark budgets must be positive")
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for repeat in range(args.repeats):
        # Alternate ordering so the same device is not always tested first.
        for device in (("cpu","cuda") if repeat%2==0 else ("cuda","cpu")):
            folder=out/f"{device}-{repeat}"
            command=[sys.executable,"-m","training.train","--model","gru","--stage","movement",
                     "--device",device,"--envs","1","--steps",str(args.steps),"--seconds",str(args.seconds),
                     "--rollout","64","--sequence","16","--epochs","2","--seed",str(7+repeat),
                     "--cooldown-ms","0","--out",str(folder)]
            with (out/f"{device}-{repeat}.txt").open("w",encoding="utf-8") as log:
                start=time.perf_counter()
                proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                wall=time.perf_counter()-start
            row=dict(device=device,repeat=repeat,exit_code=proc.returncode,wall_seconds=wall,completed=False)
            summary=folder/"summary.json"
            if proc.returncode==0 and summary.exists():
                data=json.loads(summary.read_text())
                row.update(steps=data["steps"],updates=data["updates"],
                           decisions_per_second=data["invocation_steps"]/data["elapsed_seconds"],
                           completed=data["steps"]==args.steps and data["updates"]>0)
            rows.append(row)
            (out/"jobs.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
            print(json.dumps(row),flush=True)
    selected,rates=choose_device(rows,args.repeats)
    report=dict(kind="Real official-wrapper rollout + custom PPO benchmark",selected_device=selected,
                median_decisions_per_second=rates,jobs=rows,
                note="Same model/config/step budget; seeded trajectories can differ by device. Startup wall time is reported separately.")
    (out/"selection.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
