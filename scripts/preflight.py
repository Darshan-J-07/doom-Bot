"""Verify Python/CUDA, existing map bytes and real complete engine episodes before training."""
from __future__ import annotations
import argparse
import collections
import importlib.metadata as md
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import numpy as np
import torch
from doom_env import make_env
from training.maps import load_verified_maps


def complete_episode(path,ticks,seed):
    started=time.perf_counter()
    env=make_env(path,episode_ticks=ticks,render_buffers=False)
    try:
        obs,_=env.reset(seed=seed); env.action_space.seed(seed)
        if not env.observation_space.contains(obs): raise ValueError("Reset observation violates official space")
        decisions=0
        while True:
            # Uniform actions diagnose the engine; they are not a submitted agent or training.
            action=env.action_space.sample()
            obs,reward,term,trunc,info=env.step(action)
            if not env.observation_space.contains(obs) or not np.isfinite(reward):
                raise ValueError("Step observation/reward violates the contract")
            decisions+=1
            if term or trunc: break
            if decisions>ticks//4+2: raise RuntimeError("Engine did not end at the configured tick cap")
        return dict(map=str(path),seed=seed,decisions=decisions,terminated=bool(term),truncated=bool(trunc),
                    info=info,seconds=time.perf_counter()-started)
    finally: env.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine",action="store_true")
    ap.add_argument("--require-cuda",action="store_true")
    ap.add_argument("--out",default="benchmarks/continuation/preflight.json")
    args=ap.parse_args(); torch.set_num_threads(1)
    report=dict(passed=False,platform=platform.platform(),python=sys.version,vizdoom=md.version("vizdoom"),
                torch=torch.__version__,cuda_available=torch.cuda.is_available(),engine_requested=args.engine,
                episodes=[],training_performed=False)
    try:
        if sys.version_info[:2]!=(3,13): raise RuntimeError("Competition requires Python 3.13")
        if report["vizdoom"]!="1.3.3": raise RuntimeError("Competition requires ViZDoom 1.3.3")
        if report["cuda_available"]:
            props=torch.cuda.get_device_properties(0)
            torch.cuda.set_per_process_memory_fraction(.5)
            a=torch.arange(128*128,dtype=torch.float32).reshape(128,128)/1000
            actual=(a.cuda()@a.cuda().T).cpu()
            if not torch.allclose(actual,a@a.T,rtol=1e-3,atol=.01): raise RuntimeError("CUDA compute verification failed")
            report["cuda_device"]=dict(name=props.name,total_memory_bytes=props.total_memory,
                                      capability=list(torch.cuda.get_device_capability()),tensor_compute_passed=True)
        elif args.require_cuda:
            raise RuntimeError("CUDA was requested but no usable CUDA device is available in this environment")
        rows=load_verified_maps()
        report["maps"]=dict(count=len(rows),split_counts=dict(collections.Counter(r["split"] for r in rows)),
                            actual_hashes_verified=True,split_bytes_disjoint=True)
        if args.engine:
            report["episodes"].append(complete_episode("maps/two_rooms.wad",2100,0))
            for i,row in enumerate(rows):
                report["episodes"].append(complete_episode(row["path"],40,1000+i))
        report["passed"]=True
    except Exception as exc:
        report["error"]=f"{type(exc).__name__}: {exc}"
        report["traceback"]=traceback.format_exc()
    path=Path(args.out); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))
    if not report["passed"]: raise SystemExit(1)


if __name__=="__main__": main()
