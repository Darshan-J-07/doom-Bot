"""Measured inference resources and engine throughput on this host, not evaluation hardware."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import threading
import time
import numpy as np
import psutil
import torch
from doom_env import make_env, observation_space, action_space
from doom_env import limits as L
from submission.agent import DoomAgent


def throughput(count, parallel=False, decisions=128):
    from training.workers import WorkerEnv
    envs=[]
    try:
        for _ in range(count):
            env=(WorkerEnv("training/maps/train/movement/00.wad",100000) if parallel else
                 make_env("training/maps/train/movement/00.wad",episode_ticks=100000,render_buffers=False))
            env.reset(seed=0); envs.append(env)
        rng=np.random.default_rng(1); started=time.perf_counter()
        for _ in range(decisions):
            actions=rng.integers(0,2,(count,9),dtype=np.int8)
            if parallel:
                for env,action in zip(envs,actions): env.send("step",action)
                for env in envs:
                    result=env.receive()
                    if result[2] or result[3]: env.reset(seed=0)
            else:
                for env,action in zip(envs,actions):
                    result=env.step(action)
                    if result[2] or result[3]: env.reset(seed=0)
        elapsed=time.perf_counter()-started
        return dict(envs=count,parallel=parallel,decisions=count*decisions,seconds=elapsed,
                    decisions_per_second=count*decisions/elapsed)
    finally:
        for env in envs: env.close()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--device",default="cpu")
    ap.add_argument("--steps",type=int,default=512); ap.add_argument("--throughput-only",action="store_true")
    ap.add_argument("--max-envs",type=int,choices=(1,2),default=2,
                    help="cap the worker benchmark at one environment on RAM-limited machines")
    args=ap.parse_args(); torch.set_num_threads(1)
    report={}
    if args.throughput_only:
        report["throughput"]=[throughput(1)]
        if args.max_envs==2: report["throughput"].extend([throughput(2),throughput(2,True)])
        target="benchmarks/throughput.json"
    else:
        process=psutil.Process(); rss=[]; stop=threading.Event()
        def sample():
            while not stop.wait(.01): rss.append(process.memory_info().rss)
        monitor=threading.Thread(target=sample,daemon=True); monitor.start()
        if args.device=="cuda": torch.cuda.reset_peak_memory_stats()
        t=time.perf_counter(); agent=DoomAgent(observation_space(),action_space(),args.device)
        if args.device=="cuda": torch.cuda.synchronize()
        init=time.perf_counter()-t
        t=time.perf_counter(); agent.reset(); reset=time.perf_counter()-t
        env=make_env("maps/two_rooms.wad",episode_ticks=max(2100,args.steps*4),render_buffers=False)
        latencies=[]
        try:
            obs,_=env.reset(seed=0)
            for _ in range(args.steps):
                t=time.perf_counter(); action=agent.act(obs)
                if args.device=="cuda": torch.cuda.synchronize()
                latencies.append((time.perf_counter()-t)*1000)
                assert env.action_space.contains(action)
                obs,_,term,trunc,_=env.step(action)
                if term or trunc: obs,_=env.reset(seed=0); agent.reset()
        finally:
            env.close(); stop.set(); monitor.join()
        peak_rss=max(rss+[process.memory_info().rss])
        report=dict(device=args.device,initialization_s=init,reset_s=reset,
                    decisions=len(latencies),first_act_ms=latencies[0],
                    latency_ms=dict(mean=float(np.mean(latencies)),peak=float(max(latencies)),
                                    p95=float(np.percentile(latencies,95))),
                    peak_process_rss_bytes=peak_rss,
                    windows_peak_working_set_bytes=getattr(process.memory_info(),"peak_wset",None),
                    peak_torch_vram_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=="cuda" else 0,
                    peak_torch_vram_reserved_bytes=torch.cuda.max_memory_reserved() if args.device=="cuda" else 0,
                    parameters=sum(p.numel() for p in agent.model.parameters()),
                    official_limits=dict(init_s=L.INIT_TIMEOUT_S,reset_s=L.RESET_TIMEOUT_S,
                                         first_act_s=L.FIRST_ACT_TIMEOUT_S,act_s=L.ACT_TIMEOUT_S,
                                         mean_ms=L.AGENT_MS_PER_STEP,ram_mb=L.MAX_RAM_MB,vram_mb=L.MAX_VRAM_MB),
                    scope="Local benchmark process includes game-wrapper imports; not organizer sandbox certification")
        target="benchmarks/performance.json"
    Path(target).write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))


if __name__=="__main__": main()
