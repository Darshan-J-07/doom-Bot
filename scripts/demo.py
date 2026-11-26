"""Capture real learned-policy gameplay from a held-out map, with raw frame output."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from doom_env import make_env
from submission.agent import DoomAgent
from training.rewards import EpisodeMetrics


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--stage",default="movement")
    ap.add_argument("--index",type=int,default=0); ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--ticks",type=int,default=2100); ap.add_argument("--out",default="recordings/demo")
    ap.add_argument("--device",default="cpu"); args=ap.parse_args()
    folder=Path(args.out); folder.mkdir(parents=True,exist_ok=True)
    rows=json.loads(Path("training/maps/manifest.json").read_text())["maps"]
    maps=[m for m in rows if m["split"]=="test" and m["stage"]==args.stage]
    spec=maps[args.index]
    env=make_env(spec["path"],episode_ticks=args.ticks)
    frames=[]
    try:
        agent=DoomAgent(env.observation_space,env.action_space,args.device)
        obs,_=env.reset(seed=args.seed,options={"record_path":str(folder/"episode.lmp")})
        tracker=EpisodeMetrics(); tracker.reset(env.privileged_state())
        while True:
            frames.append(obs["rgb"].copy())
            action=agent.act(obs)
            obs,_,term,trunc,_=env.step(action); tracker.step(env.privileged_state())
            if term or trunc: break
        result=dict(map=spec["path"],seed=args.seed,**tracker.summary())
        np.save(folder/"frames.npy",np.stack(frames))
        (folder/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        print(json.dumps(result))
    finally: env.close()


if __name__=="__main__": main()
