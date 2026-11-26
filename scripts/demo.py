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
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--headless", action="store_true", help="Record without opening a window")
    args = ap.parse_args()
    folder=Path(args.out); folder.mkdir(parents=True,exist_ok=True)
    rows=json.loads(Path("training/maps/manifest.json").read_text())["maps"]
    maps=[m for m in rows if m["split"]=="test" and m["stage"]==args.stage]
    spec=maps[args.index]
    env=make_env(spec["path"],episode_ticks=args.ticks)
    frames=[]
    display = None
    pygame = None
    try:
        if not args.headless:
            import pygame

            pygame.display.init()
            display = pygame.display.set_mode((640, 480))
            pygame.display.set_caption("DOOM NEXUS")
            clock = pygame.time.Clock()
        stopped = False
        agent=DoomAgent(env.observation_space,env.action_space,args.device)
        obs,_=env.reset(seed=args.seed,options={"record_path":str(folder/"episode.lmp")})
        tracker=EpisodeMetrics(); tracker.reset(env.privileged_state())
        while True:
            if display is not None:
                if any(event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ) for event in pygame.event.get()):
                    stopped = True
                    break
            frame = obs["rgb"].copy()
            frames.append(frame)
            if display is not None:
                surface = pygame.image.frombuffer(frame, (frame.shape[1], frame.shape[0]), "RGB")
                display.blit(pygame.transform.scale(surface, display.get_size()), (0, 0))
                pygame.display.flip()
                clock.tick(9)  # Approximately 35 game tics/sec at four tics per decision.
            action=agent.act(obs)
            obs,_,term,trunc,_=env.step(action); tracker.step(env.privileged_state())
            if term or trunc: break
        result=dict(map=spec["path"],seed=args.seed,stopped=stopped,**tracker.summary())
        if frames:
            np.save(folder/"frames.npy", np.stack(frames))
        (folder/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        print(json.dumps(result))
    finally:
        try:
            env.close()
        finally:
            if pygame is not None:
                pygame.display.quit()


if __name__=="__main__": main()
