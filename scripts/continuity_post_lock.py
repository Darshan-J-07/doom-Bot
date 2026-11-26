"""Train post-lock exit progression with fixed on-policy retention lanes."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil

from scripts.adaptive_mixed import OUT as ARCHIVED_RUN, MixedSchedule, checked, load_reward, pin, read, write


WARM_SHA256 = "1810650838aea8b230cb9ad34dbc4da358afb05b4934b0ebe392739ce4305724"
IMPROVED_LOCK_SHA256 = "e9e42ca9217811b7bd8b0c2987667b73533c0f4c97c40dc5cb73198a70cbd80d"


class PostLockSchedule(MixedSchedule):
    def __init__(self, seed, *, warm_start, start_steps, focus):
        super().__init__(seed, warm_start=warm_start)
        self.start_steps, self.focus = start_steps, focus

    def catalog(self, maps):
        super().catalog(maps)
        self.pools["approach"] = [row for row in self.pools["approach"] if row.get("post_lock")]
        if not self.pools["approach"]:
            raise ValueError("The approach lanes need certified post-lock practice.")
        self.post_lock_pools = [self.pools["approach"]]
        fresh = [row for row in self.pools["approach"] if row.get("continuity_post_lock")]
        if fresh:
            landing = [row for row in fresh if row["family"] == "continuity-landing"]
            corners = [row for row in fresh if row["family"] in {"continuity-landing", "continuity-elbow"}]
            self.post_lock_pools = [landing, corners, fresh, self.pools["approach"]]

    def choose(self, maps, rng, steps):
        if self.pools is None:
            self.catalog(maps)
        if len(self.post_lock_pools) == 1 or self.focus == "balanced":
            pool = self.post_lock_pools[-1]
        elif self.focus == "landing":
            pool = self.post_lock_pools[0]
        elif self.focus == "route":
            pool = self.post_lock_pools[2]
        else:
            added = steps - self.start_steps
            stage = sum(added >= cutoff for cutoff in (15_360, 39_936, 81_920))
            pool = self.post_lock_pools[stage]
        self.pools["approach"] = pool
        return super().choose(maps, rng, steps)


def learner(args):
    import torch
    from training import train

    archive = read(ARCHIVED_RUN / "mixed-plan.json")
    base = archive["inputs"]
    original = checked(base["warm_start"])
    config = checked(base["config"])
    reward = checked(base["reward"])
    resume = (args.resume or original).resolve(strict=True)
    receipt = pin(resume)
    previous_prefix = ARCHIVED_RUN / "training" / "bc-ppo-59-12000"
    for baseline, expected in ((original, WARM_SHA256),
                               (previous_prefix / "step-326496.pt", IMPROVED_LOCK_SHA256)):
        if resume == baseline and receipt["sha256"] != expected:
            raise ValueError(f"The frozen starting checkpoint changed: {resume}")
    saved = torch.load(resume, map_location="cpu", weights_only=True)
    start = saved["trained_steps"]
    if not saved.get("optimizer", {}).get("state"):
        raise ValueError("A PPO learner resume needs real Adam optimizer state.")
    if saved["model_config"] != {"hidden_size": 128, "recurrent": False, "entity_mode": "pool"}:
        raise ValueError("The post-lock comparison requires the archived MLP architecture.")
    next_seed = saved.get("args", {}).get("seed")
    restart = args.restart_rng or resume == original or next_seed != args.seed
    if args.map_root:
        root = args.map_root.resolve(strict=True)
        qa = read(root / "qa.json")
        manifest = checked(qa["manifest"])
        if manifest != root / "manifest.json" or not qa.get("verified"):
            raise ValueError("The selected dataset must have successful post-lock engine verification.")
    else:
        manifest = checked(base["training"])
        root = manifest.parent
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    options = ["--config", config, "--map-root", root,
               "--resume", resume, "--seed", args.seed, "--steps", start + args.decisions,
               "--seconds", args.seconds, "--out", output,
               "--checkpoint-every", args.checkpoint_every, "--rollout", args.rollout,
               "--visualize" if args.visualize else "--headless"]
    if args.stop_file is not None:
        options += ["--stop-file", args.stop_file.resolve()]
    if restart:
        options.append("--restart-rng")
    if args.lr is not None:
        options += ["--lr", args.lr]
    if args.visualize:
        os.environ.setdefault("SDL_VIDEODRIVER", "x11")
        options += ["--speed", args.speed]
    options = list(map(str, options))
    from training.continuity_post_lock_metrics import instrument

    schedule = PostLockSchedule(args.seed, warm_start=restart, start_steps=start, focus=args.focus)
    write(output / "continuity-invocation.json", dict(resume=receipt, manifest=pin(manifest),
          config=pin(config), reward=pin(reward), learner_source=pin(Path(__file__)),
          seed=args.seed, rng_restarted=restart, start_steps=start, start_updates=saved["updates"],
          post_lock_focus=args.focus, options=options))
    del saved
    reward_class = instrument(load_reward(reward))
    if args.active_seconds is None:
        train.main(options, select_training_map=schedule.choose, course_reward=reward_class)
    else:
        from training.learning_meter import run

        run(options, meter_id=args.meter_id or output.name, active_limit_seconds=args.active_seconds,
            select_training_map=schedule.choose, course_reward=reward_class)
    result = read(output / "summary.json")
    if result["invocation_updates"]:
        numbered = output / f"step-{result['steps']}.pt"
        if not numbered.exists():
            temporary = output / f".{numbered.name}.tmp"
            shutil.copyfile(output / "latest.pt", temporary)
            temporary.replace(numbered)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "learner"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--map-root", type=Path)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--seed", type=int, default=59)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--active-seconds", type=float)
    parser.add_argument("--meter-id")
    parser.add_argument("--decisions", type=int, default=1_000_000)
    parser.add_argument("--checkpoint-every", type=int, default=60_000)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--focus", choices=("staged", "balanced", "landing", "route"), default="staged")
    parser.add_argument("--restart-rng", action="store_true")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--speed", type=float, default=1)
    parser.add_argument("--per-family", type=int, default=12)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or args.decisions < 8 or args.decisions % 8:
        parser.error("Choose a finite positive wall allowance and a decision cap divisible by eight.")
    if args.active_seconds is not None and (not math.isfinite(args.active_seconds) or args.active_seconds <= 0):
        parser.error("Choose a finite positive active learning limit.")
    if args.phase == "prepare":
        from training.continuity_post_lock_maps import prepare

        if args.per_family < 4:
            parser.error("Map preparation needs at least four layouts per family.")
        prepare(args.out.resolve(), args.seed, args.per_family)
    else:
        learner(args)


if __name__ == "__main__":
    main()
