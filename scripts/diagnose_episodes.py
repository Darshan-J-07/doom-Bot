"""Replay the historical diagnostic split and report observed events."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from doom_env import make_env
from submission.model import GROUP_SIZES, SpatialPolicy, action_groups, buttons
from submission.preprocessing import tensor_batch
from training.maps import load_verified_maps
from training.rewards import EpisodeMetrics

TIERS = dict(movement=0, combat=1, exploration=2, keys=3, gauntlet=4)


def summarize(frames, metrics):
    counts = Counter()
    previous = None
    for row in frames:
        action = row["action"]
        hits = np.array(row["ray_types"])
        ranges = np.array(row["ray_ranges"]) * 1024
        if row["displacement"] < 2 and action[0] and np.any((hits[[63, 0, 1]] == 1)
                                                                & (ranges[[63, 0, 1]] < 40)):
            counts["forward_near_wall_low_motion"] += 1
        if (action[4] or action[5]) and row["displacement"] < 2:
            counts["turn_low_motion"] += 1
        if np.any((hits[[63, 0, 1]] == 2) & (ranges[[63, 0, 1]] <= 64)):
            counts["door_in_use_range_proxy"] += 1
            counts["door_use_rising_edge"] += bool(action[7] and not previous)
        if np.any((hits[[63, 0, 1]] == 3) & (ranges[[63, 0, 1]] <= 64)):
            counts["locked_door_in_use_range_proxy"] += 1
            counts["lock_use_rising_edge"] += bool(action[7] and not previous)
        if any(kind == 9 for kind in row["entities"]):
            counts["key_visible"] += 1
        if any(kind in (1, 2, 3) for kind in row["entities"]):
            counts["enemy_visible"] += 1
            counts["attack_with_enemy_visible"] += action[6]
        counts["selected_weapon_ammo_zero"] += row["vitals"][3] == 0
        counts["all_tracked_ammo_empty"] += not any(row["vitals"][4:8])
        previous = action[7]
    cells = [(int(np.floor(row["x"] / 96)), int(np.floor(row["y"] / 96))) for row in frames]
    transitions = [cell for i, cell in enumerate(cells) if i == 0 or cell != cells[i - 1]]
    counts["cell_revisits"] = len(transitions) - len(set(transitions))
    counts["short_cell_reversals"] = sum(a == c for a, c in zip(transitions, transitions[2:]))
    counts["longest_no_novel_cell_steps"] = longest_without_new_cell(cells)
    return dict(metrics, evidence=dict(counts))


def longest_without_new_cell(cells):
    seen = set()
    run = longest = 0
    for cell in cells:
        run = run + 1 if cell in seen else 0
        seen.add(cell)
        longest = max(longest, run)
    return longest


@torch.inference_mode()
def play(spec, seed, ticks, model, deterministic, trace, reset_memory=False, record=None):
    env = make_env(spec["path"], spec.get("map_name", "MAP01"), episode_ticks=ticks,
                   render_buffers=record is not None)
    frames, images = [], []
    tracker = EpisodeMetrics()
    try:
        options = {"record_path": str(record.with_suffix(".lmp"))} if record else None
        obs, _ = env.reset(seed=seed, options=options)
        state = env.privileged_state()
        tracker.reset(state)
        hidden, prev, start = model.initial_state(), torch.zeros(1, 9), torch.ones(1)
        generator = torch.Generator().manual_seed(seed)
        while True:
            if reset_memory:
                hidden.zero_()
            logits, _, hidden = model(tensor_batch([obs], "cpu"), hidden, prev, start)
            groups = action_groups(logits, deterministic=deterministic, generator=generator)
            action = buttons(groups)[0].numpy()
            ray = obs["rays"]
            entities = obs["visible_entities"]
            mask = entities["mask"].astype(bool)
            if record:
                images.append(obs["rgb"].copy())
            following, _, term, trunc, _ = env.step(action)
            new = env.privileged_state()
            reward = tracker.step(new)
            frame = dict(tic=state["tic"], x=state["x"], y=state["y"], angle=state["angle"],
                         displacement=float(np.hypot(new["x"]-state["x"], new["y"]-state["y"])),
                         action=action.tolist(), vitals=obs["vitals"].tolist(),
                         ray_types=ray["hit_type"].tolist(), ray_ranges=ray["distance"].tolist(),
                         entities=entities["entity_type"][mask].tolist(),
                         entity_features=entities["features"][mask].tolist(),
                         probabilities=[group.softmax(-1)[0].tolist() for group in logits.split(GROUP_SIZES, -1)],
                         reward=reward, keys=int(new["keys"].sum()), kills=new["kills"],
                         doors=len(tracker.opened), locks=len(tracker.unlocked),
                         exploration=len(tracker.visited),
                         outcome="exit" if new["exit_reached"] else "death" if new["dead"]
                         else "timeout" if trunc else None)
            frames.append(frame)
            if trace:
                trace.write(json.dumps(frame, separators=(",", ":")) + "\n")
            if term or trunc:
                break
            obs, state = following, new
            prev = torch.as_tensor(action[None]).float()
            start.zero_()
    finally:
        env.close()
    if record:
        np.save(record.with_suffix(".npy"), np.stack(images))
    return summarize(frames, dict(map=spec["path"], stage=spec["stage"], tier=TIERS[spec["stage"]],
                                 seed=seed, decision_count=len(frames), timeout=bool(trunc),
                                 **tracker.summary()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--stage", choices=tuple(TIERS))
    parser.add_argument("--index", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--ticks", type=int, default=2100)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--reset-memory", action="store_true")
    args = parser.parse_args()
    if args.record and not args.trace_dir:
        parser.error("--record requires --trace-dir")
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = SpatialPolicy(**saved["model_config"]).eval()
    model.load_state_dict(saved["model"])
    maps = [m for m in load_verified_maps() if m["split"] == args.split
            and (args.stage is None or m["stage"] == args.stage)]
    if args.index is not None:
        maps = [maps[args.index]]
    episodes = []
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
    for spec in maps:
        for seed in args.seeds:
            if args.trace_dir:
                path = args.trace_dir / f"{spec['stage']}-{Path(spec['path']).stem}-{seed}"
                with gzip.open(path.with_suffix(".jsonl.gz"), "wt") as trace:
                    row = play(spec, seed, args.ticks, model, saved["deterministic"], trace,
                               args.reset_memory, path if args.record else None)
            else:
                row = play(spec, seed, args.ticks, model, saved["deterministic"], None, args.reset_memory)
            episodes.append(row)
            print(json.dumps(row), flush=True)
    report = dict(checkpoint=str(args.checkpoint), split=args.split, diagnostic_only=True, episodes=episodes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
