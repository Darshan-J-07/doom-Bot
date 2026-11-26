"""Fixed-split local proxy evaluation and explicit validation-only checkpoint selection."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from doom_env import CONTRACT_VERSION, make_env
from submission.model import SpatialPolicy, action_groups, buttons
from submission.preprocessing import tensor_batch
from training.rewards import EpisodeMetrics
from training.maps import load_verified_maps


def decision_mode(checkpoint, override=None):
    return bool(checkpoint.get("deterministic", False)) if override is None else bool(override)


@torch.inference_mode()
def evaluate(checkpoint_path, maps, seeds=(0,), ticks=2100, deterministic=None, device="cpu", record_dir=None, reset_memory=False, policy_seed=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["contract_version"] != CONTRACT_VERSION or checkpoint["trained_steps"] <= 0:
        raise ValueError("Not a trained checkpoint for this contract")
    deterministic = decision_mode(checkpoint, deterministic)
    model = SpatialPolicy(**checkpoint["model_config"]).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    episodes, latencies = [], []
    for spec in maps:
        for seed in seeds:
            env = make_env(spec["path"], episode_ticks=ticks, render_buffers=record_dir is not None)
            try:
                options = None
                if record_dir:
                    folder = Path(record_dir); folder.mkdir(parents=True, exist_ok=True)
                    options = {"record_path": str(folder / f'{spec["stage"]}_{Path(spec["path"]).stem}_{seed}.lmp')}
                obs, _ = env.reset(seed=seed, options=options)
                metrics = EpisodeMetrics(); metrics.reset(env.privileged_state())
                hidden = model.initial_state(1, device)
                prev = torch.zeros(1, 9, device=device); starts = torch.ones(1, device=device)
                generator = torch.Generator(device=device).manual_seed(seed if policy_seed is None else policy_seed)
                count = 0
                while True:
                    t = time.perf_counter()
                    if reset_memory:
                        hidden.zero_()
                    logits, _, hidden = model(tensor_batch([obs], device), hidden, prev, starts)
                    groups = action_groups(logits, deterministic=deterministic, generator=generator)
                    action = buttons(groups)[0].cpu().numpy()
                    if device == "cuda": torch.cuda.synchronize()
                    latencies.append((time.perf_counter()-t)*1000)
                    if not env.action_space.contains(action): raise ValueError("Invalid neural action")
                    obs, _, terminated, truncated, _ = env.step(action)
                    metrics.step(env.privileged_state())
                    prev = torch.as_tensor(action[None], device=device).float(); starts.zero_()
                    count += 1
                    if terminated or truncated: break
                episodes.append(dict(map=spec["path"], stage=spec["stage"], seed=seed,
                                     decisions=count, **metrics.summary()))
            finally:
                env.close()
    mean = {k: float(np.mean([e[k] for e in episodes])) for k in
            ("proxy_score", "exit", "survived", "kills", "keys", "doors", "locks", "unique_cells", "damage")}
    return dict(checkpoint=str(checkpoint_path), sha256=hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
                trained_steps=checkpoint["trained_steps"], model_config=checkpoint["model_config"],
                deterministic=deterministic, ticks=ticks, seeds=list(seeds), episodes=episodes,
                mean=mean, latency_ms=dict(mean=float(np.mean(latencies)), peak=float(max(latencies)),
                                         p95=float(np.percentile(latencies,95))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", action="append", required=True)
    ap.add_argument("--split", choices=("validation", "test"), default="validation")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--ticks", type=int, default=2100)
    ap.add_argument("--per-stage", type=int, default=2)
    ap.add_argument("--stages", nargs="+")
    ap.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None,
                    help="default: preserve the checkpoint's saved decision mode")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--out", default="benchmarks/validation.json")
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--record-dir")
    args = ap.parse_args(); torch.set_num_threads(1)
    if args.promote and args.split != "validation":
        ap.error("Only validation maps may select the submitted checkpoint; test maps are reporting only")
    manifest = load_verified_maps()
    maps = []; counts = {}
    for m in manifest:
        if m["split"] != args.split or (args.stages and m["stage"] not in args.stages): continue
        counts[m["stage"]] = counts.get(m["stage"], 0) + 1
        if counts[m["stage"]] <= args.per_stage: maps.append(m)
    if not maps: ap.error("No maps selected")
    results = []
    for checkpoint in args.checkpoint:
        result = evaluate(checkpoint, maps, args.seeds, args.ticks, args.deterministic, args.device, args.record_dir)
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k != "episodes"}), flush=True)
    best = max(results, key=lambda r:(r["mean"]["proxy_score"], r["mean"]["exit"]))
    report = dict(kind="LOCAL PROXY; not official scoring", split=args.split, results=results,
                  selected=best["checkpoint"] if args.promote else None)
    if args.promote:
        full = torch.load(best["checkpoint"], map_location="cpu", weights_only=True)
        inference = {k:full[k] for k in ("model", "model_config", "contract_version", "trained_steps", "updates", "provenance")}
        inference.update(deterministic=best["deterministic"], selection=best["mean"], selection_split="validation")
        path = Path("submission/weights/best_model.pt"); path.parent.mkdir(exist_ok=True)
        torch.save(inference, path)
        report["promoted_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
