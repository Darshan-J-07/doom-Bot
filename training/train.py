"""Real ViZDoom rollout collection, custom recurrent PPO, checkpoint and resume CLI."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from doom_env import CONTRACT_VERSION, make_env
from submission.model import SpatialPolicy, buttons, policy_stats
from submission.preprocessing import tensor_batch
from training.generate_maps import STAGES
from training.maps import load_verified_maps
from training.ppo import update
from training.rewards import EpisodeMetrics


def curriculum_progress(steps, target, elapsed, seconds, clock="steps"):
    value = elapsed / seconds if clock == "time" else steps / target
    return max(0., min(1., value))


def model_cuda_rng(model):
    # A CPU training run must not initialize CUDA just to serialize its checkpoint.
    return torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else []


def save_checkpoint(path, model, optimizer, steps, updates, args, rng):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(model=model.state_dict(), model_config=model.config,
                   contract_version=CONTRACT_VERSION, trained_steps=steps, updates=updates,
                   optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                   cuda_rng=model_cuda_rng(model),
                   map_rng=rng.getstate(), args=vars(args), deterministic=False,
                   provenance="PPO optimizer updates from official doom_env rollouts")
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary); temporary.replace(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", help="JSON defaults; explicit CLI flags override config")
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--envs", type=int, default=2)
    ap.add_argument("--parallel", action="store_true", help="spawn independent engine workers; benchmark RAM first")
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--sequence", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy", type=float, default=.01)
    ap.add_argument("--ticks", type=int, default=700)
    ap.add_argument("--stage", choices=("auto", *STAGES, "mixed"), default="auto")
    ap.add_argument("--curriculum-clock", choices=("steps", "time"), default="steps")
    ap.add_argument("--cooldown-ms", type=float, default=0,
                    help="optional idle interval after each update; not a temperature limit")
    ap.add_argument("--gpu-memory-fraction", type=float, default=.45)
    ap.add_argument("--model", choices=("mlp", "gru"), default="gru")
    ap.add_argument("--entity-mode", choices=("pool", "none"), default="pool")
    ap.add_argument("--map-root", default="training/maps")
    ap.add_argument("--out", default="runs/gru")
    ap.add_argument("--resume")
    preliminary, _ = ap.parse_known_args(argv)
    if preliminary.config:
        defaults = json.loads(Path(preliminary.config).read_text())
        unknown = set(defaults) - {a.dest for a in ap._actions}
        if unknown: ap.error(f"Unknown config settings: {sorted(unknown)}")
        ap.set_defaults(**defaults)
    args = ap.parse_args(argv)
    if min(args.steps, args.seconds, args.envs, args.rollout, args.sequence, args.epochs) <= 0:
        ap.error("budgets, batch sizes and worker counts must be positive")
    if args.cooldown_ms < 0 or not 0 < args.gpu_memory_fraction <= 1:
        ap.error("cooldown must be nonnegative and GPU memory fraction must be in (0,1]")
    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but this PyTorch build has no available CUDA device")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)
    maps = load_verified_maps(args.map_root)
    train_maps = [m for m in maps if m["split"] == "train"]
    model = SpatialPolicy(recurrent=args.model == "gru", entity_mode=args.entity_mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    steps = updates = 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved["model_config"] != model.config or saved["contract_version"] != CONTRACT_VERSION:
            raise ValueError("Resume architecture / contract mismatch")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        steps, updates = saved["trained_steps"], saved["updates"]
        if args.steps <= steps:
            raise ValueError(f"Resume --steps must exceed checkpoint lifetime steps ({steps})")
        torch.set_rng_state(saved["torch_rng"]); rng.setstate(saved["map_rng"])
        if device.type == "cuda" and saved.get("cuda_rng"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        # Resume starts fresh episodes: engine state and recurrent state are not serialized.
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    def new_env(progress):
        stage_idx = min(4, int(progress * 5))
        allowed = STAGES[:stage_idx+1] if args.stage == "auto" else STAGES if args.stage == "mixed" else (args.stage,)
        spec = rng.choice([m for m in train_maps if m["stage"] in allowed])
        if args.parallel:
            from training.workers import WorkerEnv
            env = WorkerEnv(spec["path"], args.ticks)
        else:
            env = make_env(spec["path"], episode_ticks=args.ticks, render_buffers=False)
        try:
            obs, _ = env.reset(seed=rng.randrange(2**30))
        except BaseException:
            env.close()
            raise
        metrics = EpisodeMetrics(); metrics.reset(env.privileged_state())
        return env, obs, metrics
    envs, observations, trackers = [], [], []
    started = time.perf_counter(); first_steps = steps
    def progress():
        return curriculum_progress(steps, args.steps, time.perf_counter()-started,
                                   args.seconds, args.curriculum_clock)
    try:
        for _ in range(args.envs):
            env, obs, tracker = new_env(progress())
            envs.append(env); observations.append(obs); trackers.append(tracker)
        hidden = model.initial_state(args.envs, device)
        previous_action = torch.zeros(args.envs, 9, device=device)
        starts = torch.ones(args.envs, device=device)
        with (out / "learning.jsonl").open("a", encoding="utf-8") as log:
            while steps < args.steps and time.perf_counter() - started < args.seconds:
                data = {k: [] for k in ("hidden", "previous_action", "starts", "actions", "log_probs",
                        "values", "rewards", "next_values", "terminated", "truncated")}
                packed_seq, episodes = [], []
                for _ in range(min(args.rollout, max(1, (args.steps-steps)//args.envs))):
                    packed = tensor_batch(observations, device)
                    data["hidden"].append(hidden.detach()); data["starts"].append(starts.clone())
                    data["previous_action"].append(previous_action.clone())
                    packed_seq.append(packed)
                    with torch.no_grad():
                        logits, values, next_hidden = model(packed, hidden, previous_action, starts)
                        groups, log_probs, _ = policy_stats(logits)
                        action = buttons(groups)
                    following, rewards, terminated, truncated = [], [], [], []
                    action_array = action.cpu().numpy()
                    if args.parallel:
                        for i, env in enumerate(envs): env.send("step",action_array[i])
                    for i, env in enumerate(envs):
                        obs, _, term, trunc, _ = (env.receive() if args.parallel else env.step(action_array[i]))
                        following.append(obs)
                        rewards.append(trackers[i].step(env.privileged_state()))
                        terminated.append(term); truncated.append(trunc)
                    with torch.no_grad():
                        _, next_values, _ = model(tensor_batch(following, device), next_hidden,
                            action.float(), torch.zeros(args.envs, device=device))
                    for k, value in (("actions", groups), ("log_probs", log_probs), ("values", values),
                        ("next_values", next_values), ("rewards", torch.tensor(rewards, device=device, dtype=torch.float32)),
                        ("terminated", torch.tensor(terminated, device=device)),
                        ("truncated", torch.tensor(truncated, device=device))):
                        data[k].append(value.detach())
                    steps += args.envs
                    done = np.array(terminated) | np.array(truncated)
                    for i in np.flatnonzero(done):
                        episodes.append(trackers[i].summary()); envs[i].close()
                        envs[i], following[i], trackers[i] = new_env(progress())
                    observations = following
                    hidden = next_hidden.detach()
                    starts = torch.as_tensor(done, device=device, dtype=torch.float32)
                    previous_action = action.float()
                batch = {k: torch.stack(v) for k, v in data.items()}
                batch["obs"] = {k: torch.stack([p[k] for p in packed_seq]) for k in packed_seq[0]}
                optimizer.param_groups[0]["lr"] = args.lr * max(.1, 1 - steps / args.steps)
                stats = update(model, optimizer, batch, epochs=args.epochs,
                               sequence_length=args.sequence, entropy_coef=args.entropy)
                updates += 1
                elapsed = time.perf_counter()-started
                row = dict(steps=steps, updates=updates, seconds=elapsed,
                           decisions_per_second=(steps-first_steps)/elapsed, **stats, episodes=episodes)
                log.write(json.dumps(row)+"\n"); log.flush()
                save_checkpoint(out / "latest.pt", model, optimizer, steps, updates, args, rng)
                print(json.dumps({k:v for k,v in row.items() if k != "episodes"}), flush=True)
                if args.cooldown_ms:
                    time.sleep(args.cooldown_ms / 1000)
    finally:
        for env in envs: env.close()
    if updates == 0:
        raise RuntimeError("No training update completed; no trained artifact was produced")
    summary = dict(steps=steps, updates=updates, elapsed_seconds=time.perf_counter()-started,
                   checkpoint=str(out/"latest.pt"), parameters=sum(p.numel() for p in model.parameters()),
                   device=str(device), resumed_from=args.resume, invocation_steps=steps-first_steps,
                   cooldown_ms=args.cooldown_ms, curriculum_clock=args.curriculum_clock,
                   torch_version=torch.__version__,
                   peak_torch_vram_reserved_bytes=torch.cuda.max_memory_reserved() if device.type=="cuda" else 0)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__": main()
