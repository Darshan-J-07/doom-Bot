"""Real ViZDoom rollout collection, custom recurrent PPO, checkpoint and resume CLI."""
from __future__ import annotations
import argparse
from collections import deque
from hashlib import sha256
import json
from pathlib import Path
import random
import signal
from tempfile import NamedTemporaryFile
import time
import numpy as np
import torch
import vizdoom
from doom_env import CONTRACT_VERSION, make_env
from training.curriculum import AdaptiveCurriculum
from submission.model import MEMORY_KINDS, VISION_KINDS, SpatialPolicy, buttons, policy_stats
from submission.preprocessing import tensor_batch
from training.generate_maps import STAGES
from training.maps import load_verified_maps
from training.mechanics import CourseReward
from training.ppo import update
from training.rewards import CompetitionReward, EpisodeMetrics, ProgressReward
from training.skills import SkillReward, skill_state
from training.universe_reward import UniverseReward
from training.visualization import (
    WINDOW_EXIT_ERRORS, Playback, add_playback_arguments, make_visible_env, validate_playback_arguments,
)


def curriculum_progress(steps, target, elapsed, seconds, clock="steps"):
    value = elapsed / seconds if clock == "time" else steps / target
    return max(0., min(1., value))


def model_cuda_rng(model):
    # A CPU training run must not initialize CUDA just to serialize its checkpoint.
    return torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else []


def save_checkpoint(path, model, optimizer, steps, updates, args, rng, *, curriculum=None, dataset_sha256=None,
                    worker_rngs=()):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(model=model.state_dict(), model_config=model.config,
                   contract_version=CONTRACT_VERSION, trained_steps=steps, updates=updates,
                   optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                   cuda_rng=model_cuda_rng(model),
                   map_rng=rng.getstate(), args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   curriculum=curriculum, dataset_sha256=dataset_sha256,
                   deterministic=False,
                   provenance="PPO optimizer updates from official doom_env rollouts")
    if worker_rngs:
        payload["worker_map_rngs"] = [worker_rng.getstate() for worker_rng in worker_rngs]
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp",
                            delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None, *, select_training_map=None, course_reward=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", help="JSON defaults; explicit CLI flags override config")
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--envs", type=int, default=2)
    ap.add_argument("--parallel", action="store_true", help="spawn independent engine workers; benchmark RAM first")
    add_playback_arguments(ap)
    ap.add_argument("--visible-env", type=int, default=0,
                    help="index of the single training environment to show")
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--sequence", type=int, default=32)
    ap.add_argument("--burn-in", type=int, default=0,
                    help="replay this much recurrent context before subsequences and after updates")
    ap.add_argument("--sequences-per-batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy", type=float, default=.01)
    ap.add_argument("--gamma", type=float, default=.99, help="discount per agent decision")
    ap.add_argument("--gae-lambda", type=float, default=.95)
    ap.add_argument("--lr-decay-steps", type=int, default=0,
                    help="lifetime LR schedule horizon, independent of this invocation's step limit")
    ap.add_argument("--finite-horizon", action="store_true",
                    help="zero the value bootstrap at the actual episode cap")
    ap.add_argument("--audit-rollouts", action="store_true", help="verify log-probability replay before the first update")
    ap.add_argument("--initial-priors", action="store_true", help="initialize fresh utility logits toward idle")
    ap.add_argument("--ticks", type=int, default=700)
    ap.add_argument("--stage", choices=("auto", *STAGES, "mixed", "adaptive"), default="auto")
    ap.add_argument("--curriculum-clock", choices=("steps", "time"), default="steps")
    ap.add_argument("--mastery", type=float, default=.72)
    ap.add_argument("--curriculum-window", type=int, default=40)
    ap.add_argument("--curriculum-minimum", type=int, default=12)
    ap.add_argument("--curriculum-probe", type=float, default=.12)
    ap.add_argument("--curriculum-feedback", type=Path, help="milestone validation JSON for curriculum gating")
    ap.add_argument("--cooldown-ms", type=float, default=0,
                    help="optional idle interval after each update; not a temperature limit")
    ap.add_argument("--gpu-memory-fraction", type=float, default=.45)
    ap.add_argument("--model", choices=("mlp", "gru", *MEMORY_KINDS), default="gru")
    ap.add_argument("--vision", choices=VISION_KINDS, help="enable the residual pixel encoder")
    ap.add_argument("--render-buffers", action="store_true",
                    help="produce contract image buffers even for the structured control")
    ap.add_argument("--paired-sampling", action="store_true",
                    help="give each environment its own seeded map and engine-seed stream")
    ap.add_argument("--entity-mode", choices=("pool", "none", "attention"), default="pool")
    ap.add_argument("--map-root", default="training/maps")
    ap.add_argument("--max-drill-level", type=int, help="limit a focused training stage to this difficulty level")
    ap.add_argument("--out", default="runs/gru")
    ap.add_argument("--resume")
    ap.add_argument("--restart-rng", action="store_true", help="start a new sampling seed while restoring weights and optimizer")
    ap.add_argument("--reward", choices=("legacy", "progress", "skills", "universe", "competition", "course", "mechanics"), default="legacy")
    ap.add_argument("--trigger-discipline", action="store_true",
                    help="add bounded training costs for attacking without a visible enemy")
    ap.add_argument("--potential-weight", type=float, default=1.)
    ap.add_argument("--novelty-weight", type=float, default=.06)
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="decision interval for preserving milestone copies")
    ap.add_argument("--checkpoint-steps", type=int, nargs="*", default=[], help="exact cumulative decision milestones")
    ap.add_argument("--min-available-mib", type=int, default=0,
                    help="stop safely at an update boundary below this system memory floor")
    ap.add_argument("--stop-file", type=Path, help="stop after the current update when this file exists")
    preliminary, _ = ap.parse_known_args(argv)
    if preliminary.config:
        defaults = json.loads(Path(preliminary.config).read_text())
        unknown = set(defaults) - {a.dest for a in ap._actions}
        if unknown: ap.error(f"Unknown config settings: {sorted(unknown)}")
        ap.set_defaults(**defaults)
    args = ap.parse_args(argv)
    validate_playback_arguments(ap, args)
    if args.vision and not args.render_buffers:
        ap.error("Visual models require --render-buffers")
    if args.paired_sampling and args.stage in ("auto", "adaptive"):
        ap.error("Paired sampling requires a fixed stage or --stage mixed")
    if args.trigger_discipline and args.reward != "mechanics":
        ap.error("--trigger-discipline requires --reward mechanics")
    if min(args.steps, args.seconds, args.envs, args.rollout, args.sequence, args.epochs, args.sequences_per_batch) <= 0:
        ap.error("budgets, batch sizes and worker counts must be positive")
    if args.visualize and not 0 <= args.visible_env < args.envs:
        ap.error("--visible-env must be an index from 0 to --envs minus one")
    if args.cooldown_ms < 0 or not 0 < args.gpu_memory_fraction <= 1:
        ap.error("cooldown must be nonnegative and GPU memory fraction must be in (0,1]")
    if not 0 < args.gamma <= 1 or not 0 <= args.gae_lambda <= 1:
        ap.error("--gamma must be in (0,1] and --gae-lambda in [0,1]")
    if min(args.lr_decay_steps, args.potential_weight, args.novelty_weight, args.entropy, args.burn_in) < 0 or args.lr <= 0:
        ap.error("Reward coefficients, schedules and entropy must be nonnegative, with a positive LR")
    if args.model == "mlp" and args.burn_in:
        ap.error("--burn-in requires a stateful model")
    if min(args.checkpoint_every, args.min_available_mib) < 0:
        ap.error("checkpoint interval and memory floor must be nonnegative")
    if args.min_available_mib:
        import psutil
    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but this PyTorch build has no available CUDA device")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)
    worker_rngs = [random.Random(f"{args.seed}:{i}") for i in range(args.envs)] if args.paired_sampling else []
    maps = load_verified_maps(args.map_root)
    train_maps = [m for m in maps if m["split"] == "train"]
    world = args.reward == "universe"
    skills = args.reward == "skills"
    course = args.reward in ("course", "mechanics")
    if select_training_map is not None and (args.stage != "mixed" or not args.paired_sampling):
        ap.error("A custom practice schedule needs mixed stages and paired sampling")
    reward_class = CourseReward if course_reward is None else course_reward
    if args.max_drill_level is not None:
        if args.max_drill_level < 0:
            ap.error("--max-drill-level must be nonnegative")
        train_maps = [row for row in train_maps if row.get("level", 0) <= args.max_drill_level]
    shaped = skills or world or course
    if skills and (not train_maps or any("drill" not in row for row in train_maps)):
        ap.error("--reward skills requires a skill drill curriculum from training.build_skill_curriculum")
    if (world or course) and (not train_maps or any("targets" not in row.get("drill", {}) for row in train_maps)):
        ap.error("This reward requires a curriculum with objective targets")
    if args.stage == "adaptive" and not (world or course):
        ap.error("--stage adaptive requires objective metrics from --reward universe, course or mechanics")
    finite = args.finite_horizon or world or course or args.reward == "competition"
    sampler = AdaptiveCurriculum(train_maps, threshold=args.mastery, window=args.curriculum_window,
                                 minimum=args.curriculum_minimum, probe=args.curriculum_probe) if args.stage == "adaptive" else None
    dataset_digest = sha256((Path(args.map_root) / "manifest.json").read_bytes()).hexdigest()
    model = SpatialPolicy(recurrent=args.model != "mlp", entity_mode=args.entity_mode,
                          memory=args.model if args.model in MEMORY_KINDS else None, vision=args.vision).to(device)
    if args.initial_priors and not args.resume:
        with torch.no_grad():
            model.actor.bias.copy_(model.actor.bias.new_tensor(
                [0., .45, -.45, .50, 0., 0., .80, 0., 0., .25, 0., .75, 0., 1.5, 0.]))
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
        if args.restart_rng:
            torch.manual_seed(args.seed)
        else:
            torch.set_rng_state(saved["torch_rng"])
            rng.setstate(saved["map_rng"])
            if worker_rngs:
                states = saved.get("worker_map_rngs", ())
                if len(states) != len(worker_rngs):
                    raise ValueError("Paired sampling resume needs compatible streams or --restart-rng")
                for worker_rng, state in zip(worker_rngs, states):
                    worker_rng.setstate(state)
            if device.type == "cuda" and saved.get("cuda_rng"):
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
        if sampler is not None and saved.get("curriculum") is not None:
            if saved.get("dataset_sha256") != dataset_digest:
                raise ValueError("Adaptive resume requires the same training map manifest")
            sampler.restore(saved["curriculum"])
        # Resume starts fresh episodes: engine state and recurrent state are not serialized.
    milestones = sorted(set(args.checkpoint_steps))
    if any(not steps < target <= args.steps or (target - steps) % args.envs for target in milestones):
        ap.error("Checkpoint milestones must be within the run and aligned to the environment batch")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    visible_index = args.visible_env if args.visualize else None
    playback = Playback(f"LIVE TRAINING / env {visible_index}", args.speed,
                        args.telemetry_every) if args.visualize else None
    worker_indices = [i for i in range(args.envs) if args.parallel and i != visible_index]
    episode_counts = [0] * args.envs

    def read_validation():
        if sampler is None or not args.curriculum_feedback:
            return
        path = Path(args.curriculum_feedback)
        if not path.is_file():
            return
        feedback = json.loads(path.read_text(encoding="utf-8"))
        if feedback.get("split") != "validation" or feedback.get("dataset_sha256") != dataset_digest:
            raise ValueError("Curriculum feedback must come from this dataset's validation partition")
        sampler.set_validation(feedback["curriculum_feedback"])

    def read_state(env, index):
        if shaped and (not args.parallel or index == visible_index):
            return skill_state(env)
        return env.privileged_state()

    def new_env(progress, index):
        stage_idx = min(4, int(progress * 5))
        allowed = STAGES[:stage_idx+1] if args.stage == "auto" else STAGES if args.stage == "mixed" else (args.stage,)
        map_rng = worker_rngs[index] if worker_rngs else rng
        if select_training_map is not None:
            spec = select_training_map(train_maps, map_rng, steps)
        else:
            spec = sampler.choose(map_rng) if sampler else map_rng.choice([m for m in train_maps if m["stage"] in allowed])
        ticks = spec["ticks"] if shaped else args.ticks
        map_name = spec.get("map_name", "MAP01")
        if index == visible_index:
            env = make_visible_env(spec["path"], map_name, episode_ticks=ticks, render_buffers=args.render_buffers,
                                   contract_pixels=args.render_buffers,
                                   render_all_frames=0 < args.speed <= 4)
        elif args.parallel:
            from training.workers import WorkerEnv
            env = WorkerEnv(spec["path"], ticks, map_name=map_name, training_signals=shaped,
                            render_buffers=args.render_buffers)
        else:
            env = make_env(spec["path"], map_name, episode_ticks=ticks, render_buffers=args.render_buffers)
        try:
            engine_seed = map_rng.randrange(2**30)
            obs, _ = env.reset(seed=engine_seed)
            if args.render_buffers:
                rgb = obs["rgb"]
                if rgb.shape != (120, 160, 3) or rgb.dtype != np.uint8 or not rgb.any():
                    raise ValueError("Training reset did not provide populated uint8 RGB contract frames")
            if world:
                metrics = UniverseReward(spec, args.gamma, potential_weight=args.potential_weight,
                                         novelty_weight=args.novelty_weight)
            elif course:
                metrics = reward_class(spec, guided=args.reward == "mechanics", gamma=args.gamma,
                                       trigger_discipline=args.trigger_discipline)
            elif skills:
                metrics = SkillReward(spec["drill"])
            elif args.reward == "competition":
                metrics = CompetitionReward()
            else:
                metrics = ProgressReward() if args.reward == "progress" else EpisodeMetrics()
            metrics.reset(read_state(env, index))
            metrics.stage = spec["stage"]
            metrics.spec = spec
            episode_counts[index] += 1
            if worker_rngs:
                assignment = dict(environment=int(index), episode=episode_counts[index], steps=steps,
                                  map=spec["id"], engine_seed=engine_seed)
                with (out / "map-assignments.jsonl").open("a", encoding="utf-8") as records:
                    records.write(json.dumps(assignment) + "\n")
            if index == visible_index:
                playback.start(f"Episode {episode_counts[index]:,} | {spec['path']} {map_name} | {steps:,} training decisions")
        except BaseException:
            env.close()
            raise
        return env, obs, metrics
    envs, observations, trackers = [], [], []
    started = time.perf_counter(); first_steps = steps
    def progress():
        return curriculum_progress(steps, args.steps, time.perf_counter()-started,
                                   args.seconds, args.curriculum_clock)
    stop_reason = None
    initial_updates = updates
    engine_errors = (*WINDOW_EXIT_ERRORS, vizdoom.ViZDoomErrorException,
                     vizdoom.MessageQueueException, vizdoom.SharedMemoryException, OSError, RuntimeError)

    def request_stop(signum, _frame):
        nonlocal stop_reason
        stop_reason = signal.Signals(signum).name

    def stop_requested():
        return stop_reason is not None

    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        read_validation()
        for i in range(args.envs):
            if stop_reason:
                break
            try:
                env, obs, tracker = new_env(progress(), i)
            except engine_errors:
                if stop_reason:
                    break
                raise
            envs.append(env); observations.append(obs); trackers.append(tracker)
        hidden = model.initial_state(args.envs, device)
        context = deque(maxlen=args.burn_in)
        previous_action = torch.zeros(args.envs, 9, device=device)
        starts = torch.ones(args.envs, device=device)
        with (out / "learning.jsonl").open("a", encoding="utf-8") as log:
            while not stop_reason and steps < args.steps and time.perf_counter() - started < args.seconds:
                if args.stop_file and args.stop_file.exists():
                    stop_reason = "stop-file"
                    break
                if args.min_available_mib and psutil.virtual_memory().available < args.min_available_mib * 2**20:
                    stop_reason = "memory-floor"
                    break
                rollout_start = steps
                boundary = next((target for target in milestones if target > steps), args.steps)
                data = {k: [] for k in ("hidden", "previous_action", "starts", "actions", "log_probs",
                        "values", "rewards", "next_values", "terminated", "truncated")}
                packed_seq, episodes = [], []
                for _ in range(min(args.rollout, max(1, (boundary-steps)//args.envs))):
                    if stop_reason or time.perf_counter() - started >= args.seconds:
                        break
                    packed = tensor_batch(observations, device, vision=model.vision_kind)
                    with torch.no_grad():
                        logits, values, next_hidden = model(packed, hidden, previous_action, starts)
                        groups, log_probs, _ = policy_stats(logits)
                        action = buttons(groups)
                    following, rewards, terminated, truncated = [], [], [], []
                    action_array = action.cpu().numpy()
                    try:
                        for i in worker_indices:
                            envs[i].send("step", action_array[i])
                        for i, env in enumerate(envs):
                            if i == visible_index:
                                playback.wait(stop_requested)
                            obs, _, term, trunc, _ = (
                                env.receive() if args.parallel and i != visible_index else env.step(action_array[i])
                            )
                            state = read_state(env, i)
                            if course:
                                reward = trackers[i].step(state, action_array[i], observations[i], obs)
                            elif world:
                                reward = trackers[i].step(state, action_array[i])
                            else:
                                reward = trackers[i].step(state)
                            if shaped and trackers[i].lesson_complete:
                                term, trunc = True, False
                                state = dict(state, finished=True, lesson_complete=True)
                            if i == visible_index:
                                playback.step(env, action_array[i], trackers[i], state, reward,
                                              update=updates, log_prob=log_probs[i], global_step=steps + i + 1)
                            following.append(obs)
                            rewards.append(reward)
                            terminated.append(term)
                            truncated.append(trunc)
                    except engine_errors as error:
                        if stop_reason:
                            break
                        if i == visible_index and isinstance(error, WINDOW_EXIT_ERRORS):
                            stop_reason = "window-exited"
                            print(f"[{playback.label}] Native game exited: {error}", flush=True)
                            break
                        raise
                    if not finite:
                        with torch.no_grad():
                            _, next_values, _ = model(tensor_batch(following, device, vision=model.vision_kind), next_hidden,
                                action.float(), torch.zeros(args.envs, device=device))
                        data["next_values"].append(next_values.detach())
                    data["hidden"].append(hidden.detach())
                    if args.burn_in:
                        context.append((packed, previous_action, starts, hidden))
                    data["starts"].append(starts.clone())
                    data["previous_action"].append(previous_action.clone())
                    packed_seq.append(packed)
                    for k, value in (("actions", groups), ("log_probs", log_probs), ("values", values),
                        ("rewards", torch.tensor(rewards, device=device, dtype=torch.float32)),
                        ("terminated", torch.tensor(terminated, device=device)),
                        ("truncated", torch.tensor(truncated, device=device))):
                        data[k].append(value.detach())
                    steps += args.envs
                    done = np.array(terminated) | np.array(truncated)
                    for i in np.flatnonzero(done):
                        tracker = trackers[i]
                        spec = tracker.spec
                        episode = dict(stage=tracker.stage, **tracker.summary())
                        if world or course:
                            episode.update(skill=spec["skill"], level=spec["level"], family=spec["family"], map=spec["path"])
                        episodes.append(episode)
                        if sampler:
                            sampler.observe(spec["skill"], spec["level"], episode)
                        if not stop_reason:
                            envs[i].close()
                            try:
                                envs[i], following[i], trackers[i] = new_env(progress(), i)
                            except engine_errors:
                                if not stop_reason:
                                    raise
                    observations = following
                    hidden = next_hidden.detach()
                    starts = torch.as_tensor(done, device=device, dtype=torch.float32)
                    previous_action = action.float()
                if not packed_seq:
                    break
                if finite:
                    # Values at the next rollout observation are already available in the following row.
                    # At a boundary the following value is masked, so resetting an environment is safe.
                    with torch.no_grad():
                            _, final_values, _ = model(tensor_batch(observations, device, vision=model.vision_kind),
                                                       hidden, previous_action, starts)
                    data["next_values"] = [*data["values"][1:], final_values]
                batch = {k: torch.stack(v) for k, v in data.items()}
                batch["obs"] = {k: torch.stack([p[k] for p in packed_seq]) for k in packed_seq[0]}
                learning_rate = args.lr * max(.1, 1 - steps / (args.lr_decay_steps or args.steps))
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                visual_start = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                                if name.startswith(("visual_encoder.", "visual_readout."))} if (
                    args.audit_rollouts and updates == initial_updates and model.vision_kind) else {}
                stats = update(model, optimizer, batch, epochs=args.epochs,
                               sequence_length=args.sequence, sequences_per_batch=args.sequences_per_batch,
                               entropy_coef=args.entropy, gamma=args.gamma, lam=args.gae_lambda,
                               bootstrap_truncations=not finite,
                               audit_replay=args.audit_rollouts and updates == initial_updates,
                               burn_in=args.burn_in)
                if visual_start:
                    deltas = {name: (parameter.detach() - visual_start[name]).abs().max().item()
                              for name, parameter in model.named_parameters() if name in visual_start}
                    if not any(delta > 0 for name, delta in deltas.items() if name.startswith("visual_encoder.")) or (
                            deltas["visual_readout.weight"] <= 0):
                        raise RuntimeError("The visual encoder and readout must both update during the first audited rollout")
                    stats["visual_parameter_deltas"] = deltas
                if context:
                    with torch.no_grad():
                        hidden = context[0][3]
                        context_obs = {key: torch.cat([packed[key] for packed, *_ in context]) for key in context[0][0]}
                        actions = torch.cat([previous for _, previous, _, _ in context])
                        context_starts = torch.stack([start for _, _, start, _ in context])
                        encoded = model.encode(context_obs, actions, context_starts.flatten()).reshape(
                            len(context), args.envs, -1)
                        for latent, start in zip(encoded, context_starts):
                            _, _, hidden = model.advance(latent, hidden, start)
                updates += 1
                elapsed = time.perf_counter()-started
                row = dict(steps=steps, updates=updates, seconds=elapsed, learning_rate=learning_rate,
                           rollout_decisions=steps-rollout_start,
                           decisions_per_second=(steps-first_steps)/elapsed, **stats, episodes=episodes)
                if sampler:
                    read_validation()
                    row["curriculum"] = sampler.snapshot()
                log.write(json.dumps(row)+"\n"); log.flush()
                metadata = dict(curriculum=sampler.state_dict() if sampler else None, dataset_sha256=dataset_digest,
                                worker_rngs=worker_rngs)
                save_checkpoint(out / "latest.pt", model, optimizer, steps, updates, args, rng, **metadata)
                if (steps in milestones or
                        args.checkpoint_every and steps // args.checkpoint_every > rollout_start // args.checkpoint_every):
                    save_checkpoint(out / f"step-{steps}.pt", model, optimizer, steps, updates, args, rng, **metadata)
                    if playback:
                        print(f"[{playback.label}] Checkpoint saved at {steps:,} decisions", flush=True)
                print(json.dumps({k:v for k,v in row.items() if k not in ("episodes", "curriculum")}), flush=True)
                if args.cooldown_ms and not stop_reason:
                    time.sleep(args.cooldown_ms / 1000)
    finally:
        try:
            for env in envs:
                env.close()
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
    if updates == initial_updates and stop_reason is None:
        raise RuntimeError("No training update completed; no trained artifact was produced")
    checkpoint = str(out / "latest.pt") if updates > initial_updates else args.resume
    summary = dict(steps=steps, updates=updates, elapsed_seconds=time.perf_counter()-started,
                   checkpoint=checkpoint, parameters=sum(p.numel() for p in model.parameters()),
                   device=str(device), resumed_from=args.resume, invocation_steps=steps-first_steps,
                   invocation_updates=updates-initial_updates,
                   stop_reason=stop_reason or ("target" if steps >= args.steps else "wall-clock"),
                   mode="live" if args.visualize else "fast",
                   reward=args.reward,
                   model_config=model.config, training_map_count=len(train_maps), dataset_sha256=dataset_digest,
                   curriculum=sampler.snapshot() if sampler else None,
                   cooldown_ms=args.cooldown_ms, curriculum_clock=args.curriculum_clock,
                   torch_version=torch.__version__,
                   peak_torch_vram_reserved_bytes=torch.cuda.max_memory_reserved() if device.type=="cuda" else 0)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__": main()
