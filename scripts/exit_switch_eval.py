"""Evaluate exit-switch drills through the official isolated DoomAgent entry point."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import gzip
from hashlib import sha256
import json
from math import cos, floor, radians, sin
from pathlib import Path
import socket
import struct
import time

import numpy as np
from doom_env import limits, make_env, remote
from doom_env.sensors import _ray_line_t, _segments_blocked
from doom_env.spaces import BUTTONS

from scripts.campaign_eval import append_jsonl, digest, exported_agent, flatten_obs
from training.measure import GameplayObserver, door_extents, engine_stats
from training.visualization import Playback, add_playback_arguments, make_visible_env, validate_playback_arguments
from validate import VramSampler


USE = BUTTONS.index("USE")
USE_RANGE = 64.0
VISIBILITY_COSINE = cos(radians(45.0))
STAGES = ("area_discovered", "switch_visible", "switch_near", "switch_faced_in_range",
          "switch_attempt", "switch_activated", "exit")


def segments(values):
    lines = np.asarray(values, dtype=np.float64).reshape(-1, 2, 2)
    if not np.isfinite(lines).all() or np.any(np.all(lines[:, 0] == lines[:, 1], axis=1)):
        raise ValueError("Invalid diagnostic linedef")
    return lines


class SwitchObserver:
    """Geometry proxies sampled before each action, separate from actual engine exits."""

    def __init__(self, row, geometry):
        self.geometry = geometry
        self.exit_cell = tuple(row["exit_cell"])
        self.tile = row["tile"]
        self.target = segments([row["exit_segment"]]) if row["exit_type"] == "switch" else None
        self.distractors = segments(row.get("distractor_segments", []))
        self.samples = None
        if self.target is not None:
            start, end = self.target[0]
            self.samples = start + np.array([[.15], [.50], [.85]]) * (end - start)
        self.first = dict.fromkeys(STAGES)
        self.counts = Counter()
        self.minimum_distance = None
        self.prior_use = self.prior_facing = False

    def inspect(self, state):
        origin = np.array([state["x"], state["y"]])
        position = tuple(floor(v / width) for v, width in zip(origin, self.tile))
        in_area = position == self.exit_cell
        if self.target is None:
            return dict(area=in_area, visible=False, distance=None, near=False, facing=False, target_blocked=False,
                        distractor_facing=False)

        start, end = self.target[0]
        line = end - start
        offset = origin - start
        distance = float(np.linalg.norm(offset - line * np.clip(offset @ line / (line @ line), 0, 1)))
        self.minimum_distance = distance if self.minimum_distance is None else min(self.minimum_distance, distance)
        on_front = bool(line[0] * offset[1] - line[1] * offset[0] < 0)
        heading = radians(state["angle"])
        look = np.array([cos(heading), sin(heading)])
        sight = self.geometry.sight_block
        fov = (self.samples - origin) @ look >= VISIBILITY_COSINE * np.linalg.norm(self.samples - origin, axis=1)
        visible = bool(on_front and np.any(fov & ~_segments_blocked(
            origin, self.samples, self.geometry.p1[sight], self.geometry.p2[sight])))

        target_t = float(_ray_line_t(origin, look[None], self.target[:, 0], self.target[:, 1])[0, 0])
        distractor_t = float(_ray_line_t(origin, look[None], self.distractors[:, 0], self.distractors[:, 1]).min(initial=np.inf))
        solid = self.geometry.move_block
        obstacle_t = float(_ray_line_t(origin, look[None], self.geometry.p1[solid], self.geometry.p2[solid]).min(initial=np.inf))
        target_blocked = target_t <= USE_RANGE and (obstacle_t < target_t - .01 or distractor_t < target_t - .01)
        facing = on_front and target_t <= USE_RANGE and not target_blocked
        distractor = distractor_t <= USE_RANGE and distractor_t <= obstacle_t + .01 and distractor_t < target_t
        return dict(area=in_area, visible=visible, distance=distance, near=on_front and distance <= USE_RANGE,
                    facing=bool(facing), target_blocked=bool(target_blocked), distractor_facing=bool(distractor))

    def mark(self, name, event, tic):
        if event and self.first[name] is None:
            self.first[name] = tic

    def action(self, sample, action, tic):
        use = bool(action[USE])
        edge = use and not self.prior_use
        self.mark("area_discovered", sample["area"] or sample["visible"], tic)
        self.mark("switch_visible", sample["visible"], tic)
        self.mark("switch_near", sample["near"], tic)
        self.mark("switch_faced_in_range", sample["facing"], tic)
        self.mark("switch_attempt", edge and sample["facing"], tic)
        self.counts["approaches"] += sample["facing"] and not self.prior_facing
        self.counts["usable_decisions"] += sample["facing"]
        self.counts["activation_attempts"] += edge and sample["facing"]
        self.counts["distractor_attempts"] += edge and sample["distractor_facing"]
        self.counts["blocked_attempts"] += edge and sample["target_blocked"]
        self.counts["held_use_at_opportunity"] += use and self.prior_use and sample["facing"]
        self.counts["out_of_alignment_attempts"] += edge and sample["near"] and not sample["facing"]
        self.prior_use, self.prior_facing = use, sample["facing"]
        return edge

    def result(self, completed, tic):
        activated = completed and self.target is not None
        self.mark("switch_activated", activated, tic)
        self.mark("exit", completed, tic)
        unreached = next((key for key, value in self.first.items() if value is None), None)
        if self.target is None and not completed:
            unreached = "walkover"
        return dict(self.counts, stages=self.first, first_unreached_stage=None if completed else unreached,
                    activation=activated, activation_without_usable_edge=activated and self.first["switch_attempt"] is None,
                    minimum_distance=self.minimum_distance)


def load_rows(manifest, split, tasks=(), ids=()):
    source = json.loads(manifest.read_text(encoding="utf-8"))
    rows = [row for row in source["maps"] if row["split"] == split and (not tasks or row["task"] in tasks)
            and (not ids or row["id"] in ids)]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Expected a nonempty, unique cohort in split {split}")
    if set(ids) - {row["id"] for row in rows}:
        raise ValueError("Requested map missing from the selected split or task")
    artifacts = {}
    for row in rows:
        path = Path(row["path"]).resolve()
        if path not in artifacts:
            artifacts[path] = digest(path)
        if artifacts[path] != row["sha256"]:
            raise ValueError(f"WAD hash changed: {path}")
        if row["exit_type"] not in {"walkover", "switch"}:
            raise ValueError(f"Unknown exit type in {path}")
    return rows


def play(row, seed, agent, *, playback=None, trace=None, frames=None, frame_limit=0):
    factory = make_visible_env if playback else make_env
    render_options = {"render_all_frames": 0 < playback.speed <= 4, "contract_pixels": True} if playback else {}
    trace_context = gzip.open(trace, "wt", encoding="utf-8") if trace else nullcontext(None)
    latencies, acted_seconds, error = [], 0., None
    initial = time.perf_counter()
    with factory(row["path"], row["map_name"], episode_ticks=row["ticks"], render_buffers=True, **render_options) as env, trace_context as sink:
        lmp = trace.with_suffix(".lmp") if trace else None
        obs, _ = env.reset(seed=seed, options={"record_path": str(lmp)} if lmp else None)
        state = engine_stats(env)
        observer = GameplayObserver(state, door_polygons=door_extents(env), spec=row)
        switch = SwitchObserver(row, env._geometry)
        initial_rgb = obs["rgb"].copy()
        if not env.observation_space.contains(obs) or not np.any(initial_rgb):
            raise ValueError("Initial observation is illegal or RGB is blank")
        rgb_changed = False
        budget = limits.agent_budget_s(row["ticks"])
        if playback:
            playback.start(f"{row['id']} | seed {seed}")
        try:
            agent.reset(limits.RESET_TIMEOUT_S)
            while not state["finished"]:
                if frames is not None and len(frames) < frame_limit and observer.decisions % 7 == 0:
                    frames.append({key: value.copy() for key, value in flatten_obs(obs).items()})
                stage = switch.inspect(state)
                now = time.perf_counter()
                deadline = limits.ACT_TIMEOUT_S if latencies else limits.FIRST_ACT_TIMEOUT_S
                remaining = min(budget - acted_seconds, deadline)
                if remaining <= 0:
                    raise remote.AgentFailure("timeout", "episode compute budget exhausted", "act")
                action = agent.act(obs, remaining)
                latency = time.perf_counter() - now
                latencies.append(latency * 1000)
                acted_seconds += latency
                prior_tic = observer.clock.elapsed
                rising = switch.action(stage, action, prior_tic)
                if playback:
                    playback.wait()
                old = obs
                obs, _, term, trunc, info = env.step(action)
                state = engine_stats(env)
                if (term or trunc) != state["finished"] or info["exit_reached"] != state["exit_reached"]:
                    raise RuntimeError("Wrapper termination disagrees with authoritative engine state")
                rgb_changed = rgb_changed or not np.array_equal(initial_rgb, obs["rgb"])
                frame = observer.record(old, action, state)
                if sink:
                    frame.update(switch_before_action=stage, decision_tic_before=prior_tic, use_rising=rising,
                                 rgb_sha256_before=sha256(old["rgb"].tobytes()).hexdigest(),
                                 true_exit=state["exit_reached"])
                    sink.write(json.dumps(frame, allow_nan=False, separators=(",", ":")) + "\n")
                if playback:
                    playback.step(env, action, observer, state, frame["reward"])
        except remote.AgentFailure as exc:
            error = dict(kind=exc.kind, where=exc.where, message=exc.message)

        episode = observer.report()
        episode.update(switch=switch.result(episode["exit"], episode["ticks"]), rgb_changed=rgb_changed,
                       outcome="agent_error" if error else "exit" if episode["exit"] else "death" if episode["dead"] else "tick_cap",
                       agent_error=error, engine_error=None, invalid_actions=int(bool(error) and error["kind"] == "invalid_action"),
                       latency_ms=float(np.mean(latencies)) if latencies else None,
                       latency_p95_ms=float(np.percentile(latencies, 95)) if latencies else None,
                       latency_max_ms=max(latencies, default=None), latency_decisions=len(latencies),
                       agent_seconds=acted_seconds, wall_seconds=time.perf_counter() - initial)
    return episode


def agent_pid(connection):
    if not hasattr(socket, "SO_PEERCRED"):
        return None
    try:
        return struct.unpack("3i", connection.sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))[0]
    except OSError:
        return None


def high_water_rss(pid):
    try:
        for field in Path(f"/proc/{pid}/status").read_text().splitlines():
            if field.startswith("VmHWM:"):
                return int(field.split()[1]) / 1024
    except OSError:
        return None
    return None


def summarize(rows):
    metrics = Counter(episodes=len(rows))
    for row in rows:
        stage = row.get("switch", {})
        metrics["exits"] += bool(row["exit"])
        metrics["switch_activations"] += bool(stage.get("activation"))
        metrics["errors"] += bool(row.get("engine_error") or row.get("agent_error"))
        metrics["deaths"] += bool(row.get("dead"))
        metrics["tick_caps"] += row["outcome"] == "tick_cap"
        metrics["invalid_actions"] += row.get("invalid_actions", 0)
        for key in ("approaches", "activation_attempts", "distractor_attempts", "held_use_at_opportunity"):
            metrics[key] += stage.get(key, 0)
        for key in STAGES:
            metrics[f"episodes_{key}"] += stage.get("stages", {}).get(key) is not None
        metrics["keys"] += row.get("keys", 0)
        metrics["locks"] += row.get("locks", 0)
        metrics["kill_count"] += row.get("kills", 0)
        metrics["ammo_spent_lower_bound"] += row.get("work", {}).get("rounds_spent_lower_bound", 0)
        metrics["attack_without_enemy"] += row.get("work", {}).get("attack_without_visible_enemy", 0)
        metrics["completed_key_lock_exit"] += bool(row["exit"] and row.get("keys") and row.get("locks"))
    valid = [row for row in rows if not (row.get("engine_error") or row.get("agent_error"))]
    waited = sum(row["latency_decisions"] for row in valid)
    return dict(metrics, maps=len({row["wad_sha256"] for row in rows}),
                families=len({row["family"] for row in rows}),
                mean_exit_seconds=mean(row["ticks"] / 35 for row in valid if row["exit"]),
                mean_damage=mean(row["damage"] for row in valid),
                mean_roundtrip_latency_ms=(1000 * sum(row["agent_seconds"] for row in valid) / waited if waited else None),
                worst_episode_latency_p95_ms=max((row["latency_p95_ms"] for row in valid if row["latency_p95_ms"] is not None), default=None))


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def cohort(row):
    return f"{row['task']}_{row['exit_type']}"


def grouped(rows, by):
    groups = {}
    for row in rows:
        groups.setdefault(by(row), []).append(row)
    return {key: summarize(group) for key, group in sorted(groups.items())}


def evaluate(rows, repo, seeds, output, *, device="cpu", playback=None, traces=True, trace_every_seed=False,
             observation_count=0, manifest_sha256=None):
    output.mkdir(parents=True, exist_ok=False)
    repo = repo.resolve()
    weights = repo / "submission/weights/best_model.pt"
    checksum = digest(weights)
    episodes, frames = [], []
    gpu_baseline = VramSampler._gpu_used() if device == "cuda" else None
    with exported_agent(repo, output / "agent.log", device) as (agent, init_seconds):
        pid = agent_pid(agent)
        gpu = VramSampler(pid) if device == "cuda" and pid else None
        if gpu is not None:
            gpu.baseline = gpu_baseline
            gpu.start()
        try:
            for index, row in enumerate(rows):
                for seed in seeds:
                    trace = output / f"map-{index:03d}-seed-{seed}.jsonl.gz" if traces and (trace_every_seed or seed == seeds[0]) else None
                    result = play(row, seed, agent, playback=playback, trace=trace, frames=frames, frame_limit=observation_count)
                    result.update(map=row["id"], path=row["path"], sector_map=row["map_name"], wad_sha256=row["sha256"],
                                  family=row["family"], task=row["task"], exit_type=row["exit_type"], engine_seed=seed,
                                  policy_seed=0, trace=str(trace) if trace else None)
                    episodes.append(result)
                    append_jsonl(output / "episodes.jsonl", result)
                    print(f"{row['id']} seed {seed} | {result['outcome']} | attempt {result['switch']['activation_attempts']} "
                          f"| keys {result['keys']} | locks {result['locks']} | {result['ticks']} tics", flush=True)
        finally:
            if gpu is not None:
                gpu.stop()
                gpu.join(timeout=6)
        peak_rss = high_water_rss(pid)
    if digest(weights) != checksum:
        raise RuntimeError("Evaluation changed the checkpoint")
    if frames:
        np.savez_compressed(output / "legal-observations.npz", **{key: np.stack([frame[key] for frame in frames]) for key in frames[0]})
    result = dict(summarize(episodes), cohorts=grouped(episodes, cohort), per_family=grouped(episodes, lambda row: row["family"]),
                  checkpoint_sha256=checksum, manifest_sha256=manifest_sha256, repo=str(repo), device=device, seeds=seeds,
                  init_seconds=init_seconds, peak_agent_rss_mib=peak_rss, peak_agent_vram_mib=(gpu.peak or None) if gpu else None,
                  peak_device_vram_delta_mib=(gpu.peak_total or None) if gpu else None, native=playback is not None,
                  saved_observations=len(frames), rgb_changed_episodes=sum(row["rgb_changed"] for row in episodes))
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--task", nargs="+", default=[])
    parser.add_argument("--map", action="append", default=[])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--save-observations", type=int, default=0, metavar="COUNT")
    parser.add_argument("--no-traces", action="store_true")
    parser.add_argument("--trace-every-seed", action="store_true")
    add_playback_arguments(parser)
    args = parser.parse_args()
    validate_playback_arguments(parser, args)
    if args.save_observations < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use a nonnegative observation count and distinct seeds")
    playback = Playback("FROZEN EXIT STUDY", args.speed, args.telemetry_every) if args.visualize else None
    result = evaluate(load_rows(args.manifest, args.split, args.task, args.map), args.repo, args.seeds, args.out,
                      device=args.device, playback=playback, traces=not args.no_traces, trace_every_seed=args.trace_every_seed,
                      observation_count=args.save_observations, manifest_sha256=digest(args.manifest))
    print(json.dumps({key: result[key] for key in ("episodes", "exits", "switch_activations", "errors")}), flush=True)
    return int(bool(result["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
