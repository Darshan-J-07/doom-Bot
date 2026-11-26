"""Paired standard-engine evaluations through exported DoomAgent entry points."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from hashlib import sha256
import gzip
import json
from math import isfinite
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np
from doom_env import limits, make_env, remote
from doom_env.spaces import BUTTONS

from training.measure import GameplayObserver, door_extents, engine_stats
from training.visualization import Playback, make_visible_env


ROOT = Path(__file__).resolve().parents[1]
TIER = {"combat": 1, "exploration": 2, "keys": 3, "gauntlet": 4}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")


@contextmanager
def exported_agent(repo, log_path, device):
    with TemporaryDirectory(prefix="doom-campaign-") as work:
        server = remote.listen(Path(work) / "agent.sock")
        command = [sys.executable, "-m", "doom_env.remote", "serve", "--socket", remote.address(server),
                   "--repo", str(repo), "--device", device]
        env = {**os.environ, **limits.THREAD_ENV, "PYTHONPATH": str(ROOT / "env"),
               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
        affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        started = time.perf_counter()
        with log_path.open("ab") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=work, env=env,
                                       preexec_fn=(lambda: os.sched_setaffinity(0, {affinity[-1]})) if affinity else None)
            conn = None
            try:
                conn = remote.AgentConnection.accept(server, limits.INIT_TIMEOUT_S, alive=lambda: process.poll() is None)
                remaining = max(.1, limits.INIT_TIMEOUT_S - (time.perf_counter() - started))
                conn.init(device, remaining)
                yield conn, time.perf_counter() - started
            finally:
                if conn:
                    conn.close()
                server.close()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def flatten_obs(obs):
    return {key.replace("/", "__"): value for key, value in remote._flatten(obs)}


def play(row, seed, conn, *, playback=None, trace_path=None, legal=None):
    factory = make_visible_env if playback else make_env
    render_options = {"render_all_frames": 0 < playback.speed <= 4} if playback else {}
    ticks = row.get("ticks", 2100)
    initial = time.perf_counter()
    errors, latencies, actions = {}, [], sha256()
    attempts, play_tics, acted_seconds = 0, 0, 0.
    with factory(row["path"], row.get("map_name", "MAP01"), episode_ticks=ticks, **render_options) as env:
        options = {"record_path": str(trace_path.with_suffix(".lmp"))} if trace_path else None
        obs, _ = env.reset(seed=seed, options=options)
        observer = GameplayObserver(engine_stats(env), door_polygons=door_extents(env), spec=row)
        reset_start = time.perf_counter()
        try:
            conn.reset(limits.RESET_TIMEOUT_S)
        except remote.AgentFailure as error:
            errors = dict(kind=error.kind, where=error.where, message=error.message)
        reset_seconds = time.perf_counter() - reset_start
        if playback:
            playback.start(f"{row.get('id', row['path'])} | seed {seed}")
        term = trunc = False
        budget = limits.agent_budget_s(ticks)
        idle = np.zeros(len(BUTTONS), np.int8)
        frame_writer = gzip.open(trace_path, "wt", encoding="utf-8") if trace_path else None
        try:
            while not (term or trunc):
                action = idle
                if not errors:
                    if legal is not None and len(legal) < 192 and (observer.decisions % 11 == 0):
                        assert env.observation_space.contains(obs)
                        legal.append(flatten_obs(obs))
                    before = time.perf_counter()
                    try:
                        remaining = budget - acted_seconds
                        if remaining <= 0:
                            raise remote.AgentFailure("timeout", "agent used its episode compute budget", "episode")
                        deadline = limits.ACT_TIMEOUT_S if attempts else limits.FIRST_ACT_TIMEOUT_S
                        attempts += 1
                        action = conn.act(obs, min(remaining, deadline))
                        latencies.append((time.perf_counter() - before) * 1000)
                    except remote.AgentFailure as error:
                        errors = dict(kind=error.kind, where=error.where, message=error.message)
                    acted_seconds += time.perf_counter() - before
                if playback:
                    playback.wait()
                old = obs
                obs, _, term, trunc, _ = env.step(action)
                state = engine_stats(env)
                frame = observer.record(old, action, state)
                actions.update(action.tobytes())
                if frame_writer:
                    frame_writer.write(json.dumps(frame, allow_nan=False, separators=(",", ":")) + "\n")
                if playback:
                    playback.step(env, action, observer, state, frame["reward"])
            play_tics = observer.clock.elapsed
        finally:
            if frame_writer:
                frame_writer.close()
        result = observer.report()
    return dict(result, tick_cap=ticks, outcome="exit" if result["exit"] else "death" if result["dead"] else "tick_cap",
                agent_error=errors or None, engine_error=None, invalid_actions=int(errors.get("kind") == "invalid_action"),
                inference_timeouts=int(errors.get("kind") == "timeout"),
                crashes=int(bool(errors) and errors["kind"] in ("crash", "protocol", "error")),
                inference_ms=float(np.mean(latencies)) if latencies else None,
                p99_ms=float(np.percentile(latencies, 99)) if latencies else None,
                first_ms=latencies[0] if latencies else None, measured_agent_seconds=acted_seconds,
                reset_seconds=reset_seconds, duration_seconds=play_tics / 35, action_sha256=actions.hexdigest(),
                wall_seconds=time.perf_counter() - initial)


def read_rows(path, split):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [row for row in payload["maps"] if row["split"] == split
            and row.get("stage") in TIER]
    if not rows:
        raise ValueError(f"Empty or unsupported split {split}")
    cache = {}
    for row in rows:
        path = Path(row["path"]).resolve()
        if path not in cache:
            cache[path] = digest(path)
        if cache[path] != row["sha256"]:
            raise ValueError(f"WAD hash changed: {path}")
    return rows


def evaluate(rows, repo, seeds, output, *, device="cpu", playback=None, traces=0, save_observations=False):
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(repo).resolve()
    legal = [] if save_observations else None
    checksum = digest(repo / "submission/weights/best_model.pt")
    episodes = []
    with exported_agent(repo, output / "agent.log", device) as (conn, init_seconds):
        for index, row in enumerate(rows):
            for seed in seeds:
                trace = output / f"map-{index:02}-seed-{seed}.jsonl.gz" if index < traces and seed == seeds[0] else None
                try:
                    result = play(row, seed, conn, playback=playback, trace_path=trace, legal=legal)
                except Exception as error:
                    result = dict(engine_error=f"{type(error).__name__}: {error}", outcome="engine_error", exit=False)
                result.update(map=row.get("id", row["path"]), path=row["path"], sector_map=row.get("map_name", "MAP01"),
                              wad_sha256=row["sha256"], stage=row["stage"], tier=TIER[row["stage"]],
                              family=row.get("family", row.get("layout", row["stage"])), engine_seed=seed,
                              policy_seed=0, trace=str(trace) if trace else None)
                append_jsonl(output / "episodes.jsonl", result)
                episodes.append(result)
                print(f"{result['map']} seed {seed} | {result['outcome']} | kills {result.get('kills', 0)} "
                      f"| damage {result.get('damage', 0):g} | keys {result.get('keys', 0)} "
                      f"| locks {result.get('locks', 0)} | tics {result.get('ticks', 0)}", flush=True)
    if legal:
        np.savez_compressed(output / "legal-observations.npz", **{key: np.stack([obs[key] for obs in legal]) for key in legal[0]})
    info = dict(checkpoint_sha256=checksum, repo=str(repo), device=device, init_seconds=init_seconds,
                map_count=len(rows), engine_seeds=seeds, native=playback is not None, episodes=len(episodes),
                exits=sum(row["exit"] for row in episodes), errors=sum(bool(row.get("engine_error") or row.get("agent_error"))
                                                                     for row in episodes))
    (output / "summary.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    if checksum != digest(repo / "submission/weights/best_model.pt"):
        raise RuntimeError("Evaluation checkpoint changed")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test", "development", "probe"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--traces", type=int, default=0)
    parser.add_argument("--save-observations", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not isfinite(args.speed) or args.speed < 0:
        parser.error("Use a finite nonnegative speed. Zero is uncapped.")
    playback = Playback("FROZEN EXPORTED POLICY", args.speed, 128) if args.visualize else None
    info = evaluate(read_rows(args.manifest, args.split), args.repo, args.seeds, args.out,
                    device=args.device, playback=playback, traces=args.traces, save_observations=args.save_observations)
    print(json.dumps(info), flush=True)
    return int(bool(info["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
