"""Record live RGB and check local and transport parity for an exported DoomAgent."""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from doom_env import action_space, limits, make_env, observation_space, remote
from doom_env.spaces import BUTTONS, FRAME_SKIP
import numpy as np
import torch


def collect_frames(wad, *, map_name="MAP01", seed=0, count=24, render_buffers=True, choose_action=None):
    if not render_buffers:
        raise ValueError("RGB parity requires render_buffers=True")
    if count < 2:
        raise ValueError("At least two frames are needed to measure scene changes")
    frames, ticks, aligned, digests, actions = [], [], [], [], []
    turn = np.zeros(len(BUTTONS), np.int8)
    turn[BUTTONS.index("TURN_LEFT")] = 1
    with make_env(wad, map_name, episode_ticks=max(200, (count + 1) * FRAME_SKIP), render_buffers=render_buffers) as env:
        obs, info = env.reset(seed=seed)
        for index in range(count):
            if index:
                obs, _, terminated, truncated, info = env.step(actions[-1])
                if terminated or truncated:
                    raise RuntimeError("The episode ended before collecting all audit frames")
            if not env.observation_space.contains(obs):
                raise ValueError(f"Illegal official observation at frame {index}")
            rgb = obs["rgb"]
            if not np.any(rgb):
                raise ValueError(f"Empty RGB frame {index}. Confirm engine render buffers are enabled.")
            live = env._game.get_state().screen_buffer
            aligned.append(np.array_equal(rgb, live))
            ticks.append(info["tic"])
            frames.append({name: array.copy() for name, array in remote._flatten(obs)})
            digests.append(sha256(rgb.tobytes()).hexdigest())
            action = turn if choose_action is None else choose_action(obs)
            if not env.action_space.contains(action):
                raise ValueError(f"Illegal audit action at frame {index}")
            actions.append(action.copy())
    if not all(aligned) or not np.all(np.diff(ticks) == FRAME_SKIP) or len(set(digests)) < 2:
        raise AssertionError("Camera frames are stale, misaligned with the live engine, or have the wrong tick cadence")
    return frames, dict(wad=str(wad), map_name=map_name, seed=seed, count=count, nonzero=count,
                        unique=len(set(digests)), tics=ticks, frame_sha256=digests, shape=list(rgb.shape),
                        dtype=str(rgb.dtype), matches_live_rgb24=all(aligned),
                        actions=[action.tolist() for action in actions],
                        acted_engine_transitions=count - 1,
                        action_source="diagnostic rotation" if choose_action is None else "learner")


def frames_from_file(path):
    with np.load(path, allow_pickle=False) as data:
        channels = {key: data[key] for key in data.files}
    for i in range(len(channels["rgb"])):
        yield remote._unflatten((name, array[i]) for name, array in channels.items())


def record_paths(model, act, frames):
    recorded = defaultdict(list)

    def record(model, inputs, outputs):
        for name, value in inputs[0].items():
            recorded["packed/" + name].append(value.detach().cpu().numpy().copy())
        for name, value in zip(("logits", "values", "hidden"), outputs, strict=True):
            recorded[name].append(value.detach().cpu().numpy().copy())

    with model.register_forward_hook(record), torch.inference_mode():
        for frame in frames:
            recorded["action"].append(act(frame).copy())
    return {name: np.stack(values) for name, values in recorded.items()}


def capture_release(repo, input_path, output):
    sys.path.insert(0, str(repo))  # Isolated process, matching the evaluator's import precedence.
    from submission import agent as release

    if not Path(release.__file__).resolve().is_relative_to(repo):
        raise RuntimeError("Parity process imported a different submission")
    agent = release.DoomAgent(observation_space(), action_space(), "cpu")
    np.savez_compressed(output, **record_paths(agent.model, agent.act, frames_from_file(input_path)))


def audit(args):
    from scripts.campaign_eval import exported_agent
    from submission.model import SpatialPolicy, action_groups, buttons
    from submission.preprocessing import tensor_batch

    torch.set_num_threads(1)
    repo = args.repo.resolve()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    serving = torch.load(repo / "submission/weights/best_model.pt", map_location="cpu", weights_only=True)
    assert checkpoint["model_config"] == serving["model_config"]
    assert checkpoint["model"].keys() == serving["model"].keys()
    assert all(torch.equal(value, serving["model"][key]) for key, value in checkpoint["model"].items())
    policy = SpatialPolicy(**checkpoint["model_config"]).eval()
    policy.load_state_dict(checkpoint["model"], strict=True)
    def learner_agent():
        hidden = policy.initial_state()
        previous = torch.zeros(1, len(BUTTONS))
        starts = torch.ones(1)
        generator = torch.Generator().manual_seed(0)

        @torch.inference_mode()
        def act(frame):
            nonlocal hidden, previous
            packed = tensor_batch([frame], vision=policy.vision_kind)
            logits, _, hidden = policy(packed, hidden, previous, starts)
            selection = action_groups(logits, deterministic=serving.get("deterministic", False), generator=generator)
            action = buttons(selection)
            previous = action.float()
            starts.zero_()
            return action[0].numpy()

        return act

    frames, report = collect_frames(args.wad, map_name=args.map_name, seed=args.seed, count=args.frames,
                                    choose_action=learner_agent())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame_path = args.out.with_suffix(".frames.npz").resolve()
    np.savez_compressed(frame_path, **{key: np.stack([row[key] for row in frames]) for key in frames[0]})
    learner = record_paths(policy, learner_agent(), frames_from_file(frame_path))
    np.testing.assert_array_equal(learner["action"], report["actions"], err_msg="Replayed actions differ from live actions")
    with TemporaryDirectory(prefix="doom-vision-parity-") as temporary:
        release_path = Path(temporary) / "release.npz"
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--capture-release", str(repo),
                        str(frame_path), str(release_path)], cwd=temporary, check=True)
        with np.load(release_path, allow_pickle=False) as release:
            assert set(release.files) == learner.keys()
            for key, value in learner.items():
                np.testing.assert_array_equal(value, release[key], err_msg=f"Learner/release mismatch at {key}")

    with exported_agent(repo, args.out.with_suffix(".agent.log"), "cpu") as (server, _):
        server.reset(limits.RESET_TIMEOUT_S)
        for frame, expected in zip(frames_from_file(frame_path), learner["action"], strict=True):
            np.testing.assert_array_equal(server.act(frame, limits.ACT_TIMEOUT_S), expected)
    visual = learner.get("packed/rgb")
    swapped_max = None
    if policy.vision_kind == "rgb":
        if not np.any(visual[1:] != visual[:-1]):
            raise AssertionError("Packed RGB failed to preserve any scene changes")
        # Hold state and all structured observations fixed. This is a sensitivity
        # check for the trained network, separate from performance comparisons.
        packed = tensor_batch(list(frames_from_file(frame_path)), vision="rgb")
        state = policy.initial_state(len(frames))
        previous = torch.zeros(len(frames), len(BUTTONS))
        starts = torch.ones(len(frames))
        with torch.inference_mode():
            normal, _, _ = policy(packed, state, previous, starts)
            swapped, _, _ = policy(dict(packed, rgb=packed["rgb"].roll(1, 0)), state, previous, starts)
            swapped_max = (normal - swapped).abs().max().item()
    report.update(checkpoint=str(args.checkpoint), model_config=policy.config,
                  checkpoint_sha256=sha256(args.checkpoint.read_bytes()).hexdigest(),
                  export_sha256=sha256((repo / "submission/weights/best_model.pt").read_bytes()).hexdigest(),
                  packed_rgb_shape=None if visual is None else list(visual.shape),
                  packed_rgb_range=None if visual is None else [int(visual.min()), int(visual.max())],
                  compared_forward_tensors=sorted(learner), exact_local_outputs=True,
                  exact_remote_actions=len(frames), swapped_image_logit_max_abs_delta=swapped_max,
                  frame_artifact=str(frame_path))
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Compared {len(frames)} live engine frames and exact learner/release logits for {policy.vision_kind}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--capture-release":
        capture_release(*map(Path, sys.argv[2:]))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--repo", type=Path, required=True)
        parser.add_argument("--wad", type=Path, required=True)
        parser.add_argument("--map-name", default="MAP01")
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--frames", type=int, default=24)
        parser.add_argument("--out", type=Path, required=True)
        args = parser.parse_args()
        if args.frames < 2:
            parser.error("at least two frames are needed to measure scene changes")
        audit(args)
