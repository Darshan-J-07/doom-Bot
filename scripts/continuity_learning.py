"""Metered PPO learner profiles for the independent continuity campaign."""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from contextlib import ExitStack
from functools import partial
import math
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from doom_env import CONTRACT_VERSION
from doom_env.spaces import RAY_HIT_TYPES
from scripts.adaptive_followup import prove_initial_parity
from scripts.adaptive_imitation import PARENT, PARENT_SHA256, save
from scripts.adaptive_mixed import LANES, MixedSchedule, audit_partition, checked, classify, load_reward, pin, read, write
from scripts.export_candidate import export
from submission.model import GROUP_SIZES, SpatialPolicy
from training import train
from training.anchored_ppo import update_anchored
from training.learning_meter import METADATA_KEY, run
from training.maps import load_verified_maps


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "runs/adaptive-campaign-20261127"
PRIOR = read(ARCHIVE / "mixed-plan.json")
BC_EXPORT = ARCHIVE / "exports/bc-ppo-59-12000/export.json"
BC_START = Path(read(BC_EXPORT)["source_checkpoint"])
BC_SHA = read(BC_EXPORT)["source_checkpoint_sha256"]
DATASET = ARCHIVE / "imitation/data/trajectories.pt"
PROFILES = {
    "mixed": {"hypothesis": "Full-rate matched mixed control"},
    "retention-low": {"hypothesis": "Half original learning rate on persistent mixed practice", "rate": .5},
    "retention-anchor": {"hypothesis": "Moderate warm-policy distillation on fresh mixed rollouts", "anchor": .06},
    "retention-safe": {"hypothesis": "Lower learning-rate fallback using the same moderate anchor",
                       "rate": .25, "anchor": .06},
    "retention-distill": {"hypothesis": "Transfer only the lock-opening head from the BC-PPO teacher, retaining the warm policy elsewhere",
                          "anchor": .06, "lock_teacher": True},
    "memory-control": {"hypothesis": "Matched feedforward control for fresh-Adam memory forks", "fresh_adam": True},
    "gru": {"hypothesis": "Residual GRU with parity-preserving initialization", "memory": "residual-gru"},
    "occupancy": {"hypothesis": "Ego-centric occupancy memory with parity-preserving initialization", "memory": "occupancy"},
    "count": {"hypothesis": "Bounded coverage-based novelty on course rewards", "count_bonus": True},
    "rnd": {"hypothesis": "Bounded random-network-distillation novelty", "rnd_bonus": True},
    "mastery": {"hypothesis": "Preserve all skills while adapting map families in a mixed course pool",
                "adaptive_families": True, "rate": .55},
}


def checkpoint(path):
    path = path.resolve(strict=True)
    if path in (PARENT, BC_START):
        expected = PARENT_SHA256 if path == PARENT else BC_SHA
        if pin(path)["sha256"] != expected:
            raise ValueError(f"Frozen source identity changed: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("contract_version") != CONTRACT_VERSION or not saved.get("optimizer"):
        raise ValueError("Training requires an official-contract checkpoint with Adam state")
    model = SpatialPolicy(**saved["model_config"])
    model.load_state_dict(saved["model"], strict=True)
    torch.optim.Adam(model.parameters()).load_state_dict(saved["optimizer"])
    return saved, model


def writable_campaign(path):
    output = path.resolve(strict=True)
    immutable = (ARCHIVE, PARENT.parents[2], ROOT / "runs/exit-recovery-study",
                 ROOT / "runs/exit-switch-study", ROOT / "runs/skill-campaign")
    if any(output.is_relative_to(directory) for directory in immutable):
        raise ValueError("Choose a campaign output outside the frozen archives")
    return output


def verify_training(root):
    manifest = root / "manifest.json"
    rows = read(manifest)["maps"]
    if not rows or any(row.get("split") not in ("train", "validation") for row in rows):
        raise ValueError("Verify a training and development manifest before opening any WAD files")
    rows = [row for row in load_verified_maps(root) if row["split"] == "train"]
    expected = read(ARCHIVE / "phase2-partitions.json")["evaluations"]
    for name, evidence in expected.items():
        development = read(checked({"path": evidence["manifest"], "sha256": evidence["manifest_sha256"]}))["maps"]
        split = "validation" if name == "switch" else "test"
        audit_partition(rows, [row for row in development if row.get("split") == split])
    return rows


def prepare(output, extra):
    inputs = PRIOR["inputs"]
    for name in ("warm_start", "config", "reward", "training", "published", "warm_export"):
        checked(inputs[name])
    old = verify_training(checked(inputs["training"]).parent)
    hard = verify_training(checked(read(ARCHIVE / "hard-progression-plan.json")["manifest"]).parent)
    crossed = verify_training(checked(read(ARCHIVE / "lock-crossing-plan.json")["manifest"]).parent)
    replacement = [] if extra is None else verify_training(extra.resolve(strict=True))
    target = output / "mastery-training"
    target.mkdir(parents=True)
    maps, geometries, hashes = [], set(), set()
    for row in (*old, *hard, *crossed, *replacement):
        if row["sha256"] in hashes or row["geometry_sha256"] in geometries:
            continue
        path = target / f"{row['id']}.wad"
        if path.exists():
            raise ValueError(f"Duplicate training map name: {row['id']}")
        shutil.copyfile(row["path"], path)
        maps.append({**row, "path": str(path)})
        hashes.add(row["sha256"])
        geometries.add(row["geometry_sha256"])
    write(target / "manifest.json", {"maps": maps})
    verify_training(target)
    write(output / "continuity-profile.json", {
        "inputs": {**{name: inputs[name] for name in ("warm_start", "config", "reward", "training", "published")},
                   "lock_teacher": pin(BC_START)},
        "profile_source": pin(Path(__file__)), "profiles": PROFILES,
        "mastery_maps": len(maps), "mastery_manifest": pin(target / "manifest.json"),
        "training_only": True,
        "retention_lanes": LANES,
        "prior_hardware_comparison": pin(ARCHIVE / "hardware-followup/jobs.json"),
    })


class AdaptiveFamilies(MixedSchedule):
    """Favor underperforming families within fixed skill lanes."""

    def __init__(self, seed, *, warm_start=True):
        super().__init__(seed, warm_start=warm_start)
        self.history = defaultdict(lambda: deque(maxlen=24))

    def choose(self, maps, rng, steps):
        if self.pools is None:
            self.catalog(maps)
            self.groups = {}
            for lane, pool in self.pools.items():
                groups = defaultdict(list)
                for row in pool:
                    groups[row["family"]].append(row)
                self.groups[lane] = groups
        if rng not in self.assignments:
            index = len(self.assignments)
            if index >= len(self.lanes) or (self.warm_start and rng.getstate() != self.initial_streams[index]):
                raise ValueError("Adaptive lanes require the registered environment order")
            self.assignments[rng] = self.lanes[index]
        groups = self.groups[self.assignments[rng]]
        weights = [1.5 - (sum(self.history[name]) + 2.) / (len(self.history[name]) + 4.) for name in groups]
        family = rng.choices(tuple(groups), weights=weights, k=1)[0]
        maps = groups[family]
        return maps[int(rng.random() * len(maps))]


class LockTeacher(torch.nn.Module):
    """Train-only teacher. Steer one target distribution on lock approaches."""

    def __init__(self, warm, locks):
        super().__init__()
        self.warm, self.locks = warm, locks
        self.config, self.recurrent = warm.config, warm.recurrent

    def forward(self, observations, hidden, previous_action, episode_start):
        logits, values, state = self.warm(observations, hidden, previous_action, episode_start)
        other = self.locks(observations, hidden, previous_action, episode_start)[0]
        warm_groups = list(logits.split(GROUP_SIZES, -1))
        near_lock = ((observations["hit_type"] == RAY_HIT_TYPES.index("locked_door"))
                     & (observations["distance"] < .25)).any(-1)
        has_key = observations["vitals"][:, 8:14].any(-1)
        use_target = other.split(GROUP_SIZES, -1)[4]
        warm_groups[4] = torch.where((near_lock & has_key).unsqueeze(-1), use_target, warm_groups[4])
        return torch.cat(warm_groups, -1), values, state


def migrate(source, saved, parent, memory, folder, seed):
    if parent.recurrent:
        if parent.memory_kind != memory:
            raise ValueError("A recurrent architecture fork requires an MLP parent")
        return source, saved
    path = folder / "initial-memory.pt"
    if path.exists():
        migrated, _ = checkpoint(path)
        if migrated["provenance"]["parent_sha256"] != pin(source)["sha256"]:
            raise ValueError("Existing memory initialization belongs to another source")
        return path, migrated
    torch.manual_seed(seed)
    config = {**saved["model_config"], "recurrent": True, "memory": memory}
    model = SpatialPolicy(**config)
    missing, unexpected = model.load_state_dict(saved["model"], strict=False)
    prefixes = ("gru.", "memory_read.") if memory == "residual-gru" else ("spatial_memory.",)
    if unexpected or not missing or any(not name.startswith(prefixes) for name in missing):
        raise ValueError("Unexpected parameter difference in memory fork")
    if memory == "occupancy":
        torch.nn.init.zeros_(model.spatial_memory.fuse[0].weight)
        torch.nn.init.zeros_(model.spatial_memory.fuse[0].bias)
    parity = prove_initial_parity(parent.eval(), model.eval(), DATASET)
    migrated = {**saved, "model": model.state_dict(), "model_config": config,
                "optimizer": torch.optim.Adam(model.parameters(), lr=saved["args"]["lr"], eps=1e-5).state_dict(),
                "provenance": {"parent_sha256": pin(source)["sha256"], "fresh_adam": True, **parity}}
    save(path, migrated)
    return path, migrated


def measured_profile():
    current = read(ARCHIVE / "hardware-followup/jobs.json")
    previous = read(ARCHIVE / "hardware/jobs.json")
    combined = [item for item in (*previous, *current) if item.get("completed")]
    cpu = [item for item in combined if item["device"] == "cpu" and item["envs"] == len(LANES)]
    if not cpu:
        raise ValueError("Missing measured stable matched CPU configuration")
    return max(cpu, key=lambda item: item["decisions_per_second"])


def learner(args):
    torch.set_num_threads(1)
    root = writable_campaign(args.out)
    folder = root / "training" / args.label
    folder.mkdir(parents=True, exist_ok=True)
    profile = PROFILES[args.profile]
    meter_id = args.meter_id or args.label
    source = args.source.resolve(strict=True)
    saved, parent = checkpoint(source)
    if "memory" in profile:
        source, saved = migrate(source, saved, parent, profile["memory"], folder, args.seed)
    if profile.get("fresh_adam") and saved.get("args", {}).get(METADATA_KEY, {}).get("meter_id") != meter_id:
        if parent.recurrent:
            raise ValueError("The memory control requires the matching feedforward source")
        fresh = folder / "initial-fresh-adam.pt"
        if not fresh.exists():
            save(fresh, {**saved, "optimizer": torch.optim.Adam(parent.parameters(), lr=saved["args"]["lr"],
                                                                eps=1e-5).state_dict(),
                         "provenance": {"parent_sha256": pin(source)["sha256"], "fresh_adam": True}})
        source = fresh
        saved, _ = checkpoint(source)
    config = saved["model_config"]
    if config["hidden_size"] != 128 or config.get("vision"):
        raise ValueError("The structured profiles require a 128-wide nonvisual policy")
    model = config.get("memory", "gru" if config["recurrent"] else "mlp")
    if profile.get("anchor") and config["recurrent"]:
        raise ValueError("Choose a compatible feedforward parent for the frozen-policy anchor")
    root_map = args.map_root or (root / "mastery-training" if args.profile == "mastery" else
                                checked(PRIOR["inputs"]["training"]).parent)
    specs = verify_training(root_map)
    if set(LANES) - {classify(row) for row in specs}:
        raise ValueError("A dataset is missing a retention lane")
    initial = saved["trained_steps"]
    steps = initial + args.decisions
    fresh_rng = (args.restart_rng or saved["args"]["seed"] != args.seed or
                 len(saved.get("worker_map_rngs", ())) != len(LANES))
    schedule = AdaptiveFamilies(args.seed, warm_start=fresh_rng) if profile.get("adaptive_families") else (
        MixedSchedule(args.seed, warm_start=fresh_rng))
    measured = measured_profile()
    rollout = args.rollout or measured["rollout"]
    config_path = checked(PRIOR["inputs"]["config"])
    options = ["--config", config_path, "--map-root", root_map, "--resume", source, "--seed", args.seed,
               "--model", model, "--entity-mode", config["entity_mode"], "--out", folder,
               "--steps", steps, "--seconds", args.wall_seconds or max(90., 3 * args.active_seconds + 30),
               "--rollout", rollout, "--lr", PRIOR["training_config"]["lr"] * profile.get("rate", 1.),
               "--checkpoint-every", args.checkpoint_every, "--stop-file", root / "STOP",
               "--visualize" if args.visualize else "--headless"]
    if args.visualize:
        options += ["--speed", args.speed, "--telemetry-every", 16]
    if config["recurrent"]:
        options += ["--burn-in", 12]
    if fresh_rng:
        options.append("--restart-rng")
    reward_path = checked(PRIOR["inputs"]["reward"])
    record = dict(profile=args.profile, settings=profile, source=pin(source), dataset=pin(root_map / "manifest.json"),
                  reward=pin(reward_path), training_seed=args.seed, learner_source=pin(Path(__file__)),
                  train_source=pin(Path(train.__file__)), train_options=list(map(str, options)),
                  initial_steps=initial, initial_updates=saved["updates"], fresh_rng=fresh_rng,
                  meter_id=meter_id, active_limit_seconds=args.active_seconds, throughput_evidence=measured)
    path = folder / f"continuity-invocation-{initial}.json"
    if path.exists() and read(path) != record:
        raise ValueError("A prior invocation used a different source, curriculum, or training setup")
    if not path.exists():
        write(path, record)
    with ExitStack() as stack:
        updater = train.update
        if profile.get("anchor"):
            _, warm = checkpoint(PARENT)
            reference = warm
            if profile.get("lock_teacher"):
                _, lock = checkpoint(BC_START)
                reference = LockTeacher(warm, lock)
            updater = partial(update_anchored, reference=reference.eval().requires_grad_(False),
                              anchor_coefficient=profile["anchor"])
        reward = load_reward(reward_path)
        if profile.get("count_bonus"):
            from training.continuity_aux import count_reward
            reward = count_reward(reward)
        auxiliary = (saved.get("curriculum") or {}).get("continuity", {})
        state = {}
        if isinstance(schedule, AdaptiveFamilies):
            for family, outcomes in auxiliary.get("history", {}).items():
                schedule.history[family].extend(outcomes)
            state["history"] = lambda: {family: list(outcomes) for family, outcomes in schedule.history.items()}
            base = reward

            class MasteryReward(base):
                def summary(self):
                    result = super().summary()
                    schedule.history[self.spec["family"]].append(bool(result["drill_success"]))
                    return result

            reward = MasteryReward
        if profile.get("rnd_bonus"):
            from training.continuity_aux import Novelty
            novelty = Novelty(args.seed, auxiliary.get("rnd"))
            updater = partial(novelty.update, updater)
            state["rnd"] = novelty.state_dict
        if state:
            direct_save = train.save_checkpoint

            def save_state(*args, **kwargs):
                kwargs["curriculum"] = {"continuity": {name: get() for name, get in state.items()}}
                return direct_save(*args, **kwargs)

            stack.enter_context(patch.object(train, "save_checkpoint", save_state))
        stack.enter_context(patch.object(train, "update", updater))
        summary = run(options, meter_id=meter_id, active_limit_seconds=args.active_seconds,
                      select_training_map=schedule.choose, course_reward=reward)
    if summary.get("added_updates", 0) == 0:
        raise RuntimeError("This block has no committed learning yet")
    latest = Path(summary["checkpoint"])
    numbered = folder / f"step-{summary['steps']}.pt"
    if not numbered.exists():
        os.link(latest, numbered)
    destination = root / "exports" / args.label
    if summary["stop_reason"] in ("active-budget", "target") and not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".continuity-export-", dir=destination.parent) as work:
            staging = Path(work) / "artifact"
            export(numbered, staging)
            staging.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare")
    setup.add_argument("--out", type=Path, required=True)
    setup.add_argument("--map-root", type=Path)
    ppo = commands.add_parser("learner")
    ppo.add_argument("--out", type=Path, required=True)
    ppo.add_argument("--label", required=True)
    ppo.add_argument("--meter-id", help="Keep this ID unchanged when using a fresh folder for recovery")
    ppo.add_argument("--source", type=Path, required=True)
    ppo.add_argument("--profile", choices=PROFILES, required=True)
    ppo.add_argument("--seed", type=int, required=True)
    ppo.add_argument("--active-seconds", type=float, required=True)
    ppo.add_argument("--wall-seconds", type=float)
    ppo.add_argument("--decisions", type=int, default=2_000_000)
    ppo.add_argument("--checkpoint-every", type=int, default=32768)
    ppo.add_argument("--map-root", type=Path)
    ppo.add_argument("--rollout", type=int)
    ppo.add_argument("--restart-rng", action="store_true")
    ppo.add_argument("--visualize", action="store_true")
    ppo.add_argument("--speed", type=float, default=1.)
    from scripts.continuity_imitation import add_commands
    add_commands(commands)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(writable_campaign(args.out), args.map_root)
    elif args.command == "learner":
        if Path(args.label).name != args.label or args.label in ("", ".", ".."):
            parser.error("Use a simple training-block label")
        if args.decisions <= 0 or args.decisions % len(LANES):
            parser.error("Decision cap must be a positive multiple of eight")
        if not math.isfinite(args.active_seconds) or args.active_seconds <= 0:
            parser.error("Use a positive finite active-time cap")
        learner(args)
    else:
        if args.command == "clone" and (Path(args.label).name != args.label or args.label in ("", ".", "..")):
            parser.error("Use a simple training-block label")
        args.handler(args)


if __name__ == "__main__":
    main()
