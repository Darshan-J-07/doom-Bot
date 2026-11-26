"""Collect successful engine trajectories and test a bounded actor-head warm start."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from doom_env import make_env
from doom_env.spaces import BUTTONS
from scripts.campaign_map_qa import route
from scripts.exit_recovery_qa import RecoveryWalk
from scripts.exit_switch_qa import vector
from submission.model import GROUP_SIZES, SpatialPolicy, action_groups, buttons
from submission.preprocessing import tensor_batch
from training.exit_switch_maps import twist_vector, write_map
from training.skills import KEY_INDEX
from training.universe import layout_digest, orient, rectangle


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "runs/limit-campaign-20261127/training/approach-41/step-314496.pt"
PARENT_SHA256 = "1810650838aea8b230cb9ad34dbc4da358afb05b4934b0ebe392739ce4305724"
PRACTICE = ROOT / "runs/exit-recovery-study/training-data/curriculum/manifest.json"
CLONED_GROUPS = (0, 1, 2, 4)  # travel, strafe, turn, use
LEARNER, TEACHER = 0, 1


def digest(path):
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def save(path, value):
    if path.exists():
        raise FileExistsError(path)
    partial = path.with_name(f".{path.name}.tmp")
    with partial.open("xb") as stream:
        torch.save(value, stream)
    partial.replace(path)


def load_parent(path):
    if digest(path) != PARENT_SHA256:
        raise ValueError("Expected the exact approach-41-12000 PPO checkpoint")
    parent = torch.load(path, map_location="cpu", weights_only=True)
    model = SpatialPolicy(**parent["model_config"])
    model.load_state_dict(parent["model"], strict=True)
    if model.recurrent or model.vision_kind is not None or parent.get("optimizer") is None:
        raise ValueError("The probe expects the verified structured MLP and its archived Adam state")
    return parent, model.eval()


def labels(action):
    assert action.shape == (len(BUTTONS),) and np.isin(action, (0, 1)).all()
    groups = torch.tensor([int(action[i]) + 2 * int(action[i + 1]) for i in (0, 2, 4)]
                          + action[6:].tolist(), dtype=torch.long)
    if not np.array_equal(buttons(groups).numpy(), action):
        raise ValueError("Opposing action buttons in demonstration")
    return groups


def practice_rows(source):
    grouped = {}
    for row in json.loads(source.read_text(encoding="utf-8"))["maps"]:
        if row["split"] == "train" and row.get("recovery_practice") and row.get("task") == "conditional":
            grouped.setdefault((row["level"], row["exit_type"]), []).append(row)
    if len(grouped) != 8:
        raise ValueError("Expected four training stages with both switch and walkover maps")
    return [row for (level, mode), rows in sorted(grouped.items())
            for row in rows[:12 if mode == "switch" else 4]]


def door_plan(rng, *, locked, guarded):
    length = rng.randint(4, 7)
    key_drop, wing = rng.randint(2, 4), rng.randint(2, 5)
    before, after = {(x, 0) for x in range(length + 1)}, {(length + 1, y) for y in range(wing + 1)}
    stair = {(length + 2, wing), (length + 3, wing), (length + 4, wing)}
    # A T entrance and a kinked exit wing require an outward key trip and a return.
    start_x, key_x = rng.randint(0, 1), rng.randint(2, length - 1)
    entrance = {(start_x, y) for y in range(1, rng.randint(2, 4) + 1)}
    key_arm = {(key_x, -y) for y in range(1, key_drop + 1)}
    key_arm |= {(key_x + 1, -key_drop)}
    blind = {(length + 1, -y) for y in range(1, rng.randint(1, 3) + 1)}
    cells = dict.fromkeys(before | after | stair | entrance | key_arm | blind, ".")
    start = max(entrance)
    exit_cell, door, key = (length + 4, wing), (length, 0), (key_x + 1, -key_drop)
    targets = [door, exit_cell]
    cells.update({start: "P", exit_cell: "E", door: "D"})
    if locked:
        color = rng.choice("rby")
        cells.update({key: color, door: color.upper()})
        targets.insert(0, key)
    if guarded:
        guard = (length + 2, wing + 1)
        cells.update({guard: "z", (length + 3, wing + 1): "h"})
    for i, point in enumerate(targets):
        cells[point] = str(i) + cells[point]
    turns, mirror = rng.randrange(4), bool(rng.randrange(2))
    cells = orient(cells, turns, mirror)
    for point, glyph in cells.items():
        if len(glyph) == 2:
            targets[int(glyph[0])] = point
            cells[point] = glyph[1]
    tile = rng.choice((78, 90, 98)), rng.choice((74, 84, 94))
    if turns % 2:
        tile = tile[::-1]
    heading = rng.randrange(360)
    offset = twist_vector((rng.randint(-12, 12), rng.randint(-12, 12)), turns, mirror)
    return cells, targets, twist_vector((1, 0), turns, mirror), tile, heading, offset, 1500


def build_progression_maps(out, source, *known_manifests):
    # A previous training manifest is the only source of demonstration levels.
    used = {row["geometry_sha256"] for manifest in (source, *known_manifests)
            for row in json.loads(manifest.read_text(encoding="utf-8"))["maps"]
            if row.get("geometry_sha256")}
    rng = random.Random(680417)
    out.mkdir()
    for family, task, count, locked, guarded in (
            ("bc-closed-wing", "conditional", 6, False, False),
            ("bc-key-detour", "keycard", 10, True, False),
            ("bc-guarded-detour", "gauntlet", 8, True, True)):
        for index in range(count):
            for _ in range(100):
                plan = door_plan(rng, locked=locked, guarded=guarded)
                shape = layout_digest(plan[0], labeled=False)
                if shape not in used:
                    used.add(shape)
                    break
            else:
                raise ValueError(f"Duplicate training geometry for {family}")
            name = f"{family}-{index:03d}"
            yield write_map(out / f"{name}.wad", "train", family, name, plan, "switch", rng, task=task)


class RecordingWalk(RecoveryWalk):
    def __init__(self, env, row, seed):
        self.previous = np.zeros(len(BUTTONS), dtype=np.int8)
        self.samples = []
        self.source = TEACHER
        self.visited_cells = set()
        targets = [tuple(goal) for goal in row["drill"]["targets"]]
        cells = {tuple((x, y)): glyph for x, y, glyph in row["drill"]["cells"]}
        recorded_door = next((cell for cell in targets if cells[cell] in "DRBY"), None)
        super().__init__(env, row, recorded_door, seed)
        self.initial_door_gap = self.maximum_gap

    def step(self, *names):
        action = vector(names)
        self.visited_cells.add(self.cell())
        self.samples.append((tensor_batch([self.obs]), torch.from_numpy(self.previous.copy()),
                             labels(action), self.decisions == 0, self.source))
        super().step(*names)
        self.previous = action

    def learner_prefix(self, model, length, seed):
        self.source = LEARNER
        generator = torch.Generator().manual_seed(seed)
        with torch.inference_mode():
            for _ in range(length):
                packed = tensor_batch([self.obs])
                previous = torch.from_numpy(self.previous).unsqueeze(0)
                first = torch.tensor([self.decisions == 0])
                logits, _, _ = model(packed, model.initial_state(), previous, first)
                action = buttons(action_groups(logits, generator=generator))[0].numpy()
                self.step(*(name for name, pressed in zip(BUTTONS, action, strict=True) if pressed))
                if self.state["exit_reached"]:
                    break
        self.source = TEACHER

    def finish(self):
        # A previous learner action may already have taken the key.
        original = self.row["drill"]["targets"]
        remaining = [goal for goal in original
                     if (glyph := self.layout[tuple(goal)]) not in KEY_INDEX
                     or not self.state["keys"][KEY_INDEX[glyph]]]
        if remaining == original:
            return super().finish()
        # Keep the key's floor traversable even after its goal is satisfied.
        for goal in original:
            if goal not in remaining:
                self.layout[tuple(goal)] = "."
        self.row = {**self.row, "drill": {**self.row["drill"], "targets": remaining}}
        return super().finish()


def collect(out, parent_path, source, seconds):
    start = time.monotonic()
    _, model = load_parent(parent_path)
    source = source.resolve()
    out.mkdir(parents=True, exist_ok=False)
    original = practice_rows(source)
    generated = list(build_progression_maps(out / "maps", source,
        ROOT / "runs/exit-recovery-study/maps/manifest.json",
        ROOT / "runs/exit-switch-study/maps/manifest.json",
        ROOT / "runs/skill-campaign/maps/manifest.json"))
    rows = [*original, *generated]
    if len({row["geometry_sha256"] for row in rows}) != len(rows):
        raise ValueError("Duplicate demonstration geometry")
    write_json(out / "training-manifest.json", {"maps": rows})
    episodes, failed, data = [], [], []
    for index, row in enumerate(rows):
        if row["split"] != "train" or digest(Path(row["path"])) != row["sha256"]:
            raise ValueError(f"Unverified demonstration map {row['id']}")
        pure_success = False
        for kind in ("teacher", "learner-then-rescue"):
            prefix = 0 if kind == "teacher" else (8 + index % 8 if int(row["ticks"]) <= 280
                                                   else 12 + (index * 7) % 33)
            seeds = (0, 11, 23) if kind == "teacher" else (1709 + index,)
            for seed in seeds:
                if time.monotonic() - start >= seconds:
                    raise TimeoutError("Demonstration collection exceeded its wall clock limit")
                try:
                    with make_env(row["path"], row["map_name"], episode_ticks=int(row["ticks"]),
                                  render_buffers=False) as env:
                        walk = RecordingWalk(env, row, seed)
                        spawn = walk.position()
                        walk.learner_prefix(model, prefix, 997 + seed + index)
                        takeover = walk.position()
                        already_completed = bool(walk.state["exit_reached"])
                        if already_completed:
                            walk.check_exit()
                            receipt = walk.receipt()
                        else:
                            receipt = walk.finish()
                        if row["task"] in {"keycard", "gauntlet"}:
                            assert receipt["keycards"] == 1 and walk.initial_door_gap == 0
                            assert walk.crossed and walk.maximum_gap >= 56
                    result = dict(map=row["id"], family=row["family"], stage=row["stage"],
                                  exit_type=row["exit_type"], seed=seed, controller=kind,
                                  first=len(data), stop=len(data) + len(walk.samples),
                                  learner_prefix_requested=prefix, spawn=list(spawn), takeover=list(takeover),
                                  visited_cells=len(walk.visited_cells), learner_exited=already_completed,
                                  initial_door_gap=walk.initial_door_gap, door_crossed=walk.crossed,
                                  maximum_door_gap=walk.maximum_gap, **receipt)
                    episodes.append(result)
                    data.extend(walk.samples)
                except (AssertionError, RuntimeError, ValueError) as failure:
                    failed.append(dict(map=row["id"], seed=seed, controller=kind, error=repr(failure)))
                    continue
                pure_success |= kind == "teacher"
                break
        if not pure_success and row in generated:
            raise ValueError(f"Generated teacher fixture could not be verified: {row['id']}: {failed[-3:]}")
        print(json.dumps({"collected": index + 1, "planned": len(rows), "map": row["id"],
                          "trajectory_steps": len(data), "teacher_verified": pure_success}), flush=True)
    packed, previous, actions, starts, controllers = zip(*data, strict=True)
    tensors = {"observations": {key: torch.cat([item[key] for item in packed]) for key in packed[0]},
               "previous_action": torch.stack(previous), "actions": torch.stack(actions),
               "episode_start": torch.tensor(starts), "controller": torch.tensor(controllers)}
    save(out / "trajectories.pt", tensors)
    write_json(out / "collection.json", dict(parent_sha256=PARENT_SHA256, geometry_manifest_sha256=
               digest(out / "training-manifest.json"), dataset_sha256=digest(out / "trajectories.pt"),
               source_manifest_sha256=digest(source), collection_seconds=time.monotonic() - start,
               steps=len(data), steps_by_controller=Counter(controllers), episodes=episodes,
               failed_attempts=failed, source_sha256=digest(Path(__file__)),
               policy_inputs=list(tensors["observations"])))


def logits(model, data, indices):
    obs = {key: value[indices] for key, value in data["observations"].items()}
    return model(obs, model.initial_state(len(indices)), data["previous_action"][indices],
                 data["episode_start"][indices])[0]


def control(parent_path, out):
    parent, model = load_parent(parent_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=parent["args"]["lr"], eps=1e-5)
    save(out, {**parent, "optimizer": optimizer.state_dict(),
               "provenance": {"source": PARENT_SHA256, "probe": "fresh-Adam no-BC control"}})


def sample_weights(data, episodes):
    controllers = data["controller"]
    teacher = torch.zeros(len(controllers))
    learner = torch.zeros(len(controllers))
    covered = 0
    for episode in episodes:
        first, stop = episode["first"], episode["stop"]
        if first != covered or stop <= first:
            raise ValueError("Collection receipts do not describe contiguous episodes")
        sources = controllers[first:stop]
        for category, weights in ((TEACHER, teacher), (LEARNER, learner)):
            matches = sources == category
            count = int(matches.sum())
            if count:
                weights[first:stop][matches] = 1 / count
        covered = stop
    if covered != len(controllers) or not teacher.any() or not learner.any():
        raise ValueError("Both teacher and real learner-state samples are required")
    # Interaction is uncommon even in short routes. Weight it within the
    # otherwise trajectory-balanced sampling budget.
    teacher[data["actions"][:, 4] != 0] *= 3
    return teacher, learner


def fit(parent_path, dataset, out, seed, updates, seconds, use_loss_weight=1.0):
    start = time.monotonic()
    parent, reference = load_parent(parent_path)
    model = SpatialPolicy(**parent["model_config"])
    model.load_state_dict(parent["model"], strict=True)
    model.requires_grad_(False)
    model.actor.requires_grad_(True)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=3e-5)
    data = torch.load(dataset, weights_only=True)
    generator = torch.Generator().manual_seed(seed)
    expected = json.loads((dataset.parent / "collection.json").read_text(encoding="utf-8"))
    if digest(dataset) != expected["dataset_sha256"] or expected["parent_sha256"] != PARENT_SHA256:
        raise ValueError("Dataset identity changed")
    teacher_weights, learner_weights = sample_weights(data, expected["episodes"])
    use_weights = None if use_loss_weight == 1 else torch.tensor([1., use_loss_weight])
    draw_hash = sha256()
    out.mkdir(parents=True, exist_ok=False)
    records = []
    milestones = {20, updates}
    for update in range(1, updates + 1):
        if time.monotonic() - start >= seconds:
            raise TimeoutError("Imitation optimization exceeded its wall clock limit")
        targets = torch.multinomial(teacher_weights, 128, replacement=True, generator=generator)
        support = torch.multinomial(learner_weights, 128, replacement=True, generator=generator)
        draw_hash.update(torch.stack((targets, support)).numpy().tobytes())
        predicted = logits(model, data, targets).split(GROUP_SIZES, -1)
        new = logits(model, data, support).split(GROUP_SIZES, -1)
        with torch.no_grad():
            old = logits(reference, data, support).split(GROUP_SIZES, -1)
        nll = torch.stack([F.cross_entropy(predicted[group], data["actions"][targets, group],
                                           weight=use_weights if group == 4 else None)
                           for group in CLONED_GROUPS]).mean()
        kl = torch.stack([F.kl_div(new[group].log_softmax(-1), old[group].softmax(-1),
                                   reduction="batchmean") for group in CLONED_GROUPS]).sum()
        loss = nll + 2. * kl
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite supervised loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.actor.parameters(), .5, error_if_nonfinite=True)
        optimizer.step()
        record = dict(update=update, nll=float(nll.detach()), anchor_kl=float(kl.detach()),
                      gradient_norm=float(grad_norm), elapsed_seconds=time.monotonic() - start)
        if use_weights is not None:
            positives = int(data["actions"][targets, 4].sum())
            positive_mass = positives * use_loss_weight
            record.update(positive_use_targets=positives, negative_use_targets=len(targets) - positives,
                          normalized_positive_use_loss_mass=positive_mass / (positive_mass + len(targets) - positives))
        records.append(record)
        if update in milestones:
            changed = [key for key, tensor in model.state_dict().items()
                       if not torch.equal(tensor, parent["model"][key])]
            if changed != ["actor.weight", "actor.bias"]:
                raise ValueError(f"Unexpected BC parameter updates: {changed}")
            for key in changed:
                before = parent["model"][key].split(GROUP_SIZES)
                after = model.state_dict()[key].split(GROUP_SIZES)
                if any(not torch.equal(before[group], after[group]) for group in (3, 5)):
                    raise ValueError("BC changed the frozen attack or weapon-selection logits")
            frozen = {**parent, "model": model.state_dict(),
                      "args": {**parent["args"], "seed": seed},
                      "optimizer": torch.optim.Adam(model.parameters(), lr=parent["args"]["lr"],
                                                    eps=1e-5).state_dict(),
                      "provenance": {"source": PARENT_SHA256, "dataset": expected["dataset_sha256"],
                                     "supervised_updates": update, "ppo_optimizer": "fresh Adam",
                                     "supervised_seed": seed}}
            if use_weights is not None:
                frozen["provenance"].update(use_positive_ce_weight=use_loss_weight,
                                            sampled_indices_sha256=draw_hash.hexdigest())
            save(out / f"bc-{update:03d}.pt", frozen)
            print(json.dumps(record), flush=True)
    write_json(out / "learning.json", {"elapsed_seconds": time.monotonic() - start, "records": records,
               "source_sha256": digest(Path(__file__)), "supervised_seed": seed,
               "use_positive_ce_weight": use_loss_weight, "sampled_indices_sha256": draw_hash.hexdigest()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "fit", "control"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--manifest", type=Path, default=PRACTICE)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--seed", type=int, default=59)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--use-loss-weight", type=float, default=1.0,
                        help="positive USE label weight inside only the USE cross-entropy")
    args = parser.parse_args()
    if not 1 <= args.updates <= 120 or args.seconds < 1:
        parser.error("Updates must be 1 to 120 and seconds must be positive")
    if not math.isfinite(args.use_loss_weight) or not 1 <= args.use_loss_weight <= 16:
        parser.error("Positive USE label weight must be between 1 and 16")
    torch.set_num_threads(1)
    if args.action == "collect":
        collect(args.out.resolve(), args.parent, args.manifest, args.seconds)
    elif args.action == "control":
        control(args.parent, args.out)
    else:
        if not args.dataset:
            parser.error("fit requires --dataset")
        fit(args.parent, args.dataset, args.out, args.seed, args.updates, args.seconds, args.use_loss_weight)


if __name__ == "__main__":
    main()
