"""Collect real training trajectories and optimize a short actor-only clone."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import random
from tempfile import NamedTemporaryFile
import time

import torch
from torch.nn import functional as F

from doom_env import make_env
from scripts.adaptive_imitation import CLONED_GROUPS, RecordingWalk, digest, logits, sample_weights
from scripts.adaptive_mixed import pin, read
from submission.model import GROUP_SIZES
from training.learning_meter import METADATA_KEY, write_json


def save_checkpoint(path, payload):
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}-", delete=False) as partial:
        temporary = Path(partial.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def collect(args):
    from scripts.continuity_learning import PARENT, checkpoint, verify_training, writable_campaign

    torch.set_num_threads(1)
    root = writable_campaign(args.out)
    directory = root / "imitation-data"
    directory.mkdir()
    _, model = checkpoint(PARENT)
    model.eval()
    rows = [row for row in verify_training(args.map_root)
            if row.get("task") in ("conditional", "keycard", "gauntlet")]
    draws = random.Random(args.seed)

    def stratified(pool):
        by_family = {}
        for row in pool:
            by_family.setdefault(row["family"], []).append(row)
        groups = list(by_family.values())
        draws.shuffle(groups)
        for group in groups:
            draws.shuffle(group)
        return [group[index] for index in range(max(map(len, groups), default=0))
                for group in groups if index < len(group)]

    post_lock = [row for row in rows if row.get("post_lock") and row["family"].startswith("continuity-")]
    hard = [row for row in rows if row["task"] in ("keycard", "gauntlet")]
    unguarded = [row for row in rows if row["task"] == "conditional" and not row.get("post_lock")]
    selection = [*stratified(post_lock)[:args.maps // 2], *stratified(hard)[:args.maps // 3]]
    selected = {row["id"] for row in selection}
    remainder = [row for row in (*stratified(unguarded), *stratified(rows))
                 if row["id"] not in selected]
    for row in remainder:
        if len(selection) >= args.maps:
            break
        if row["id"] not in selected:
            selection.append(row)
            selected.add(row["id"])
    data, episodes, failed = [], [], []
    start = time.monotonic()
    for index, row in enumerate(selection):
        for prefix in (0, draws.randint(8, 34)):
            if time.monotonic() - start >= args.seconds or (root / "STOP").exists():
                break
            for seed in (0, 11, 23) if prefix == 0 else (2719 + index,):
                try:
                    with make_env(row["path"], row["map_name"], episode_ticks=row["ticks"],
                                  render_buffers=False) as env:
                        walk = RecordingWalk(env, row, seed)
                        walk.learner_prefix(model, prefix, args.seed + seed + index)
                        if walk.state["exit_reached"]:
                            walk.check_exit()
                            evidence = walk.receipt()
                        else:
                            evidence = walk.finish()
                    if not evidence["exit"]:
                        raise ValueError("Engine did not confirm the demonstration exit")
                    if (row["task"] != "conditional"
                            and (evidence["keycards"] < 1 or not walk.crossed or walk.maximum_gap < 56)):
                        raise ValueError("The full progression demonstration did not use a verified locked passage")
                except (AssertionError, RuntimeError, ValueError) as error:
                    failed.append(dict(map=row["id"], seed=seed, prefix=prefix, error=str(error)))
                    continue
                episodes.append(dict(map=row["id"], family=row["family"], stage=row["stage"], task=row["task"],
                                     post_lock=bool(row.get("post_lock")), seed=seed,
                                     learner_prefix=prefix, first=len(data), stop=len(data)+len(walk.samples),
                                     door_crossed=walk.crossed, initial_door_gap=walk.initial_door_gap, **evidence))
                data.extend(walk.samples)
                break
    if not data:
        raise RuntimeError("No verified native-engine demonstration episodes")
    packed, previous, actions, starts, controller = zip(*data, strict=True)
    tensors = {"observations": {key: torch.cat([item[key] for item in packed]) for key in packed[0]},
               "previous_action": torch.stack(previous), "actions": torch.stack(actions),
               "episode_start": torch.tensor(starts), "controller": torch.tensor(controller)}
    sample_weights(tensors, episodes)
    metadata = directory / "trajectories.pt"
    save_checkpoint(metadata, tensors)
    write_json(directory / "collection.json", dict(dataset_sha256=digest(metadata),
               source_manifest=pin(args.map_root / "manifest.json"), steps=len(data),
               planned_maps=selection and [row["id"] for row in selection],
               steps_by_controller=dict(Counter(controller)), episodes=episodes, failed_attempts=failed,
               collection_seconds=time.monotonic()-start, training_credit_seconds=0.,
               policy_inputs=list(tensors["observations"])))


def clone(args):
    from scripts.continuity_learning import DATASET, checkpoint, writable_campaign

    torch.set_num_threads(1)
    root = writable_campaign(args.out)
    meter_id = args.meter_id or args.label
    source = args.source.resolve(strict=True)
    folder = root / "training" / args.label
    folder.mkdir(parents=True, exist_ok=True)
    parent, model = checkpoint(source)
    _, reference = checkpoint(source)
    if model.recurrent:
        raise ValueError("The actor-only supervised arm requires the feedforward parent")
    parent_seed = parent["args"]["seed"]
    model.requires_grad_(False)
    model.actor.requires_grad_(True)
    reference.eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=3e-5)
    prior = parent.get("curriculum") or {}
    state = prior.get("continuity_bc") if prior.get("continuity_bc", {}).get("meter_id") == meter_id else None
    dataset = args.dataset or root / "imitation-data/trajectories.pt"
    if args.dataset is None:
        if state and "dataset" in state:
            dataset = Path(state["dataset"]["path"])
        elif not dataset.exists():
            dataset = DATASET
    evidence = read(dataset.with_name("collection.json"))
    if digest(dataset) != evidence["dataset_sha256"]:
        raise ValueError("Demonstration tensor identity disagrees with its native-engine receipt")
    data = torch.load(dataset, map_location="cpu", weights_only=True)
    weights, support = sample_weights(data, evidence["episodes"])
    generator = torch.Generator().manual_seed(args.seed)
    summary = dict(meter_id=meter_id, active_training_seconds=0., active_training_ns=0, added_decisions=0,
                   added_updates=0, added_supervised_updates=0, status="running", checkpoint=str(source),
                   checkpoint_steps=parent["trained_steps"], checkpoint_updates=parent["updates"])
    settings = {"source": pin(source), "dataset": pin(dataset), "seed": args.seed,
                "supervised_updates": args.max_updates, "actor_groups": CLONED_GROUPS,
                "regularizer": 2.0, "source_code": pin(Path(__file__))}
    # Restart from the last committed optimizer and random stream for this same BC segment.
    latest = folder / "latest.pt"
    if state:
        if (state["dataset_sha256"] != evidence["dataset_sha256"] or state["seed"] != args.seed
                or state["requested_updates"] != args.max_updates):
            raise ValueError("This BC block must resume with its original dataset, seed, and cap")
        _, reference = checkpoint(Path(state["original_source"]["path"]))
        if pin(Path(state["original_source"]["path"])) != state["original_source"]:
            raise ValueError("The original BC reference checkpoint has changed")
        reference.eval().requires_grad_(False)
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["rng"])
        summary.update(state["summary"])
        summary["status"] = "running"
    else:
        write_json(folder / "imitation-invocation.json", settings)
    write_json(folder / "training-progress.json", summary)
    accumulated = int(summary["active_training_ns"])
    start_update = int(summary["added_supervised_updates"])
    with (folder / "imitation-learning.jsonl").open("a", encoding="utf-8") as log:
        for update in range(start_update+1, args.max_updates+1):
            if (root / "STOP").exists():
                break
            start = time.monotonic_ns()
            selected = torch.multinomial(weights, 128, replacement=True, generator=generator)
            sampled = torch.multinomial(support, 128, replacement=True, generator=generator)
            predicted = logits(model, data, selected).split(GROUP_SIZES, -1)
            current = logits(model, data, sampled).split(GROUP_SIZES, -1)
            with torch.no_grad():
                frozen = logits(reference, data, sampled).split(GROUP_SIZES, -1)
            imitation = torch.stack([F.cross_entropy(predicted[group], data["actions"][selected, group])
                                     for group in CLONED_GROUPS]).mean()
            anchor = torch.stack([F.kl_div(current[group].log_softmax(-1), frozen[group].softmax(-1),
                                            reduction="batchmean") for group in CLONED_GROUPS]).sum()
            loss = imitation + 2 * anchor
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite imitation objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.actor.parameters(), .5, error_if_nonfinite=True)
            optimizer.step()
            elapsed = time.monotonic_ns() - start
            accumulated += elapsed
            summary.update(active_training_ns=accumulated, active_training_seconds=accumulated/1e9,
                           checkpoint=str(latest), added_supervised_updates=update)
            ppo = parent["optimizer"]
            param_ids = ppo["param_groups"][0]["params"]
            actor_ids = {ident for ident, (name, _) in zip(param_ids, model.named_parameters(), strict=True)
                         if name.startswith("actor.")}
            preserved = {ident: moments for ident, moments in ppo["state"].items() if ident not in actor_ids}
            # Preserve the exact moments of unchanged PPO tensors and reinitialize
            # only the actor moments after a supervised optimizer changes that head.
            checkpoint_state = {
                **parent, "model": model.state_dict(), "args": dict(parent["args"]),
                "optimizer": {**ppo, "state": preserved},
                "curriculum": {"continuity_bc": {"meter_id": meter_id, "summary": dict(summary),
                               "original_source": state["original_source"] if state else settings["source"],
                               "dataset": settings["dataset"], "dataset_sha256": evidence["dataset_sha256"], "seed": args.seed,
                               "requested_updates": args.max_updates, "rng": generator.get_state(),
                               "optimizer": optimizer.state_dict()}},
            }
            checkpoint_state["args"].pop(METADATA_KEY, None)
            checkpoint_state["args"]["seed"] = parent_seed
            save_checkpoint(latest, checkpoint_state)
            if update % 60 == 0 or update == args.max_updates:
                numbered = folder / f"bc-{update:05d}.pt"
                if not numbered.exists():
                    os.link(latest, numbered)
            write_json(folder / "training-progress.json", summary)
            log.write(json.dumps(dict(update=update, loss=float(loss.detach()),
                                      nll=float(imitation.detach()), kl=float(anchor.detach()),
                                      grad=float(grad), active_training_seconds=summary["active_training_seconds"]))+"\n")
            log.flush()
    summary["status"] = "completed" if summary["added_supervised_updates"] == args.max_updates else "interrupted"
    write_json(folder / "training-progress.json", summary)
    write_json(folder / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def add_commands(commands):
    collector = commands.add_parser("collect")
    collector.add_argument("--out", required=True, type=Path)
    collector.add_argument("--map-root", required=True, type=Path)
    collector.add_argument("--seed", type=int, default=3159)
    collector.add_argument("--maps", type=int, choices=range(12, 121), default=48)
    collector.add_argument("--seconds", type=float, default=210.)
    collector.set_defaults(handler=collect)
    supervised = commands.add_parser("clone")
    supervised.add_argument("--out", required=True, type=Path)
    supervised.add_argument("--source", required=True, type=Path)
    supervised.add_argument("--dataset", type=Path)
    supervised.add_argument("--label", required=True)
    supervised.add_argument("--meter-id", help="Keep this ID unchanged when using a fresh folder for recovery")
    supervised.add_argument("--seed", type=int, required=True)
    supervised.add_argument("--max-updates", type=int, choices=range(1, 601), default=240)
    supervised.set_defaults(handler=clone)
