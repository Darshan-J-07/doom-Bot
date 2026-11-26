"""Run matched PPO follow-ups for an experimental warm start."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts import adaptive_campaign_eval
from scripts.adaptive_imitation import PARENT, PARENT_SHA256, digest, load_parent, save
from scripts import adaptive_mixed
from scripts.adaptive_mixed import (OUT, WARM_STEPS, MixedSchedule, audit, checked,
                                   load_reward, pin, read, seconds_available, write)
from scripts.export_candidate import export
from submission.model import SpatialPolicy
from training import train


MEMORIES = ("residual-gru", "occupancy")


def prove_initial_parity(reference, student, dataset):
    data = torch.load(dataset, map_location="cpu", weights_only=True)
    receipt = read(dataset.with_name("collection.json"))
    if digest(dataset) != receipt["dataset_sha256"] or receipt["parent_sha256"] != PARENT_SHA256:
        raise ValueError("Memory parity requires the pinned real engine dataset")
    starts = torch.where(data["episode_start"])[0]
    rows = torch.linspace(0, len(starts) - 1, 18).long()
    hidden = student.initial_state(len(rows))
    changed = 0
    with torch.inference_mode():
        for offset in range(40):
            indices = (starts[rows] + offset).clamp(max=len(data["episode_start"]) - 1)
            observations = {key: value[indices] for key, value in data["observations"].items()}
            previous = data["previous_action"][indices]
            first = data["episode_start"][indices]
            expected, values, _ = reference(observations, reference.initial_state(len(rows)), previous, first)
            actual, predictions, next_hidden = student(observations, hidden, previous, first)
            if not torch.equal(actual, expected) or not torch.equal(predictions, values):
                raise ValueError("Memory warm start must preserve every policy logit and critic value")
            changed += int((hidden != next_hidden).any(dim=-1).sum())
            if first.any():
                _, _, explicit_reset = student(observations, hidden * (1 - first.float()).unsqueeze(-1),
                                                previous, first)
                if not torch.equal(explicit_reset[first], next_hidden[first]):
                    raise ValueError("The memory did not clear on map starts")
            hidden = next_hidden
    if changed == 0:
        raise ValueError("The stateful model never wrote to its history")
    return {"compared_observations": 40 * len(rows), "output_parity_exact": True,
            "nonzero_state_update_steps": changed, "dataset_sha256": receipt["dataset_sha256"]}


def initialize_memory(model_kind, dataset, output, seed):
    parent, reference = load_parent(PARENT)
    torch.manual_seed(seed)
    config = {**parent["model_config"], "recurrent": True, "memory": model_kind}
    model = SpatialPolicy(**config)
    missing, unexpected = model.load_state_dict(parent["model"], strict=False)
    allowed = ("gru.", "memory_read.") if model_kind == "residual-gru" else ("spatial_memory.",)
    if unexpected or not missing or any(not name.startswith(allowed) for name in missing):
        raise ValueError(f"Unexpected difference from the verified warm start: {missing}, {unexpected}")
    if model_kind == "occupancy":
        torch.nn.init.zeros_(model.spatial_memory.fuse[0].weight)
        torch.nn.init.zeros_(model.spatial_memory.fuse[0].bias)
    parity = prove_initial_parity(reference, model.eval(), dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=parent["args"]["lr"], eps=1e-5)
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = {"parent_sha256": PARENT_SHA256, "training_seed": seed, "model": config,
                  "initialization": "Exact MLP and zero memory readout; fresh Adam for both architectures",
                  "added_parameters": [name for name in missing if name in dict(model.named_parameters())],
                  "optimizer_state_entries": len(optimizer.state), **parity}
    save(output, {**parent, "args": {**parent["args"], "model": model_kind, "seed": seed},
                  "model": model.state_dict(), "model_config": config, "optimizer": optimizer.state_dict(),
                  "provenance": provenance})
    write(output.with_suffix(".json"), {**provenance, "checkpoint": pin(output)})
    print(f"{model_kind} seed {seed}: exact output parity on {parity['compared_observations']} observations.")


def ppo(output, args):
    plan_path = output / "mixed-plan.json"
    plan = read(plan_path)
    source = args.source.resolve(strict=True)
    if not source.is_relative_to(output.resolve()):
        raise ValueError("Use a separately prepared checkpoint inside the current campaign")
    parent = torch.load(source, map_location="cpu", weights_only=True)
    start = parent["trained_steps"]
    if start < WARM_STEPS:
        raise ValueError("Checkpoint predates the campaign")
    fresh_rng = start == WARM_STEPS
    target = start + args.decisions
    if args.decisions % plan["training_config"]["envs"]:
        raise ValueError("Decision budget must align with the environment batch")
    remaining = seconds_available(output)
    if remaining < 90 or (output / "STOP").exists():
        raise ValueError("Training cutoff is too close or a stop was requested")
    config = parent["model_config"]
    architecture = config.get("memory") if config["recurrent"] else "mlp"
    if architecture not in ("mlp", *MEMORIES) or config.get("vision"):
        raise ValueError("Unexpected post-comparison architecture")
    source_config = checked(plan["inputs"]["config"])
    reward = checked(plan["inputs"]["reward"])
    maps = checked(plan["inputs"]["training"]).parent
    schedule = MixedSchedule(args.seed, warm_start=fresh_rng)
    folder = output / "training" / args.label
    folder.mkdir(parents=True, exist_ok=False)
    options = ["--config", source_config, "--map-root", maps, "--resume", source, "--seed", args.seed,
               "--model", architecture, "--out", folder, "--steps", target, "--checkpoint-steps", target,
               "--seconds", min(args.seconds, remaining - 30), "--stop-file", output / "STOP"]
    if args.visualize:
        options += ["--visualize", "--speed", args.speed]
    else:
        options.append("--headless")
    if config["recurrent"]:
        options += ["--burn-in", 12]
    if fresh_rng:
        options.append("--restart-rng")
    if args.learning_rate is not None:
        options += ["--lr", args.learning_rate]
    if args.checkpoint_every > 0:
        options += ["--checkpoint-every", args.checkpoint_every]
    options = list(map(str, options))
    write(folder / "followup-invocation.json", {"label": args.label, "source": pin(source),
          "plan": pin(plan_path), "training_seed": args.seed, "initial_steps": start, "target_steps": target,
          "new_optimizer": not bool(parent["optimizer"]["state"]), "fresh_environment_rng": fresh_rng,
          "train_options": options, "source_files": [pin(Path(module)) for module in
                                                     (__file__, train.__file__, adaptive_mixed.__file__)]})
    train.main(options, select_training_map=schedule.choose, course_reward=load_reward(reward))
    summary = read(folder / "summary.json")
    if summary["steps"] != target:
        latest = folder / "latest.pt"
        if latest.is_file():
            recovered = folder / f"step-{summary['steps']}.pt"
            if not recovered.exists():
                save(recovered, torch.load(latest, map_location="cpu", weights_only=True))
        else:
            recovered = source
        raise RuntimeError(f"Incomplete {args.label}. Recover from {recovered}.")
    audit(folder, plan, "mixed", args.label)
    export(folder / f"step-{target}.pt", output / "exports" / args.label)


def compare_paired(output, name, pairs):
    settings = adaptive_campaign_eval.protocol(output)
    cohorts = adaptive_campaign_eval.panels(settings)
    contrasts = []
    for pair in pairs:
        control, candidate = pair.split("=", 1)
        if any(not label or Path(label).name != label for label in (control, candidate)):
            raise ValueError("Compare two existing export labels with CONTROL=CANDIDATE")
        runs = [read(output / "training" / label / "followup-invocation.json")
                for label in (control, candidate)]
        for key in ("plan", "training_seed", "initial_steps", "target_steps", "new_optimizer",
                    "fresh_environment_rng"):
            if runs[0][key] != runs[1][key]:
                raise ValueError(f"Comparison is not matched in {key}")
        if not runs[0]["new_optimizer"]:
            raise ValueError("The architectural or imitation arms need equally fresh Adam state")
        control_rows, candidate_rows, identities = {}, {}, {}
        for scope in ("switch", "familiar"):
            control_pin, control_rows[scope] = adaptive_campaign_eval.frozen_scope(
                output, scope, control, cohorts[scope])
            candidate_pin, candidate_rows[scope] = adaptive_campaign_eval.frozen_scope(
                output, scope, candidate, cohorts[scope])
            observed = {control: control_pin, candidate: candidate_pin}
            if identities and identities != observed:
                raise ValueError("Both evaluation scopes must share the exact trained artifacts")
            identities = observed
        paired = adaptive_campaign_eval.compare(control_rows["switch"], candidate_rows["switch"],
                                                control_rows["familiar"], candidate_rows["familiar"])
        contrasts.append({"training_seed": runs[0]["training_seed"], "control": control,
                          "candidate": candidate, "identities": identities, "paired": paired})
    if len({row["training_seed"] for row in contrasts}) != len(contrasts):
        raise ValueError("Each comparison requires distinct training seeds")
    path = output / "evaluation" / f"paired-followup-{name}.json"
    write(path, {"pair_count": len(pairs), "contrasts": contrasts})
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    init = subparsers.add_parser("initialize")
    init.add_argument("--model", choices=MEMORIES, required=True)
    init.add_argument("--dataset", type=Path, required=True)
    init.add_argument("--out", type=Path, required=True)
    init.add_argument("--seed", type=int, required=True)
    run = subparsers.add_parser("ppo")
    run.add_argument("--out", type=Path, default=OUT)
    run.add_argument("--source", type=Path, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--label", required=True)
    run.add_argument("--decisions", type=int, default=12000)
    run.add_argument("--seconds", type=int, default=240)
    run.add_argument("--learning-rate", type=float)
    run.add_argument("--checkpoint-every", type=int, default=0)
    run.add_argument("--visualize", action="store_true")
    run.add_argument("--speed", type=float, default=1.)
    comparison = subparsers.add_parser("compare")
    comparison.add_argument("--out", type=Path, default=OUT)
    comparison.add_argument("--name", required=True)
    comparison.add_argument("--pair", action="append", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "initialize":
        initialize_memory(args.model, args.dataset, args.out, args.seed)
    elif args.action == "compare":
        if Path(args.name).name != args.name:
            parser.error("Specify a simple comparison name")
        compare_paired(args.out, args.name, args.pair)
    else:
        if Path(args.label).name != args.label or not 1 <= args.decisions <= 250_000 or args.seconds < 1:
            parser.error("Use a directory label, 1 to 250,000 decisions, and a positive time budget")
        ppo(args.out, args)


if __name__ == "__main__":
    main()
