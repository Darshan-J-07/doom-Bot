"""Export a frozen checkpoint into a separate DoomAgent submission tree."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

import torch
from doom_env import CONTRACT_VERSION

from submission.model import SpatialPolicy


ROOT = Path(__file__).resolve().parents[1]
INFERENCE_FIELDS = ("model", "model_config", "contract_version", "trained_steps", "updates", "deterministic", "provenance")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def export(checkpoint, output, *, deterministic=None):
    torch.set_num_threads(1)
    checkpoint = checkpoint.resolve()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["contract_version"] != CONTRACT_VERSION or saved["trained_steps"] < 1 or saved["updates"] < 1:
        raise ValueError("A trained checkpoint for the installed official contract is required")
    model = SpatialPolicy(**saved["model_config"])
    model.load_state_dict(saved["model"], strict=True)
    if any(not torch.isfinite(weight).all() for weight in model.state_dict().values()):
        raise ValueError("Checkpoint contains nonfinite tensors")
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "submission"
    shutil.copytree(ROOT / "submission", destination, ignore=shutil.ignore_patterns("weights", "__pycache__", "*.pyc"))
    weight = destination / "weights/best_model.pt"
    weight.parent.mkdir()
    data = {key: saved[key] for key in INFERENCE_FIELDS if key in saved}
    data["deterministic"] = saved.get("deterministic", False) if deterministic is None else deterministic
    torch.save(data, weight)
    files = {str(path.relative_to(output)): digest(path) for path in sorted(destination.rglob("*")) if path.is_file()}
    result = dict(source_checkpoint=str(checkpoint), source_checkpoint_sha256=digest(checkpoint),
                  model_config=data["model_config"], deterministic=data["deterministic"],
                  trained_steps=data["trained_steps"], updates=data["updates"], files=files,
                  training_config=saved.get("args"), dataset_sha256=saved.get("dataset_sha256"))
    (output / "export.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(output.resolve()), "trained_steps": data["trained_steps"],
                      "weights_sha256": files["submission/weights/best_model.pt"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    export(args.checkpoint, args.out, deterministic=args.deterministic)
