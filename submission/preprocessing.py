"""Contract 1.0.0-draft spatial preprocessing; no engine or privileged inputs."""
from __future__ import annotations

import numpy as np
import torch

VITAL_SCALE = np.array([100, 100, 7, 200, 200, 50, 50, 300,
                        1, 1, 1, 1, 1, 1, 20, 20, 20], np.float32)
ENTITY_SCALE = np.array([4, 1, 1, 20, 20], np.float32)


def bounded(value, scale=1):
    # Clip before dividing, including finite float32 extrema, to avoid overflow.
    a = np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0,
                      posinf=1e6, neginf=-1e6)
    return np.clip(np.clip(a, -1e6, 1e6) / scale, -10, 10).astype(np.float32)


def pack(observation: dict) -> dict[str, np.ndarray]:
    entities = observation["visible_entities"]
    mask = np.asarray(entities["mask"], np.float32)
    return {
        "distance": bounded(observation["rays"]["distance"]),
        "hit_type": np.asarray(observation["rays"]["hit_type"], np.int64),
        "vitals": bounded(observation["vitals"], VITAL_SCALE),
        "entities": bounded(entities["features"], ENTITY_SCALE) * mask[:, None],
        "entity_type": np.where(mask > 0, entities["entity_type"], 0).astype(np.int64),
        "mask": mask,
    }


def tensor_batch(observations: list[dict], device="cpu") -> dict[str, torch.Tensor]:
    packed = [pack(o) for o in observations]
    return {k: torch.as_tensor(np.stack([p[k] for p in packed]), device=device)
            for k in packed[0]}
