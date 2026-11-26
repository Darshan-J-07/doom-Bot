"""Verify the existing curriculum from actual WAD bytes before consuming it."""
from __future__ import annotations
import hashlib
from itertools import combinations
import json
from pathlib import Path
from doom_env.mapinfo import load_mapinfo
from doom_env.wad import list_maps, read_wad


def load_verified_maps(root="training/maps"):
    root = Path(root).resolve()
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing map manifest: {manifest}; do not regenerate existing maps silently")
    rows = json.loads(manifest.read_text(encoding="utf-8"))["maps"]
    seen_paths, split_hashes = set(), {s: set() for s in ("train", "validation", "development", "test")}
    geometry_owner = {}
    for row in rows:
        path = Path(row["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Map path leaves the dataset root: {path}")
        if path in seen_paths:
            raise ValueError(f"Duplicate map path: {path}")
        seen_paths.add(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"WAD hash mismatch: {path}")
        split = row["split"]
        if split not in split_hashes:
            raise ValueError(f"Unknown map split: {split}")
        geometry = row.get("geometry_sha256")
        if geometry is not None:
            if geometry_owner.setdefault(geometry, split) != split:
                raise ValueError(f"Normalized geometry overlaps partitions: {path}")
        split_hashes[split].add(digest)
        map_name = row.get("map_name", "MAP01")
        if map_name not in list_maps(read_wad(path)):
            raise ValueError(f"Missing {map_name} in {path}")
        info = load_mapinfo(path, map_name)
        if not 0 < info.num_sectors <= 2048:
            raise ValueError(f"Invalid or unsupported sector count in {path}")
    for left, right in combinations(split_hashes, 2):
        if split_hashes[left] & split_hashes[right]:
            raise ValueError(f"Actual WAD bytes overlap between {left} and {right}")
    if not rows:
        raise ValueError("The map manifest is empty")
    return rows
