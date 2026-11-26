"""Materialize the predeclared final map seeds in a fresh ignored output directory."""
import argparse
from pathlib import Path
import json

from training.generate_maps import create_map
from training.maps import load_verified_maps
from scripts.compare_phase2 import TIER_WEIGHTS

MAP_SEED_OFFSET = 410000
MAPS_PER_TIER = 5
FINAL_SEEDS = tuple(range(41000, 41005))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    previous = {row["sha256"] for row in load_verified_maps()}
    rows = []
    for stage_index, stage in enumerate(TIER_WEIGHTS):
        folder = args.out / "test" / stage
        folder.mkdir(parents=True)
        for index in range(MAPS_PER_TIER):
            row = create_map(folder / f"{index:02}.wad", stage, MAP_SEED_OFFSET + stage_index * 100 + index)
            if row["sha256"] in previous:
                raise ValueError(f"Duplicate map bytes for {row['path']}")
            previous.add(row["sha256"])
            rows.append(dict(row, split="test"))
    manifest = dict(maps=rows, seeds=FINAL_SEEDS, purpose="Final comparison after candidate freeze")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    load_verified_maps(args.out)


if __name__ == "__main__":
    main()
