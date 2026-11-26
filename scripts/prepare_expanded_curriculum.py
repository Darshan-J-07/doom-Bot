"""Create an opt-in train-set expansion without modifying existing maps."""
import argparse
import json
from pathlib import Path
import shutil

from training.generate_maps import create_map
from training.maps import load_verified_maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    source = Path("training/maps").resolve()
    if args.out.resolve().is_relative_to(source):
        parser.error("Use a new output directory outside the original curriculum.")
    rows = load_verified_maps(source)
    args.out.mkdir(parents=True, exist_ok=False)
    for row in rows:
        previous = Path(row["path"]).resolve()
        target = args.out / previous.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(previous, target)
        row["path"] = str(target)
    folder = args.out / "train" / "gauntlet"
    for index in range(6):
        row = create_map(folder / f"two-lock-{index:02}.wad", "gauntlet", 310000 + index,
                         two_locks=True)
        rows.append(dict(row, split="train", layout="two-lock"))
    (args.out / "manifest.json").write_text(json.dumps(dict(maps=rows), indent=2) + "\n")
    load_verified_maps(args.out)


if __name__ == "__main__":
    main()
