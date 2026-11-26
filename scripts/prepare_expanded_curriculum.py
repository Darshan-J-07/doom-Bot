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
    parser.add_argument(
        "--replace-gauntlets", type=int, metavar="COUNT",
        help="replace this many existing training gauntlets to keep tier sampling matched",
    )
    args = parser.parse_args()
    source = Path("training/maps").resolve()
    if args.out.resolve().is_relative_to(source):
        parser.error("Use a new output directory outside the original curriculum.")
    rows = load_verified_maps(source)
    replace = args.replace_gauntlets
    slots = [i for i, row in enumerate(rows)
             if row["split"] == "train" and row["stage"] == "gauntlet"]
    if replace is not None and not 1 <= replace <= min(6, len(slots)):
        parser.error("--replace-gauntlets must be in the range 1 through 6")

    args.out.mkdir(parents=True, exist_ok=False)
    replaced_slots = set(slots[:replace]) if replace else set()
    for i, row in enumerate(rows):
        if i in replaced_slots:
            continue
        previous = Path(row["path"]).resolve()
        target = args.out / previous.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(previous, target)
        row["path"] = str(target)
    folder = args.out / "train" / "gauntlet"
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(6 if replace is None else replace):
        row = create_map(folder / f"two-lock-{index:02}.wad", "gauntlet", 310000 + index,
                         two_locks=True)
        row = dict(row, split="train", layout="two-lock")
        if replace is None:
            rows.append(row)
        else:
            rows[slots[index]] = row
    (args.out / "manifest.json").write_text(json.dumps(dict(maps=rows), indent=2) + "\n")
    load_verified_maps(args.out)


if __name__ == "__main__":
    main()
