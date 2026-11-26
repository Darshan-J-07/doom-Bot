"""Generate training drills and a separate, fresh test split."""
from __future__ import annotations

import argparse
from collections import deque
from hashlib import sha256
import json
from pathlib import Path
import random

from doom_env.wad import write_wad
from examination.maps import CHALLENGES, Challenge, build_lumps, verified_maps
from training.generate_maps import STAGES, create_map
from training.maps import load_verified_maps
from training.skills import NEIGHBORS

FAMILIES = {
    "corridor": ("movement", 10),
    "bends": ("movement", 10),
    "obstacle": ("movement", 8),
    "shoot": ("combat", 12),
    "fight-and-exit": ("combat", 10),
    "doors": ("exploration", 8),
    "loop": ("exploration", 8),
    "health": ("exploration", 6),
    "ammo": ("exploration", 6),
    "key": ("keys", 8),
    "two-locks": ("keys", 8),
    "fight-and-unlock": ("gauntlet", 8),
}


def grid_text(cells):
    horizontal = range(min(x for x, _ in cells), max(x for x, _ in cells) + 1)
    vertical = range(max(y for _, y in cells), min(y for _, y in cells) - 1, -1)
    return "\n".join("".join(cells.get((x, y), "#") for x in horizontal) for y in vertical)


def drill(family, rng):
    length = rng.randint(5, 9)
    cells = {(x, 0): "." for x in range(length)}
    start, exit = (0, 0), (length - 1, 0)
    heading = rng.choice((0, 0, 0, 20, 340, 180))
    health, bullets, ticks = 100, 50, 840
    required = {}

    if family == "corridor":
        for y in rng.choice(((), (-1,), (1,), (-1, 1))):
            cells.update({(x, y): "." for x in range(length - 1)})
    elif family == "bends":
        side = rng.choice((-1, 1))
        turn = rng.randint(2, length - 2)
        height = rng.randint(2, 4)
        cells = {(x, 0): "." for x in range(turn + rng.choice((1, 2)))}
        cells.update({(turn, i * side): "." for i in range(1, height + 1)})
        cells.update({(x, height * side): "." for x in range(turn, length + 1)})
        exit = (length, height * side)
    elif family == "obstacle":
        cells.update({(x, y): "." for x in range(1, length - 1) for y in (-1, 1)})
        for x in rng.sample(range(2, length - 2), rng.randint(1, min(2, length - 4))):
            cells.pop((x, 0))
    elif family in ("shoot", "fight-and-exit"):
        cells.update({(x, y): "." for x in range(length) for y in (-1, 1)})
        opponent = (rng.randint(2, min(4, length - 2)), rng.choice((-1, 0, 1)))
        cells[opponent] = rng.choice(("z", "z", "i"))
        heading = rng.choice((0, 15, 345))
        required["kills"] = 1
        if family == "fight-and-exit" and length >= 7 and rng.choice((False, True)):
            cells[length - 2, rng.choice((-1, 1))] = "z"
            required["kills"] = 2
        ticks = 420 if family == "shoot" else 1050
        if family == "fight-and-exit":
            bullets = rng.choice((12, 16, 50, 50))
    elif family == "doors":
        door_columns = [2] if length < 7 else [2, length - 3]
        cells.update({(x, y): "." for x in range(length) if x not in door_columns for y in (-1, 1)})
        for x in door_columns:
            cells[x, 0] = "D"
        required["doors"] = len(door_columns)
        ticks = 1050
    elif family == "loop":
        width, height = rng.randint(4, 7), rng.randint(2, 3)
        cells = {(x, y): "." for x in range(1, width + 1) for y in range(-height, height + 1)
                 if x in (1, width) or abs(y) == height}
        exit = (width + 1, 0)
        route_side = rng.choice((-1, 1))
        cells[1, route_side * height] = cells[width, route_side * height] = "*"
        required["checkpoints"] = 2
        ticks = 1260
    elif family in ("health", "ammo", "key", "two-locks", "fight-and-unlock"):
        path_length = rng.randint(9, 12) if family in ("two-locks", "fight-and-unlock") else max(6, length)
        cells = {(x, 0): "." for x in range(path_length)}
        exit = (path_length - 1, 0)
        branch, side, depth = rng.randint(1, 2), rng.choice((-1, 1)), rng.randint(1, 3)
        cells.update({(branch, y * side): "." for y in range(1, depth + 1)})
        pickup = branch, side * depth
        if family == "health":
            cells[pickup] = "h"
            health = rng.choice((30, 45, 60, 75))
            required["health_gained"] = 10
        elif family == "ammo":
            cells[pickup] = "a"
            bullets = rng.choice((4, 8, 12))
            required["ammo_gained"] = 5
        else:
            cells[pickup] = "r"
            first_door = min(branch + 2, path_length - 3)
            cells[first_door, 0] = "R"
            required.update(keycards=1, locks=1)
            if family in ("two-locks", "fight-and-unlock"):
                second_branch = first_door + 1
                cells[second_branch, -side] = "b"
                cells[path_length - 3, 0] = "B"
                required.update(keycards=2, locks=2)
            if family == "fight-and-unlock":
                cells[first_door + 1, 0] = rng.choice(("z", "i"))
                cells[first_door + 1, side] = "h"
                required["kills"] = 1
        ticks = 1400

    cells[start], cells[exit] = "P", "E"
    # Reflections change the direction of turns relative to the spawn.
    if rng.choice((False, True)):
        cells = {(x, -y): glyph for (x, y), glyph in cells.items()}
        heading = -heading % 360
    return Challenge(family, family, family, grid_text(cells), required, ticks, heading, health, bullets)


def verify_layout(design):
    """Check reachability with keys and a separate exit boundary before writing a WAD."""
    cells = design.cells()
    start = next(cell for cell, glyph in cells.items() if glyph == "P")
    goal = next(cell for cell, glyph in cells.items() if glyph == "E")
    if all(cells.get((goal[0] + dx, goal[1] + dy), "#") in "#DRBY" for dx, dy in NEIGHBORS):
        raise ValueError(f"Missing walkover exit boundary in {design.slug}")
    queue = deque([(start, frozenset())])
    seen = set(queue)
    reached, reachable = False, set()
    while queue:
        (x, y), keys = queue.popleft()
        reachable.add((x, y))
        if (x, y) == goal:
            reached |= len(keys) >= design.minimums.get("keycards", 0)
            continue
        for dx, dy in NEIGHBORS:
            next_cell = (x + dx, y + dy)
            glyph = cells.get(next_cell)
            if glyph is None or (glyph in "RBY" and glyph.lower() not in keys):
                continue
            next_state = (next_cell, keys | {glyph} if glyph in "rby" else keys)
            if next_state not in seen:
                seen.add(next_state)
                queue.append(next_state)
    if not reached or reachable != cells.keys():
        raise ValueError(f"Unreachable goal or tile in {design.slug}")


def build_training(folder, seed):
    folder.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    rows = []
    seen = {row["sha256"] for row in verified_maps()}
    diagnostic_layouts = {tuple(sorted(challenge.cells().items())) for challenge in CHALLENGES}
    for family, (stage, limit) in FAMILIES.items():
        for index in range(limit):
            path = folder / f"{family}-{index:02d}.wad"
            for _ in range(500):
                design = drill(family, rng)
                if tuple(sorted(design.cells().items())) in diagnostic_layouts:
                    continue
                verify_layout(design)
                write_wad(path, build_lumps(design))
                digest = sha256(path.read_bytes()).hexdigest()
                if digest not in seen:
                    break
            else:
                raise ValueError(f"Could not generate a distinct {family} layout")
            seen.add(digest)
            rows.append(dict(path=path.as_posix(), map_name="MAP01", stage=stage, split="train",
                             sha256=digest, ticks=design.ticks, family=family,
                             drill=dict(cells=[[*cell, glyph] for cell, glyph in design.cells().items()],
                                        minimums=design.minimums, kill_lesson=family == "shoot")))
    manifest = {"maps": rows, "source": "Original skill drill layouts", "seed": seed}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    load_verified_maps(folder)
    return rows


def build_final(folder, seed):
    folder.mkdir(parents=True, exist_ok=False)
    rows = []
    for group, stage in enumerate(STAGES):
        for index in range(5):
            row = create_map(folder / f"{stage}-{index:02d}.wad", stage, seed + 100 * group + index)
            rows.append(dict(row, split="test"))
    manifest = {"maps": rows, "seeds": list(range(seed + 10000, seed + 10003)),
                "purpose": "Untouched final comparison after candidate selection"}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    load_verified_maps(folder)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=791037)
    parser.add_argument("--final-out", type=Path)
    args = parser.parse_args()
    training = build_training(args.out, args.seed)
    print(f"Prepared {len(training)} training drills in {args.out}")
    if args.final_out:
        final = build_final(args.final_out, args.seed + 9100)
        if {r["sha256"] for r in training} & {r["sha256"] for r in final}:
            raise ValueError("Training and final WADs overlap")
        print(f"Prepared {len(final)} untouched final maps in {args.final_out}")


if __name__ == "__main__":
    main()
