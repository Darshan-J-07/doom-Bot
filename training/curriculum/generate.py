"""Generate a separate, deterministic UDMF curriculum for the official wrapper."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import random

from doom_env.mapinfo import LOCK_BLUE, LOCK_RED, LOCK_YELLOW
from training.generate_maps import STAGES, write_grid_wad
from training.maps import load_verified_maps


# Each layout has a name, starting angle, and grid. Rows run north to south.
# S/G = start/exit, D = ordinary door, R/B/Y and r/b/y = locks and their keys.
# e/i = zombieman/imp, h/a = stimpack/ammo. A space has no floor.
LAYOUTS = {
    "train": {
        "movement": (
            ("first-steps", 0, ("S.G",)),
            ("long-hall", 0, ("S...G",)),
            ("first-turn", 0, ("S..", "  .", "  G")),
        ),
        "combat": (
            ("single-target", 0, ("Se.G",)),
            ("target-arena", 0, (".....", "S.e.G", ".....")),
            ("target-cover", 0, (".. ..", "S.e.G", ".. ..")),
        ),
        "exploration": (
            ("first-door", 0, ("S.D.G", " .", "..")),
            ("door-around-corner", 0, ("S..", "  .", "  D", "  ..G", "   .")),
            ("door-with-side-rooms", 0, ("  ..", "S...D..G", "  .", "  ..")),
        ),
        "keys": (
            ("key-before-lock", 0, ("SrR.G",)),
            ("blue-key-detour", 0, ("S..B..G", " .", " b")),
            ("yellow-key-detour", 0, ("  y", "  .", "S...Y..G")),
        ),
        "gauntlet": (
            ("two-lock-practice", 0, ("SrR.bB.G",)),
            ("two-lock-skirmish", 0, ("S..R..eB..G", " r   b", " .   h")),
            ("full-route", 0, ("S...B.e.R..G", "  b   r  i", "  .   a  h")),
        ),
    },
    "validation": {
        "movement": (("bent-hall", 0, ("S.", " .", " ..G")),),
        "combat": (("offset-target", 0, (" e.G", "S..", " ...")),),
        "exploration": (("west-door", 180, ("..S", "D", "..", " ...G", " ...")),),
        "keys": (("junction-red-key", 90, ("..r", ".", "...R..G", "  .", "  S")),),
        "gauntlet": (
            ("guarded-bend", 0, ("..b", ".", "S..B..e", "      .", "      r", "      R", "      ..G")),
        ),
    },
    "test": {
        "movement": (("loop-and-turn", 0, ("S..", ". .", "...", " .", " G")),),
        "combat": (("flank-target", 0, ("..e.", "S..G", "..")),),
        "exploration": (
            ("loop-past-door", 0, ("S..  ..G", "  .  .", "  D  .", "  ....", "  ..")),
        ),
        "keys": (
            ("blue-key-return", 270, ("   b", " ...", " S", " ..B..", "     .", "     G")),
        ),
        "gauntlet": (
            ("lock-and-ambush", 0, ("r..", "  .", "S....R..e", "   h    .", "   .    .",
                                   "        y", "        Y", "        .iG")),
        ),
    },
}
VARIANTS = {"train": 4, "validation": 2, "test": 2}
DOORS = {"D": 0, "R": LOCK_RED, "B": LOCK_BLUE, "Y": LOCK_YELLOW}
KEYS = {"r": 13, "b": 5, "y": 6}
THINGS = {**KEYS, "S": 1, "e": 3004, "i": 3001, "h": 2011, "a": 2007}
NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def grid(rows):
    return {(x, -y): symbol for y, row in enumerate(rows) for x, symbol in enumerate(row)
            if symbol != " "}


def rotate(cell, turns):
    x, y = cell
    return ((x, y), (-y, x), (-x, -y), (y, -x))[turns]


def build_layout(name, seed):
    heading, rows = next((angle, rows) for stages in LAYOUTS.values()
                         for layouts in stages.values() for label, angle, rows in layouts
                         if label == name)
    rng = random.Random(seed)
    turns = rng.randrange(4)
    tiles = {rotate(cell, turns): symbol for cell, symbol in grid(rows).items()}
    return tiles, rng.choice((128, 144, 160)), (heading + 90 * turns) % 360


def neighbors(cell):
    x, y = cell
    return ((x + dx, y + dy) for dx, dy in NEIGHBORS)


def exit_reachable(tiles, start, goal, *, blocked=None):
    queue = deque([(start, frozenset())])
    seen = set(queue)
    while queue:
        cell, keys = queue.popleft()
        if cell == goal:
            return True
        for candidate in neighbors(cell):
            symbol = tiles.get(candidate)
            if symbol is None or candidate == blocked:
                continue
            if symbol in "RBY" and symbol.lower() not in keys:
                continue
            state = (candidate, keys | {symbol} if symbol in KEYS else keys)
            if state not in seen:
                seen.add(state)
                queue.append(state)
    return False


def check_layout(name, tiles):
    values = list(tiles.values())
    if values.count("S") != 1 or values.count("G") != 1 or set(values) - DOORS.keys() - THINGS.keys() - {".", "G"}:
        raise ValueError(f"{name} needs one start, one exit, and valid symbols")
    start = next(cell for cell, symbol in tiles.items() if symbol == "S")
    goal = next(cell for cell, symbol in tiles.items() if symbol == "G")
    if not exit_reachable(tiles, start, goal):
        raise ValueError(f"{name} has no route that obtains the required keys")
    for (x, y), symbol in tiles.items():
        if symbol not in DOORS:
            continue
        connected = {cell for cell in neighbors((x, y)) if cell in tiles}
        if connected not in ({(x - 1, y), (x + 1, y)}, {(x, y - 1), (x, y + 1)}):
            raise ValueError(f"{name} has a door outside a straight corridor")
        if exit_reachable(tiles, start, goal, blocked=(x, y)):
            raise ValueError(f"{name} has a route that bypasses a required door")
    return start, goal


def topology(tiles, start):
    """Canonicalize geometry under translation, rotation and reflection."""
    x0, y0 = start
    return min(tuple(sorted(
        (*rotate(((x - x0) * mirror, y - y0), turns), symbol if symbol in "SGDRBYrby" else ".")
        for (x, y), symbol in tiles.items()
    )) for turns in range(4) for mirror in (-1, 1))


def generate(root):
    root = Path(root)
    if any((root / part).exists() for part in (*LAYOUTS, "manifest.json")):
        raise FileExistsError(f"{root} already contains a curriculum. Choose a fresh output.")
    authored = set()
    for stages in LAYOUTS.values():
        for layouts in stages.values():
            for name, _, rows in layouts:
                tiles = grid(rows)
                start, _ = check_layout(name, tiles)
                shape = topology(tiles, start)
                if shape in authored:
                    raise ValueError(f"{name} reuses another authored layout")
                authored.add(shape)

    entries = []
    for split_index, (split, stages) in enumerate(LAYOUTS.items()):
        for stage_index, stage in enumerate(STAGES):
            folder = root / split / stage
            folder.mkdir(parents=True, exist_ok=False)
            for layout_index, (name, _, _) in enumerate(stages[stage]):
                for variant in range(VARIANTS[split]):
                    seed = 720000 + 10000 * split_index + 1000 * stage_index + 100 * layout_index + variant
                    tiles, tile, heading = build_layout(name, seed)
                    _, goal = check_layout(name, tiles)
                    locks = {cell: DOORS[symbol] for cell, symbol in tiles.items() if symbol in DOORS}

                    def populate(thing):
                        for cell, symbol in sorted(tiles.items()):
                            if symbol in THINGS:
                                thing(cell, THINGS[symbol], angle=heading if symbol == "S" else None)

                    path = folder / f"{name}-{variant:02}.wad"
                    write_grid_wad(path, tiles.keys(), goal, locks, tile, random.Random(seed), populate)
                    entries.append(dict(path=path.as_posix(), split=split, stage=stage, layout=name,
                                        seed=seed, map_name="MAP01", cells=len(tiles),
                                        sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest = dict(source="Self-authored UDMF grid layouts", license="CC0-1.0 for map geometry",
                    maps=entries)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return load_verified_maps(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(f"Verified {len(generate(args.out))} generated map files in {args.out}.")
