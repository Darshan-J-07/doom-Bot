"""Generate disjoint exit-recovery practice and assessment layouts."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import dist
from pathlib import Path
import random

from doom_env.mapinfo import parse_udmf
from doom_env.wad import Lump, read_wad, write_wad
from training.exit_switch_maps import (
    PRIOR_MAPS, add_annexes, fixtures, historical_geometry, point, properties,
    twist_vector, wall_segments, write_map,
)
from training.maps import load_verified_maps
from training.universe import layout_digest, orient, rectangle, verify_layout


BASES = {
    "train": ("align-pockets", "cross-recess", "recover-fork", "after-lock-terrace"),
    "validation": ("align-notches", "cross-overlap", "recover-loopway", "after-lock-elbow"),
    "test": ("align-balcony", "cross-tiered", "recover-alternating", "after-lock-chambers"),
}
FULL_FAMILIES = {"keycard": ("keycard-overpass", "keycard-loopback"),
                 "gauntlet": ("gauntlet-guardhouse", "gauntlet-crossfire")}
STAGE_NAMES = ("align-and-interact", "cross-the-room", "recover-the-route", "finish-after-unlocking")
ASSESSMENT_INPUTS = (
    "runs/exit-switch-study/maps/manifest.json", "runs/exit-switch-study/maps/final.json",
    "runs/exit-switch-study/maps/fixtures.json", "runs/limit-campaign-20261127/maps/manifest.json",
)


def room(rng, split, *, near):
    width, height = rng.randint(4, 6), rng.randint(3, 5)
    row = rng.randint(1, height - 2)
    cells = rectangle(0, 0, width, height)
    start = (width - 2, row) if near else (rng.randrange(0, 2), row)
    goal = (width - 1, row)
    if near:
        tile = rng.choice((36, 40, 44)), rng.choice((76, 88, 104))
        offset = rng.randint(9, 12), rng.randint(-16, 16)
        heading = rng.choice((-150, -105, -55, -20, 20, 55, 105, 150)) + rng.randint(-15, 15)
        if split == "train":
            depth = rng.randint(1, 3)
            cells |= rectangle(0, -depth, rng.randint(1, width - 2), depth)
        elif split == "validation":
            cells -= rectangle(0, height - 1, rng.randint(1, width - 2), 1)
            cells |= {(0, -1), (0, -2)}
        else:
            cells |= rectangle(1, height, width - 2, 2) | {(width - 2, height + 2)}
            cells.remove((0, 0))
        ticks = 280
    else:
        tile = rng.choice((52, 68, 84, 96)), rng.choice((76, 92, 108))
        offset = rng.randint(-12, 12), rng.randint(-15, 15)
        heading = rng.randint(-140, 140)
        if split == "train":
            cells.remove((rng.randint(1, width - 2), (row + 1) % height))
            cells |= rectangle(0, -1, rng.randint(2, width - 1), 1)
        elif split == "validation":
            cells = rectangle(0, row - 2, width - 1, 3) | rectangle(width - 2, row, 2, height)
        else:
            cells |= rectangle(0, -2, width, 2) | rectangle(1, height, width - 2, 1)
            cells -= {(width - 1, -2), (0, -2)}
        ticks = 520
    cells = add_annexes(cells, rng, [(width, row)])
    return cells, start, goal, (1, 0), tile, heading, offset, ticks


def recovery_room(rng, split):
    width = rng.randint(5, 8)
    left, right = rng.randint(2, 4), rng.randint(2, 4)
    cells = rectangle(0, 0, width, 1) | rectangle(0, 1, 1, left) | rectangle(width - 1, 1, 1, right)
    start, goal = (0, left), (width - 1, right)
    if split == "train":
        depth = rng.randint(1, 3)
        cells |= rectangle(rng.randint(1, width - 2), -depth, 1, depth)
    elif split == "validation":
        cells |= rectangle(1, -2, width - 2, 1) | {(1, -1), (width - 2, -1)}
    else:
        cells |= rectangle(1, -3, 1, 3) | rectangle(width - 2, -2, 1, 2)
        cells |= {(width - 1, -2), (width, -2)}
    # Keep the initial leg and the approach leg separated so the only route
    # out of the initial dead end first retreats down its corridor.
    protected = {(1, y) for y in range(1, left + 2)} | {(width - 1, right + 1)}
    protected |= {(0, left + 1), (-1, left)}
    cells = add_annexes(cells, rng, protected)
    tile = rng.choice((68, 80, 88)), rng.choice((68, 84, 96))
    return cells, start, goal, (0, 1), tile, rng.randint(-75, 165), (0, rng.randint(-10, 10)), 820


def portal_room(rng, split, task):
    gate_x = rng.randint(3, 5)
    right = rng.randint(3, 5)
    gate, last = (gate_x, 0), gate_x + right
    cells = rectangle(0, -1, gate_x, 3) | rectangle(gate_x + 1, -1, right, 3) | {gate}
    goal = (last, rng.choice((-1, 0, 1)))
    start = (rng.randint(0, 1), rng.choice((-1, 0, 1)))
    if task == "conditional":
        if split == "train":
            cells |= rectangle(0, -3, rng.randint(1, gate_x - 1), 2)
        elif split == "validation":
            cells |= rectangle(last - 1, 2, 2, rng.randint(2, 3))
            cells.remove((gate_x + 1, -1))
        else:
            cells |= rectangle(0, 2, gate_x - 1, 2) | rectangle(last - 1, -3, 2, 2)
        return cells, start, goal, (1, 0), rng.choice((68, 80, 92)), gate

    # The key recess is reachable only before the central, initially locked
    # door. Each final family adds a different kind of detour or combat room.
    key = (rng.randint(0, gate_x - 2), -3)
    cells |= {key, (key[0], -2)}
    return cells, start, goal, (1, 0), 88, gate, key


def geometry(rng, split, level, family, task):
    keys_and_door = []
    open_door = None
    items = {}
    if level < 2:
        cells, start, goal, normal, tile, heading, offset, ticks = room(rng, split, near=level == 0)
    elif level == 2:
        cells, start, goal, normal, tile, heading, offset, ticks = recovery_room(rng, split)
    else:
        portal = portal_room(rng, split, task)
        cells, start, goal, normal, width, gate, *key_data = portal
        tile = (width, rng.choice((76, 88, 96)))
        heading = rng.randint(-95, 95)
        offset = (rng.randint(-12, 12), rng.randint(-12, 12))
        protected = [(gate[0], y) for y in range(-8, 9) if y != 0] + [point(goal, normal)]
        if task == "conditional":
            open_door = gate
            items[gate] = "D"
            ticks = 760
        else:
            key, = key_data
            color = rng.choice("rby")
            keys_and_door = [key, gate]
            items.update({key: color, gate: color.upper()})
            if family in ("keycard-overpass", "gauntlet-crossfire"):
                cells |= rectangle(0, 2, gate[0] - 1, 2)
            else:
                cells |= rectangle(goal[0] - 1, 2, 2, 2)
            if task == "gauntlet":
                # Off-route guard recesses preserve a real combat threat while
                # leaving an unobstructed route for the diagnostic walker.
                first_guard, second_guard = (gate[0] - 1, -2), (gate[0] + 2, 2)
                cells |= {first_guard, second_guard}
                items.update({first_guard: "z", second_guard: "i"})
            ticks = 1250
        cells = add_annexes(cells, rng, protected)

    layout = dict.fromkeys(cells, ".")
    layout.update(items)
    layout.update({start: "P", goal: "E"})
    goals = [*keys_and_door, goal]
    labels = goals + ([open_door] if open_door is not None else [])
    ordered_glyphs = [layout[cell] for cell in labels]
    for index, cell in enumerate(labels):
        layout[cell] = str(index)
    turns, mirror = rng.randrange(4), bool(rng.randrange(2))
    layout = orient(layout, turns, mirror)
    destinations = [None] * len(labels)
    for cell, glyph in layout.items():
        if glyph.isdigit():
            destinations[int(glyph)] = cell
            layout[cell] = ordered_glyphs[int(glyph)]
    heading = (90 * turns + (-heading if mirror else heading)) % 360
    tile = tile[::-1] if turns % 2 else tile
    plan = (layout, destinations[:len(goals)], twist_vector(normal, turns, mirror), tile, heading,
            twist_vector(offset, turns, mirror), ticks)
    return plan, destinations[-1] if open_door is not None else None


def prior_identities():
    known = historical_geometry()
    hashes = set()
    receipts = {}
    for name in (*PRIOR_MAPS, *ASSESSMENT_INPUTS):
        path = Path(name)
        if not path.is_file():
            if name in ASSESSMENT_INPUTS:
                raise FileNotFoundError(f"Missing required prior map manifest: {name}")
            continue
        contents = path.read_bytes()
        receipts[name] = sha256(contents).hexdigest()
        for row in json.loads(contents)["maps"]:
            if row.get("geometry_sha256"):
                known.add(row["geometry_sha256"])
            if row.get("sha256"):
                hashes.add(row["sha256"])
            if row.get("drill", {}).get("cells"):
                known.add(layout_digest({(x, y): glyph for x, y, glyph in row["drill"]["cells"]}, labeled=False))
    return known, hashes, receipts


def preopen(row, cell):
    path = Path(row["path"])
    lumps = read_wad(path)
    blocks = parse_udmf(next(lump.data.decode() for lump in lumps if lump.name == "TEXTMAP"))
    ordered = sorted((x, y) for x, y, _ in row["drill"]["cells"])
    sector = blocks["sector"][ordered.index(cell)]
    if sector["heightceiling"] != sector["heightfloor"]:
        raise ValueError("The authored door must start as a standard door sector before preopening")
    # A fully raised Doom door stops four units below its neighboring ceiling.
    sector["heightceiling"] = 124
    text = ['namespace = "zdoom";']
    text.extend(f"{kind} {{ {properties(item)} }}" for kind, items in blocks.items() for item in items)
    write_wad(path, [Lump("TEXTMAP", "\n".join(text).encode()) if lump.name == "TEXTMAP" else lump for lump in lumps])
    row["sha256"] = sha256(path.read_bytes()).hexdigest()
    # Training diagnostics model the pre-open tile as traversable. The actual
    # engine still identifies the scene as a standard unlocked door.
    row["drill"]["cells"] = [[x, y, "." if (x, y) == cell else glyph]
                              for x, y, glyph in row["drill"]["cells"]]
    row.update(post_lock=True, open_door_cell=list(cell))


def distance_to_wall(position, segment):
    a, b = segment
    edge = tuple(end - start for start, end in zip(a, b))
    norm = sum(value * value for value in edge)
    u = sum((value - start) * step for value, start, step in zip(position, a, edge)) / norm
    closest = tuple(start + max(0, min(1, u)) * step for start, step in zip(a, edge))
    return dist(position, closest)


def generated(out, seed, split, family, level, count, task, used, old_hashes):
    folder = out / split / family
    folder.mkdir(parents=True)
    rng = random.Random(f"{seed}:{split}:{family}")
    for index in range(count):
        mode = "walkover" if task == "conditional" and index % 4 == 3 else "switch"
        for _ in range(1000):
            plan, open_door = geometry(rng, split, level, family.removeprefix("recovery-"), task)
            layout, targets, normal, tile, _, offset, _ = plan
            shape = layout_digest(layout, labeled=False)
            if shape in used:
                continue
            player = next(point for point, glyph in layout.items() if glyph == "P")
            position = tuple(int((value + .5) * side + delta) for value, side, delta in zip(player, tile, offset))
            wall = next(seg for cell, facing, seg in wall_segments(layout, tile)
                        if cell == targets[-1] and facing == normal)
            initial_range = distance_to_wall(position, wall)
            if (level == 0) != (initial_range <= 64):
                continue
            try:
                verify_layout(layout, targets, mode == "switch")
            except ValueError:
                continue
            path = folder / f"{index:03d}.wad"
            row = write_map(path, split, family, f"{family}-{index:03d}", plan, mode, rng, task=task)
            if open_door is not None:
                preopen(row, open_door)
            if row["sha256"] in old_hashes:
                raise ValueError(f"Previously used WAD bytes at {path}")
            used.add(shape)
            old_hashes.add(row["sha256"])
            row.update(level=level, curriculum_stage=STAGE_NAMES[level], expected_start=list(position),
                       initial_switch_distance=initial_range if mode == "switch" else None,
                       starts_in_use_range=bool(mode == "switch" and initial_range <= 64))
            yield row
            break
        else:
            raise ValueError(f"Unable to create independent map {split}/{family}/{index}")


def build(out, seed):
    out.mkdir(parents=True, exist_ok=False)
    used, old_hashes, parents = prior_identities()
    def save(name, rows):
        data = dict(seed=seed, generator="training.exit_recovery_maps", prior_manifest_sha256=parents, maps=rows)
        (out / name).write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")

    save("fixtures.json", list(fixtures(out, seed)))
    by_split = {split: [row for level, family in enumerate(families)
                         for row in generated(out, seed, split, f"recovery-{family}", level,
                                              24 if split == "train" else 8, "conditional", used, old_hashes)]
                for split, families in BASES.items()}
    for task, families in FULL_FAMILIES.items():
        for family in families:
            by_split["test"].extend(generated(out, seed, "test", f"recovery-{family}", 3, 4, task, used, old_hashes))
    save("manifest.json", by_split["train"] + by_split["validation"])
    save("final.json", by_split["test"])
    load_verified_maps(out)
    print(json.dumps({split: dict(Counter(f"{row['task']}/{row['exit_type']}" for row in rows))
                      for split, rows in by_split.items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=743911)
    args = parser.parse_args()
    build(args.out, args.seed)
