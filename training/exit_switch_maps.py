"""Build disjoint standard-rule switch drills and diagnostic ViZDoom fixtures."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random

from doom_env.mapinfo import parse_udmf
from doom_env.wad import Lump, write_wad
from examination.maps import CHALLENGES, Challenge, build_lumps
from training.build_skill_curriculum import grid_text
from training.maps import load_verified_maps
from training.universe import layout_digest, orient, rectangle, verify_layout


TRAIN = {"near-bays": 36, "annex-halls": 24, "cross-gallery": 24, "turn-bays": 12}
VALIDATION = {"dogleg-landing": 4, "offset-chambers": 4, "open-frontage": 4}
FINAL = {"side-branches": 4, "wide-return": 4, "corner-annex": 4, "cross-landing": 4,
         "card-vestibule": 4, "guarded-vestibule": 4}

WALL_TEXTURES = ("STARTAN2", "STONE2", "BRICK7", "BROWN1", "TEKWALL4", "METAL1", "GRAY5")
EXIT_TEXTURES = ("SW1EXIT", "SW2EXIT")
EXIT_SIGNS = ("EXITSIGN", "EXITSGN2", "EXITSTON")
DISTRACTOR_TEXTURES = ("SW1COMP", "SW1BRCOM", "SW1BRN1", "SW1WOOD", "SW1GARG")
SIDES = (((-1, 0), ((0, 0), (0, 1))), ((0, 1), ((0, 1), (1, 1))),
         ((1, 0), ((1, 1), (1, 0))), ((0, -1), ((1, 0), (0, 0))))
PRIOR_MAPS = (
    "runs/skill-campaign/maps/manifest.json", "runs/skill-campaign/maps/final.json",
    "runs/skill-campaign/maps/probe.json", "runs/universal/maps/manifest.json",
    "runs/paper-drills/verified-maps/manifest.json", "training/maps/manifest.json",
)


def point(cell, offset):
    return tuple(c + delta for c, delta in zip(cell, offset))


def twist_vector(vector, turns, mirror):
    x, y = vector
    if mirror:
        y = -y
    for _ in range(turns):
        x, y = -y, x
    return x, y


def wall_segments(cells, tile):
    for x, y in sorted(cells):
        for normal, endpoints in SIDES:
            if point((x, y), normal) not in cells:
                segment = tuple(((x + dx) * tile[0], (y + dy) * tile[1]) for dx, dy in endpoints)
                yield (x, y), normal, segment


def add_annexes(cells, rng, protected):
    result = set(cells)
    choices = sorted({point(cell, direction) for cell in result for direction, _ in SIDES}
                     - result - set(protected))
    result.update(rng.sample(choices, rng.randint(1, min(len(choices), 5))))
    return result


def conditional_geometry(family, rng):
    width = rng.randint(4, 6)
    height = rng.randint(2, 4)
    row = rng.randrange(height)
    east = (1, 0)
    tiles = (rng.choice((72, 80, 96, 112)), rng.choice((72, 88, 104, 120)))
    heading = rng.randrange(360)
    offset = (rng.randint(-12, 12), rng.randint(-12, 12))
    start, exit_cell = (0, row), (width - 1, row)
    if family == "near-bays":
        cells = rectangle(0, 0, width, height)
        start = width - 2, row
        tiles = rng.choice((40, 44, 48)), rng.choice((88, 96, 112))
        heading = rng.randint(-45, 45)
        offset = rng.randint(12, 16), rng.randint(-16, 16)
        ticks = 240
    elif family == "annex-halls":
        cells = rectangle(0, row, width, 1) | rectangle(1, row + 1, rng.randint(1, width - 2), height)
        heading = rng.randint(-100, 100)
        ticks = 352
    elif family == "cross-gallery":
        cells = rectangle(0, 0, width, height) | {(1, -1), (1, -2), (width - 2, height), (width - 2, height + 1)}
        if rng.random() < .5:
            exit_cell, east = (width - 2, height + 1), (0, 1)
        ticks = 448
    elif family in ("turn-bays", "dogleg-landing"):
        cells = rectangle(0, 0, width, 1) | rectangle(width - 1, 1, 1, height)
        start, exit_cell, east = (0, 0), (width - 1, height), (0, 1)
        if family == "dogleg-landing":
            cells |= rectangle(width - 2, height, 2, 2)
            exit_cell = width - 1, height + 1
        ticks = 512
    elif family == "offset-chambers":
        cells = rectangle(0, 0, 3, 3) | {(3, 2), (4, 2)} | rectangle(4, 2, width - 1, height)
        start, exit_cell = (1, 2), (width + 2, height + 1)
        ticks = 576
    elif family == "open-frontage":
        cells = rectangle(0, 0, width, height + 2) | rectangle(1, -2, width - 2, 2)
        start, exit_cell = (width - 2, row + 1), (width - 1, row + 1)
        heading = rng.randint(-100, 100)
        ticks = 384
    elif family == "side-branches":
        cells = rectangle(0, 0, width + 1, 1) | rectangle(1, 1, 1, height) | rectangle(width - 1, -height, 1, height)
        start, exit_cell, east = (0, 0), (width - 1, -height), (0, -1)
        ticks = 576
    elif family == "wide-return":
        cells = rectangle(0, 0, width, height + 3) - rectangle(2, 0, 1, height + 1)
        start, exit_cell, east = (0, 0), (width - 1, 0), (0, -1)
        ticks = 608
    elif family == "corner-annex":
        cells = rectangle(0, 0, width, height + 1) | rectangle(width - 1, height + 1, 2, 2)
        start, exit_cell, east = (1, row), (width, height + 2), (0, 1)
        ticks = 512
    elif family == "cross-landing":
        cells = rectangle(0, row, width, 2) | rectangle(1, row - 2, 2, height + 4)
        start, exit_cell, east = (1, row - 1), (2, row + height + 1), (0, 1)
        ticks = 512
    else:
        raise ValueError(f"Unknown conditional family {family}")
    cells = add_annexes(cells, rng, [point(exit_cell, east)])
    return dict.fromkeys(cells, "."), start, exit_cell, east, tiles, heading, offset, ticks


def full_geometry(family, rng):
    before = rng.randint(4, 6)
    after = rng.randint(3, 4)
    end = before + after
    center = rectangle(0, 0, end + 1, 1)
    first_side = rng.choice((-1, 1))
    before_bay = rectangle(1, min(0, first_side * 2), before - 2, 3)
    after_bay = rectangle(before + 1, -1, after, 3)
    key = rng.randint(1, before - 2), first_side * 2
    lock, exit_cell = (before, 0), (end, 0)
    core = center | before_bay | after_bay
    protected = {(before, y) for y in range(-5, 6)} | {(end + 1, 0)}
    cells = dict.fromkeys(add_annexes(core, rng, protected), ".")
    color = rng.choice("rby")
    cells[key], cells[lock] = color, color.upper()
    if family == "guarded-vestibule":
        cells[before - 1, 0] = "z"
        cells[end - 1, 1] = "i"
        cells[0, -1] = "h"
    return cells, (0, 0), exit_cell, (1, 0), (96, 96), rng.randint(-55, 55), (0, 0), 1050, [key, lock]


def properties(props):
    def value(raw):
        if isinstance(raw, str):
            return json.dumps(raw)
        if isinstance(raw, bool):
            return str(raw).lower()
        return str(raw)
    return " ".join(f"{name} = {value(raw)};" for name, raw in props.items())


def decorate(lumps, exit_segment, distractors, marking, rng):
    """Apply appearance and trigger edits to a map authored by the shared grid builder."""
    blocks = parse_udmf(next(lump.data.decode() for lump in lumps if lump.name == "TEXTMAP"))
    vertices, sectors = blocks["vertex"], blocks["sector"]
    sides = blocks["sidedef"]
    exit_texture, sign_texture = marking
    base = rng.choice(WALL_TEXTURES)
    light = rng.randrange(152, 233, 8)
    for i, sector in enumerate(sectors):
        sector["lightlevel"] = min(248, max(136, light + rng.choice((-16, -8, 0, 8, 16))))
        sector["id"] = i + 1
    walls = []
    for line in blocks["linedef"]:
        if "sideback" in line:
            if exit_segment is not None and line.get("special") == 243:
                del line["special"]
                del line["playercross"]
            continue
        wall = tuple((vertices[line[side]]["x"], vertices[line[side]]["y"]) for side in ("v1", "v2"))
        walls.append(wall)
        side = sides[line["sidefront"]]
        was_exit_sign = side["texturemiddle"] == "EXITSIGN"
        side["texturemiddle"] = (sign_texture if was_exit_sign and exit_segment is None
                                 else rng.choice(WALL_TEXTURES) if rng.random() < .24 else base)
        if wall == exit_segment:
            side["texturemiddle"] = exit_texture
            line.update(special=243, playeruse=True)
        elif wall in distractors:
            side["texturemiddle"] = rng.choice(DISTRACTOR_TEXTURES)
            # Light_ChangeToValue uses a sector tag and has no exit or progression effect.
            line.update(special=112, arg0=side["sector"] + 1, arg1=250, playeruse=True)
    if exit_segment is not None and exit_segment not in walls:
        raise ValueError(f"Selected exit is not a one-sided wall: {exit_segment}")
    if set(distractors) - set(walls):
        raise ValueError("A selected distractor is not a one-sided wall")
    # Mark a second exit-cell wall where one exists. Different signs can accompany
    # the same exit switch across maps without making an ordinary switch an exit.
    if exit_segment is not None:
        switch_line = next(line for line in blocks["linedef"] if line.get("playeruse") and line.get("special") == 243)
        exit_sector = sides[switch_line["sidefront"]]["sector"]
        for line in blocks["linedef"]:
            if ("sideback" not in line and not line.get("playeruse")
                    and sides[line["sidefront"]]["sector"] == exit_sector):
                sides[line["sidefront"]]["texturemiddle"] = sign_texture
                break
    text = ['namespace = "zdoom";']
    text.extend(f"{kind} {{ {properties(row)} }}" for kind, rows in blocks.items() for row in rows)
    return [Lump("TEXTMAP", "\n".join(text).encode()) if lump.name == "TEXTMAP" else lump for lump in lumps]


def plan(family, rng, turns, mirror):
    if family in ("card-vestibule", "guarded-vestibule"):
        cells, start, goal, normal, tile, heading, offset, ticks, targets = full_geometry(family, rng)
    else:
        cells, start, goal, normal, tile, heading, offset, ticks = conditional_geometry(family, rng)
        targets = []
    cells[start], cells[goal] = "P", "E"
    points = [*targets, goal]
    glyphs = [cells[cell] for cell in points]
    cells.update({cell: str(i) for i, cell in enumerate(points)})
    cells = orient(cells, turns, mirror)
    goal_order = [None] * len(points)
    for cell, glyph in cells.items():
        if glyph.isdigit():
            index = int(glyph)
            goal_order[index], cells[cell] = cell, glyphs[index]
    return (cells, goal_order, twist_vector(normal, turns, mirror), tile[::-1] if turns % 2 else tile,
            (90 * turns + (-heading if mirror else heading)) % 360,
            twist_vector(offset, turns, mirror), ticks)


def write_map(path, split, family, name, plan, mode, rng, *, task, fixture=None):
    cells, targets, normal, tile, heading, offset, ticks = plan
    goal = targets[-1]
    switch = mode == "switch"
    verify_layout(cells, targets, switch)
    walls = list(wall_segments(cells, tile))
    exit_segment = next(seg for cell, direction, seg in walls if cell == goal and direction == normal) if switch else None
    distractor_walls = [seg for cell, _, seg in walls if cell != goal and cells[cell] not in "DRBY"]
    n = 1 if fixture else rng.randint(1, 3)
    distractors = rng.sample(distractor_walls, min(n, len(distractor_walls)))
    minimums = {"keycards": 1, "locks": 1, "doors": 1} if task in ("keycard", "gauntlet") else {}
    challenge = Challenge(name, name, "Reach the engine exit after the specified progression.",
                          grid_text(cells), minimums, ticks, heading)
    marking = rng.choice(EXIT_TEXTURES), rng.choice(EXIT_SIGNS)
    lumps = build_lumps(challenge, tile=tile, spawn_offset=offset)
    write_wad(path, decorate(lumps, exit_segment, distractors, marking, rng))
    midpoint = (tuple((a + b) / 2 for a, b in zip(*exit_segment)) if switch else None)
    approach = (tuple(axis - direction * 26 for axis, direction in zip(midpoint, normal)) if switch else None)
    row = dict(id=name, path=path.as_posix(), sha256=sha256(path.read_bytes()).hexdigest(), map_name="MAP01",
               geometry_sha256=layout_digest(cells, labeled=False), split=split, family=family, task=task,
               stage={"keycard": "keys", "gauntlet": "gauntlet"}.get(task, "exploration"),
               skill="exit-switch", level=0, ticks=ticks, tile=tile, initial_heading=heading, spawn_offset=offset,
               exit_type=mode, exit_cell=goal, exit_segment=exit_segment, exit_approach=approach,
               exit_texture=marking[0] if switch else None, sign_texture=marking[1], distractor_segments=distractors,
               drill=dict(cells=[[*cell, glyph] for cell, glyph in sorted(cells.items())],
                          targets=targets, minimums=minimums, exit_switch=switch, kill_lesson=False),
               source="Original UDMF course using Freedoom 2 assets")
    if fixture:
        row["fixture"] = fixture
    return row


def historical_geometry():
    known = {layout_digest(challenge.cells(), labeled=False) for challenge in CHALLENGES}
    for filename in PRIOR_MAPS:
        path = Path(filename)
        if not path.is_file():
            continue
        for row in json.loads(path.read_text())["maps"]:
            if row.get("geometry_sha256"):
                known.add(row["geometry_sha256"])
            if row.get("drill", {}).get("cells"):
                cells = {(x, y): glyph for x, y, glyph in row["drill"]["cells"]}
                known.add(layout_digest(cells, labeled=False))
    return known


def generated(out, seed, split, families, used):
    for family, count in families.items():
        folder = out / split / family
        folder.mkdir(parents=True)
        rng = random.Random(f"{seed}:{split}:{family}")
        task = {"card-vestibule": "keycard", "guarded-vestibule": "gauntlet"}.get(family, "conditional")
        for index in range(count):
            mode = "walkover" if task == "conditional" and index % 4 == 3 else "switch"
            for _ in range(1500):
                layout = plan(family, rng, rng.randrange(4), bool(rng.randrange(2)))
                geometry = layout_digest(layout[0], labeled=False)
                if geometry in used:
                    continue
                try:
                    verify_layout(*layout[:2], mode == "switch")
                except ValueError:
                    continue
                used.add(geometry)
                break
            else:
                raise ValueError(f"Unable to create a disjoint layout for {split}/{family}/{index}")
            yield write_map(folder / f"{index:03d}.wad", split, family, f"{family}-{index:03d}",
                            layout, mode, rng, task=task)


def fixtures(out, seed):
    folder = out / "fixtures"
    folder.mkdir()
    rng = random.Random(f"{seed}:fixtures")
    hall = dict.fromkeys(rectangle(0, 0, 4, 3), ".")
    hall[0, 1], hall[3, 1] = "P", "E"
    # Along the thin axis both the closed door and the exit wall lie within USE
    # reach at spawn. The same held USE can be carried through the opened door.
    barrier = {(0, 0): ".", (1, 0): "P", (2, 0): "D", (3, 0): "E"}
    cases = (
        ("front-switch", (hall, [(3, 1)], (1, 0), (96, 96), 19, (8, -14), 700), "switch"),
        ("reversed-switch", (hall, [(3, 1)], (0, 1), (88, 80), 223, (-16, 15), 700), "switch"),
        ("use-edge", (barrier, [(2, 0), (3, 0)], (1, 0), (24, 112), 0, (-4, 0), 700), "switch"),
        ("walkover", (hall, [(3, 1)], (1, 0), (96, 96), 315, (-10, 8), 700), "walkover"),
    )
    for name, layout, mode in cases:
        if name == "reversed-switch":
            taller = dict.fromkeys(rectangle(0, 0, 4, 2), ".")
            taller[0, 0], taller[3, 1] = "P", "E"
            layout = (taller, *layout[1:])
        yield write_map(folder / f"{name}.wad", "development", "exit-fixtures", name, layout,
                        mode, rng, task="diagnostic", fixture=name)


def build(out, seed):
    out.mkdir(parents=True, exist_ok=False)
    old = historical_geometry()
    def save(name, rows):
        path = out / name
        data = dict(seed=seed, generator="training.exit_switch_maps", maps=rows)
        path.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
        return sha256(path.read_bytes()).hexdigest()
    diagnostic = list(fixtures(out, seed))
    used = old | {row["geometry_sha256"] for row in diagnostic}
    final = list(generated(out, seed, "test", FINAL, used))
    valid = list(generated(out, seed, "validation", VALIDATION, used))
    train = list(generated(out, seed, "train", TRAIN, used))
    manifests = {"manifest.json": train + valid, "final.json": final, "fixtures.json": diagnostic}
    hashes = {name: save(name, rows) for name, rows in manifests.items()}
    load_verified_maps(out)
    print(json.dumps(dict(training=Counter(row["exit_type"] for row in train), validation=len(valid),
                          test=Counter(row["task"] for row in final), fixtures=len(diagnostic),
                          manifest_sha256=hashes), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=513081)
    args = parser.parse_args()
    build(args.out, args.seed)
