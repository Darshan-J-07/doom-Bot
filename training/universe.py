"""Build disjoint, reproducible ViZDoom curricula from original layouts."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from hashlib import sha256
import json
from pathlib import Path
import random

from doom_env.wad import write_wad
from examination.maps import CHALLENGES, DOORS, Challenge, build_lumps
from training.build_skill_curriculum import grid_text
from training.maps import load_verified_maps
from training.skills import NEIGHBORS


TRAIN_FAMILIES = {
    "hall": ("movement", 0), "elbow": ("movement", 0),
    "switchback": ("movement", 1), "obstacle-room": ("movement", 1),
    "fork": ("exploration", 0), "room-field": ("exploration", 1),
    "rectangular-loop": ("exploration", 1), "maze": ("exploration", 2),
    "target": ("combat", 0), "ambush": ("combat", 1), "survival": ("combat", 2),
    "single-door": ("doors", 0), "door-series": ("doors", 1), "exit-switch": ("doors", 2),
    "key-in-path": ("keys", 0), "side-key": ("keys", 1), "double-lock": ("keys", 2),
    "fight-and-lock": ("gauntlet", 0), "supply-route": ("gauntlet", 1),
    "three-lock": ("gauntlet", 2),
}
COMBINATIONS = {"fork-ambush": ("combat", 2), "loop-switch": ("doors", 2),
                "early-lock": ("keys", 2), "supply-vault": ("gauntlet", 2)}
DEVELOPMENT = {"hub-vault": ("gauntlet", 2), "walled-hub": ("exploration", 2)}
FINAL = {"braided-maze": ("exploration", 2), "loop-vault": ("gauntlet", 2),
         "corner-guard": ("combat", 2), "yellow-vault": ("keys", 2)}
FAMILIES = TRAIN_FAMILIES | COMBINATIONS | DEVELOPMENT | FINAL


def rectangle(x0, y0, width, height):
    return {(x, y) for x in range(x0, x0 + width) for y in range(y0, y0 + height)}


def neighbors(cell):
    x, y = cell
    return [(x + dx, y + dy) for dx, dy in NEIGHBORS]


def orient(cells, turns, mirror):
    result = {}
    for (x, y), glyph in cells.items():
        if mirror:
            y = -y
        if turns == 1:
            x, y = -y, x
        elif turns == 2:
            x, y = -x, -y
        elif turns == 3:
            x, y = y, -x
        result[x, y] = glyph
    left = min(x for x, _ in result)
    bottom = min(y for _, y in result)
    return {(x - left, y - bottom): glyph for (x, y), glyph in result.items()}


def layout_digest(cells, *, labeled):
    """Canonicalize translation, reflection and rotation before partitioning."""
    geometry = cells if labeled else dict.fromkeys(cells, ".")
    canonical = min(grid_text(orient(geometry, turn, mirror))
                    for turn in range(4) for mirror in (False, True))
    return sha256(canonical.encode()).hexdigest()


def path_distances(cells, start):
    queue = deque([start])
    distance = {start: 0}
    while queue:
        cell = queue.popleft()
        for other in neighbors(cell):
            if other in cells and other not in distance:
                distance[other] = distance[cell] + 1
                queue.append(other)
    return distance


def carved_maze(rng, braided):
    width, height = rng.randint(4, 7), rng.randint(3, 6)
    graph = {(2 * x, 2 * y) for x in range(width) for y in range(height)}
    start = rng.choice(sorted(graph))
    stack, cells, visited = [start], {start}, {start}
    while stack:
        x, y = stack[-1]
        available = [(x + 2 * dx, y + 2 * dy) for dx, dy in NEIGHBORS
                     if (x + 2 * dx, y + 2 * dy) in graph - visited]
        if not available:
            stack.pop()
            continue
        cell = rng.choice(available)
        cells.update((cell, ((x + cell[0]) // 2, (y + cell[1]) // 2)))
        visited.add(cell)
        stack.append(cell)
    if braided:
        gaps = sorted({(x + dx, y + dy) for x, y in graph for dx, dy in NEIGHBORS
                       if (x + 2 * dx, y + 2 * dy) in graph} - cells)
        cells.update(rng.sample(gaps, min(len(gaps), rng.randint(3, 7))))
    distance = path_distances(cells, start)
    goal = rng.choice(sorted(cell for cell, length in distance.items() if length >= max(distance.values()) - 1))
    return cells, start, goal


def cells_for_family(family, rng):
    length = rng.randint(5, 12)
    start, goal = (0, 0), (length - 1, 0)
    cells = rectangle(0, 0, length, 1)
    if family in ("hall", "exit-switch"):
        for x in range(1, length - 1):
            if rng.random() < (.27 if family == "hall" else .48):
                cells.add((x, rng.choice((-1, 1))))
    elif family in ("elbow", "switchback", "corner-guard"):
        width, height = rng.randint(3, 8), rng.randint(2, 6)
        cells = rectangle(0, 0, width, 1) | rectangle(width - 1, 1, 1, height)
        goal = (width - 1, height)
        if family != "elbow":
            approach = rng.randint(2, 5)
            direction = rng.choice((-1, 1))
            cells |= {(width - 1 + i * direction, height) for i in range(1, approach + 1)}
            goal = (width - 1 + approach * direction, height)
            if rng.random() < .55:
                extra = rng.randint(1, 3)
                cells |= rectangle(goal[0], height + 1, 1, extra)
                goal = goal[0], height + extra
        for x, y in sorted(cells):
            if (x, y) not in (start, goal) and rng.random() < .24:
                cells.add((x, y + rng.choice((-1, 1))))
    elif family in ("obstacle-room", "target", "ambush", "survival", "room-field"):
        half = rng.randint(1, 3)
        cells = rectangle(1, -half, length - 2, 2 * half + 1) | {start, goal}
        if family in ("obstacle-room", "room-field"):
            for x in rng.sample(range(2, length - 2), rng.randint(1, min(3, length - 4))):
                cells.discard((x, 0))
                if half > 1 and rng.random() < .5:
                    cells.discard((x, rng.choice((-1, 1))))
        if family == "room-field":
            for x in rng.sample(range(2, length - 1), min(3, length - 3)):
                side = rng.choice((-1, 1))
                cells |= {(x, side * (half + 1))}
                cells |= rectangle(x - 1, side * (half + 2), rng.randint(2, 4), 1)
        if family in ("target", "ambush", "survival"):
            for x in range(1, length - 1):
                if rng.random() < .35:
                    cells.add((x, rng.choice((-1, 1)) * (half + 1)))
    elif family in ("fork", "fork-ambush", "walled-hub"):
        axes = rng.sample(range(1, length - 1), min(length - 2, rng.randint(2, 5)))
        for i, x in enumerate(axes):
            depth, side = rng.randint(2, 5), (-1) ** i
            cells.update((x, side * y) for y in range(1, depth + 1))
            if i == 0 or family == "walled-hub":
                cells |= rectangle(x - 1, side * depth, rng.randint(3, 5), 1)
            if i == 0 and rng.random() < .60:
                goal = (x, side * depth)
        if family == "walled-hub":
            cells |= rectangle(1, -1, 3, 3)
            cells -= {(2, -1), (2, 1)}
    elif family in ("rectangular-loop", "loop-switch"):
        width, half = rng.randint(4, 10), rng.randint(2, 5)
        cells = {(x, y) for x, y in rectangle(1, -half, width, 2 * half + 1)
                 if x in (1, width) or abs(y) == half}
        for x in rng.sample(range(2, width), rng.randint(1, min(4, width - 2))):
            y = rng.choice((-half, half))
            cells |= {(x, y + k * (1 if y > 0 else -1)) for k in range(1, rng.randint(2, 5))}
        goal = (width + 1, 0)
        cells |= {start, goal}
    elif family in ("maze", "braided-maze"):
        cells, start, goal = carved_maze(rng, family == "braided-maze")
    return dict.fromkeys(cells, "."), start, goal


def progression(family, rng):
    skill, _ = FAMILIES[family]
    nkeys = {"double-lock": 2, "early-lock": 2, "supply-vault": 2, "hub-vault": 2,
             "loop-vault": 2, "yellow-vault": 2, "three-lock": 3}.get(family, int(skill in ("keys", "gauntlet")))
    plain = {"single-door": 1, "door-series": rng.randint(2, 3), "supply-route": 1,
             "hub-vault": 1, "loop-vault": 1}.get(family, 0)
    colors = rng.sample("rby", nkeys)
    if family == "yellow-vault" and "y" not in colors:
        colors[-1] = "y"
    labels = list(colors) + ["D"] * plain
    rng.shuffle(labels)
    doors, keys, cells = [], [], {}
    cursor = 0
    for label in labels:
        span = rng.randint(3, 5 if family in ("key-in-path", "single-door") else 7)
        door = cursor + span
        cells.update(dict.fromkeys(rectangle(cursor, 0, span + 1, 1), "."))
        door_cell = door, 0
        doors.append(door_cell)
        cells[door_cell] = label.upper()
        side = rng.choice((-1, 1))
        for x in range(cursor + 1, door - 1):
            if rng.random() < .45 or family in ("hub-vault", "loop-vault"):
                cells[x, side] = "."
        if label != "D":
            x = rng.randint(cursor + 1, door - 1)
            depth = 0 if family in ("key-in-path", "fight-and-lock") else rng.randint(1, 4)
            if family in ("early-lock", "hub-vault", "loop-vault"):
                x, depth = door - 1, rng.randint(2, 5)
            cells.update({(x, side * y): "." for y in range(1, depth + 1)})
            key = x, side * depth
            cells[key] = label
            keys.append(key)
            if family in ("hub-vault", "loop-vault"):
                cells.update({(x - 1, side * y): "." for y in range(1, depth + 1) if x - 1 > cursor})
        cursor = door + 1
    tail = rng.randint(3, 6)
    cells.update(dict.fromkeys(rectangle(cursor, 0, tail, 1), "."))
    for x in range(cursor, cursor + tail - 1):
        if rng.random() < .55:
            side = rng.choice((-1, 1))
            cells[x, side] = "."
            if rng.random() < .30:
                cells[x, 2 * side] = "."
    if skill == "gauntlet" and family != "fight-and-lock":
        cells.update(dict.fromkeys(rectangle(cursor + 1, 1, tail - 1, 2), "."))
    return cells, (0, 0), (cursor + tail - 1, 0), doors, keys


def design(family, rng):
    skill, level = FAMILIES[family]
    switch = family in ("exit-switch", "loop-switch", "hub-vault", "yellow-vault")
    if skill in ("keys", "gauntlet") or family in ("single-door", "door-series"):
        cells, start, goal, doors, keys = progression(family, rng)
    else:
        cells, start, goal = cells_for_family(family, rng)
        doors, keys = [], []
    required = {}
    if doors:
        required["doors"] = len(doors)
    if keys:
        required.update(keycards=len(keys), locks=len(keys))
    available = sorted(cell for cell, glyph in cells.items() if glyph == "." and cell not in (start, goal))
    health, ammo = 100, 50
    if skill == "combat" or skill == "gauntlet":
        count = (1, 2, 3)[level]
        distance = path_distances(cells, start)
        candidates = [cell for cell in available if distance.get(cell, 0) >= 2]
        if family == "target":
            candidates = [cell for cell in candidates if 2 <= cell[0] <= 4 and abs(cell[1]) <= 1]
        enemies = rng.sample(candidates, min(count, len(candidates)))
        for cell in enemies:
            cells[cell] = rng.choice("zzi" if level == 0 else "zziid")
        required["kills"] = len(enemies) if skill == "combat" else 1
        if level > 0:
            supplies = [cell for cell in available if cells[cell] == "."]
            for index, cell in enumerate(rng.sample(supplies, min(len(supplies), rng.randint(2, 4)))):
                cells[cell] = "hamh"[index]
            health = rng.choice((42, 56, 72, 100))
            ammo = rng.choice((12, 20, 32, 50))
    cells[start], cells[goal] = "P", "E"
    key_by_color = {cells[cell].upper(): cell for cell in keys}
    ordered = []
    for door in doors:
        key = key_by_color.get(cells[door])
        if key is not None:
            ordered.append(key)
        ordered.append(door)
    ordered.append(goal)
    distances = path_distances(cells, ordered[0])
    first_hop = min((cell for cell in neighbors(start) if cell in cells),
                    key=lambda cell: distances.get(cell, float("inf")))
    dx, dy = first_hop[0] - start[0], first_hop[1] - start[1]
    heading = {(1, 0): 0, (0, 1): 90, (-1, 0): 180, (0, -1): 270}[dx, dy]
    heading = (heading + rng.choice((-30, -15, 0, 0, 0, 15, 30) if level == 0
                                    else (-120, -90, -45, -20, 0, 20, 45, 90, 120))) % 360
    # Rotate the complete map, including its target ordering and spawn bearing.
    targets = [cells[cell] for cell in ordered]
    tagged = dict(cells)
    for index, cell in enumerate(ordered):
        tagged[cell] = str(index)
    turns, mirror = rng.randrange(4), bool(rng.randrange(2))
    tagged = orient(tagged, turns, mirror)
    order = {int(glyph): cell for cell, glyph in tagged.items() if glyph.isdigit()}
    for index, cell in order.items():
        tagged[cell] = targets[index]
    angle = ((-heading if mirror else heading) + 90 * turns) % 360
    ticks = (700, 1260, 2100)[level] if skill != "gauntlet" else (1400, 2100, 2800)[level]
    lesson = family == "target"
    challenge = Challenge(family, family, family, grid_text(tagged), required, ticks, angle, health, ammo)
    tile = tuple(rng.choice((96, 112, 128, 144, 176, 208)) for _ in range(2))
    settings = dict(tile=tile, spawn_offset=(rng.randint(-15, 15), rng.randint(-15, 15)),
                    door_wait=rng.choice((70, 105, 150)), exit_switch=switch)
    return challenge, settings, [order[i] for i in range(len(order))], lesson


def verify_layout(cells, targets, switch):
    """Prove grid reachability with ordered keys, doors, and an independent exit."""
    start = next(cell for cell, glyph in cells.items() if glyph == "P")
    goal = targets[-1]
    adjacent = neighbors(goal)
    if not (any(cell not in cells for cell in adjacent) if switch else
            any(cell in cells and cells[cell] not in DOORS for cell in adjacent)):
        raise ValueError("Missing independent exit trigger")
    reachable = set()
    reached = False
    queue = deque([(start, 0)])
    states = set(queue)
    while queue:
        cell, next_target = queue.popleft()
        reachable.add(cell)
        if cell == goal:
            reached |= next_target == len(targets)
            continue
        keys = {cells[point] for point in targets[:next_target] if cells[point] in "rby"}
        for other in neighbors(cell):
            glyph = cells.get(other)
            if glyph is None or glyph in "RBY" and glyph.lower() not in keys:
                continue
            index = next_target + (next_target < len(targets) and other == targets[next_target])
            entry = other, index
            if entry not in states:
                states.add(entry)
                queue.append(entry)
    if not reached or reachable != cells.keys():
        raise ValueError("Unreachable tile or ordered goal")


def build_dataset(root, seed, per_family, validation_per_family, dev_per_family, final_per_family):
    root.mkdir(parents=True, exist_ok=False)
    old_layouts = {layout_digest(challenge.cells(), labeled=False) for challenge in CHALLENGES}
    owner, seen, rows = {}, set(), []
    splits = (
        ("test", FINAL, final_per_family, "held-out family"),
        ("development", DEVELOPMENT, dev_per_family, "held-out family"),
        ("validation", COMBINATIONS, validation_per_family, "unseen combination"),
        ("validation", TRAIN_FAMILIES, validation_per_family, "familiar distribution"),
        ("train", TRAIN_FAMILIES, per_family, "procedural"),
    )
    rejected = Counter()
    for split, families, count, category in splits:
        for family, (skill, level) in families.items():
            stage = "exploration" if skill == "doors" else skill
            rng = random.Random(f"{seed}/{split}/{family}")
            dest = root / split / family
            if count:
                dest.mkdir(parents=True)
            for index in range(count):
                for _ in range(2000):
                    challenge, settings, targets, lesson = design(family, rng)
                    cells = challenge.cells()
                    geometry = layout_digest(cells, labeled=False)
                    layout = layout_digest(cells, labeled=True)
                    if geometry in old_layouts or owner.get(geometry, split) != split or layout in seen:
                        rejected["overlap"] += 1
                        continue
                    try:
                        verify_layout(cells, targets, settings["exit_switch"])
                    except ValueError:
                        rejected["reachability"] += 1
                        continue
                    break
                else:
                    raise ValueError(f"Insufficient distinct solvable {split}/{family} layouts after {index} maps")
                path = dest / f"{index:04d}.wad"
                write_wad(path, build_lumps(challenge, **settings))
                owner[geometry] = split
                seen.add(layout)
                rows.append(dict(path=path.as_posix(), map_name="MAP01", split=split, stage=stage,
                                 skill=skill, level=level, family=family, category=category,
                                 sha256=sha256(path.read_bytes()).hexdigest(), geometry_sha256=geometry,
                                 layout_sha256=layout, ticks=challenge.ticks, tile=settings["tile"],
                                 drill=dict(cells=[[*point, glyph] for point, glyph in cells.items()],
                                            targets=targets, minimums=challenge.minimums,
                                            kill_lesson=lesson, exit_switch=settings["exit_switch"],
                                            switch_wait=settings["door_wait"], initial_health=challenge.initial_health,
                                            initial_bullets=challenge.initial_bullets)))
    manifest = dict(seed=seed, generator="universe-v1", rejected=dict(rejected), maps=rows)
    (root / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n", encoding="utf-8")
    load_verified_maps(root)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=304811)
    parser.add_argument("--per-family", type=int, default=110)
    parser.add_argument("--validation-per-family", type=int, default=2)
    parser.add_argument("--development-per-family", type=int, default=4)
    parser.add_argument("--final-per-family", type=int, default=6)
    args = parser.parse_args()
    if min(args.per_family, args.validation_per_family, args.development_per_family, args.final_per_family) < 1:
        parser.error("Each split needs at least one map per family")
    manifest = build_dataset(args.out, args.seed, args.per_family, args.validation_per_family,
                             args.development_per_family, args.final_per_family)
    print(json.dumps(dict(counts=dict(Counter(row["split"] for row in manifest["maps"])),
                          rejected=manifest["rejected"], manifest=str(args.out / "manifest.json"))))


if __name__ == "__main__":
    main()
