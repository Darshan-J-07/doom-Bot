"""Self-authored UDMF curricula; uses the official WAD writer, no external map assets."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
from doom_env.mapinfo import LOCK_BLUE, LOCK_RED
from doom_env.wad import Lump, write_wad

STAGES = ("movement", "combat", "exploration", "keys", "gauntlet")


def create_map(path, stage, seed, *, two_locks=False):
    if two_locks and stage != "gauntlet":
        raise ValueError("Two-lock layouts require the gauntlet stage")
    rng = random.Random(seed)
    door = key = second_door = second_key = None
    if stage == "movement":
        length = rng.randint(3, 5)
        cells = {(x, 0) for x in range(length)}
        start, goal = (0, 0), (length - 1, 0)
    elif stage == "combat":
        cells = {(x, y) for x in range(3) for y in range(3)}
        start, goal = (0, 1), (2, 1)
    elif stage == "exploration":
        # Random connected polyomino, with branches and dead ends.
        cells = {(0, 0)}
        for _ in range(rng.randint(8, 16)):
            x, y = rng.choice(sorted(cells))
            dx, dy = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            cells.add((x + dx, y + dy))
        start = (0, 0)
        goal = max(sorted(cells), key=lambda c: abs(c[0]) + abs(c[1]))
        # Door placement off the start tile; some layouts have alternative routes.
        candidates = sorted(cells - {start, goal})
        door = rng.choice(candidates) if candidates else None
    elif two_locks:
        length = rng.randint(10, 12)
        first_branch, second_branch = rng.choice((1, 2)), rng.choice((6, 7))
        cells = {(x, 0) for x in range(length)}
        cells.update(((first_branch, 1), (first_branch, 2),
                      (second_branch, -1), (second_branch, -2)))
        start, goal = (0, 0), (length - 1, 0)
        door, key = (4, 0), (first_branch, 2)
        second_door, second_key = (length - 2, 0), (second_branch, -2)
    else:
        length = rng.randint(6, 8)
        branch = rng.randint(1, 2)
        cells = {(x, 0) for x in range(length)} | {(branch, 1), (branch, 2)}
        start, goal = (0, 0), (length - 1, 0)
        door, key = (length - 3, 0), (branch, 2)
    # Rotations change layouts relative to the random initial heading.
    rotations = rng.randrange(4)
    def rotate(c):
        if c is None: return None
        x, y = c
        for _ in range(rotations): x, y = -y, x
        return (x, y)
    cells = {rotate(c) for c in cells}
    start, goal, door, key, second_door, second_key = map(
        rotate, (start, goal, door, key, second_door, second_key))
    door_locks = {door: LOCK_RED if key is not None else 0} if door is not None else {}
    key_things = {key: 13} if key is not None else {}
    if two_locks:
        door_locks[second_door] = LOCK_BLUE
        key_things[second_key] = 5
    tile = rng.choice((128, 160, 192))
    sectors = {cell: i for i, cell in enumerate(sorted(cells))}
    vertices, sides, lines = [], [], []
    vertex_ids = {}
    def vertex(p):
        if p not in vertex_ids:
            vertex_ids[p] = len(vertices)
            vertices.append(f"vertex {{ x = {p[0]}; y = {p[1]}; }}")
        return vertex_ids[p]
    def side(cell, solid):
        idx = len(sides)
        middle = "STARTAN2" if solid else "-"
        sides.append(f'sidedef {{ sector = {sectors[cell]}; texturetop = "BIGDOOR2"; '
                     f'texturemiddle = "{middle}"; texturebottom = "STARTAN2"; }}')
        return idx
    done = set()
    for cell in sorted(cells):
        x, y = cell
        # Clockwise edges: the current tile is on the front (right) side.
        edges = [((x,y),(x,y+1),(-1,0)), ((x,y+1),(x+1,y+1),(0,1)),
                 ((x+1,y+1),(x+1,y),(1,0)), ((x+1,y),(x,y),(0,-1))]
        for a, b, delta in edges:
            neighbor = (x + delta[0], y + delta[1])
            edge = tuple(sorted((a, b)))
            if edge in done: continue
            done.add(edge)
            front, back = cell, neighbor if neighbor in cells else None
            # Manual door specials act on the sector behind the line.
            if front in door_locks and back is not None:
                a, b, front, back = b, a, back, front
            v1, v2 = vertex((a[0]*tile, a[1]*tile)), vertex((b[0]*tile, b[1]*tile))
            sf = side(front, back is None)
            props = f"v1 = {v1}; v2 = {v2}; sidefront = {sf}; "
            if back is None:
                props += "blocking = true;"
            else:
                props += f"sideback = {side(back, False)}; twosided = true; "
                if back in door_locks:
                    lock = door_locks[back]
                    special = 13 if lock else 12
                    props += (f"special = {special}; arg0 = 0; arg1 = 16; arg2 = 150; "
                              + (f"arg3 = {lock}; " if lock else "")
                              + "playeruse = true; repeatspecial = true;")
                elif goal in (front, back):
                    props += "special = 243; playercross = true;"
            lines.append("linedef { " + props + " }")
    sector_text = [f'sector {{ heightfloor = 0; heightceiling = {0 if c in door_locks else 128}; '
                   f'texturefloor = "FLOOR0_1"; textureceiling = "CEIL1_1"; lightlevel = {rng.randint(160,224)}; }}'
                   for c in sorted(cells)]
    things = []
    def thing(cell, kind, jitter=0, angle=None):
        x, y = (cell[0]+.5)*tile, (cell[1]+.5)*tile
        x += rng.uniform(-jitter, jitter); y += rng.uniform(-jitter, jitter)
        angle = rng.randrange(360) if angle is None else angle
        things.append(f"thing {{ x = {x:.3f}; y = {y:.3f}; type = {kind}; angle = {angle}; "
                      "skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; }")
    thing(start, 1, tile*.12)
    for location, kind in key_things.items():
        thing(location, kind)
    available = sorted(cells - {start, goal} - door_locks.keys() - key_things.keys())
    if stage in ("combat", "gauntlet"):
        for c in rng.sample(available, min(len(available), 1 if stage == "combat" else 3)):
            thing(c, rng.choice((3004, 3001)), tile*.12)
        for c in rng.sample(available, min(len(available), 1 if two_locks else 2)):
            thing(c, rng.choice((2011, 2007)), tile*.2)
    text = '\n'.join(['namespace = "zdoom";', *vertices, *sector_text, *sides, *lines, *things])
    write_wad(path, [Lump("MAP01", b""), Lump("TEXTMAP", text.encode()), Lump("ENDMAP", b"")])
    return dict(path=Path(path).as_posix(), stage=stage, seed=seed, map_name="MAP01",
                cells=len(cells), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def generate(root="training/maps", per_stage=6):
    root = Path(root)
    entries = []
    for split, offset, count in (("train", 1000, per_stage), ("validation", 90000, 2), ("test", 190000, 2)):
        for stage_id, stage in enumerate(STAGES):
            folder = root / split / stage
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                row = create_map(folder / f"{i:02}.wad", stage, offset + 100*stage_id + i)
                row["split"] = split
                entries.append(row)
    hashes = {split: {r["sha256"] for r in entries if r["split"] == split}
              for split in ("train", "validation", "test")}
    assert not (hashes["train"] & hashes["validation"] or hashes["train"] & hashes["test"]
                or hashes["validation"] & hashes["test"])
    manifest = dict(source="Self-authored procedural UDMF geometry; Freedoom textures at runtime",
                    license="Generated map geometry: CC0-1.0; engine/IWAD retain upstream licenses",
                    maps=entries)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return entries


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="training/maps")
    ap.add_argument("--per-stage", type=int, default=6)
    args = ap.parse_args()
    print(json.dumps(dict(generated=len(generate(args.root, args.per_stage)))))
