"""Seeded post-lock drills with real doorway and exit witnesses."""
from __future__ import annotations

from collections import Counter
from math import floor, isclose
from pathlib import Path
from random import Random
import shutil

from doom_env import make_env
from scripts.adaptive_mixed import OUT as ARCHIVE, checked, pin, read, write
from scripts.exit_recovery_qa import RecoveryWalk
from scripts.exit_switch_qa import static_check
from training.exit_recovery_maps import distance_to_wall, preopen
from training.exit_switch_maps import twist_vector, write_map
from training.maps import load_verified_maps
from training.universe import layout_digest, orient, path_distances, rectangle


FAMILIES = {
    "train": ("landing", "elbow", "annex", "branch"),
    "validation": ("stagger", "court"),
}


def geometry(rng, family, before_door):
    gate = (2, 0)
    behind = {(0, 0), (1, 0), gate}
    behind |= rectangle(0, -rng.randrange(1, 4), rng.choice((1, 2)), rng.randrange(1, 3))
    behind |= {(0, y) for y in range(-2, 1)}
    x, width, depth = rng.randrange(5, 9), rng.randrange(2, 5), rng.randrange(2, 5)
    entry = {(i, 0) for i in range(3, x + 1)}
    if family == "landing":
        room = rectangle(3, -depth // 2, x - 2, depth + 1)
        goal, normal = (x, rng.choice(sorted(y for xx, y in room if xx == x))), (1, 0)
    elif family in {"elbow", "annex", "stagger"}:
        entry |= {(x, y) for y in range(depth)}
        room = rectangle(x - width + 1, depth, width, rng.choice((1, 2, 3)))
        goal = min(room)
        normal = (-1, 0)
        if family == "annex":
            entry |= {(3, -y) for y in range(1, rng.randrange(2, 5))}
        elif family == "stagger":
            entry |= rectangle(x - 2, -rng.randrange(2, 4), 2, 1)
            low = min(y for xx, y in entry if xx == x - 2)
            entry |= {(x - 2, y) for y in range(low, 0)}
    elif family in {"branch", "court"}:
        entry |= rectangle(3, -depth, 1, depth)
        entry |= rectangle(4, -depth, rng.randrange(1, x - 2), 1)
        room = rectangle(x, 0, width, depth)
        if family == "court":
            room |= rectangle(x + 1, -2, width - 1, 2)
            if width > 2 and depth > 2:
                room.remove((x + 1, 1))
        goal = max(room)
        normal = (1, 0)
    else:
        raise ValueError(f"Unknown post-lock geometry family: {family}")
    behind = {cell for cell in behind if cell[0] <= 2}
    cells = dict.fromkeys(behind | entry | room, ".")
    spawn_options = [(3, 0)]
    if family == "landing":
        spawn_options += sorted((x, y) for x, y in room if x < 5 and (x, y) != goal)
    elif family in {"branch", "court", "annex"}:
        spawn_options += [(3, -1), (3, -2)]
    start = (1, 0) if before_door else rng.choice([point for point in spawn_options if point in cells])
    cells.update({start: "P", gate: "0", goal: "1"})
    turns, reflected = rng.randrange(4), bool(rng.randrange(2))
    layout = orient(cells, turns, reflected)
    door, exit_cell = (next(c for c, value in layout.items() if value == label) for label in ("0", "1"))
    layout.update({door: "D", exit_cell: "E"})
    tile = rng.choice((72, 80, 88, 96)), rng.choice((76, 84, 92))
    heading = rng.choice((-155, -95, -47, 12, 54, 110, 161)) + rng.randrange(-8, 9)
    offset = (rng.randrange(-12, 13), rng.randrange(-12, 13))
    return ((layout, [exit_cell], twist_vector(normal, turns, reflected), tile[::-1] if turns % 2 else tile,
             (90 * turns + (-heading if reflected else heading)) % 360,
             twist_vector(offset, turns, reflected), 680 if family == "landing" else 920), door)


def far_side(row):
    door = tuple(row["open_door_cell"])
    tiles = {(x, y) for x, y, _ in row["drill"]["cells"]} - {door}
    return set(path_distances(tiles, tuple(row["exit_cell"])))


def spawn_cell(row):
    return next((x, y) for x, y, glyph in row["drill"]["cells"] if glyph == "P")


def verify_route(row):
    static_check(row)
    door = tuple(row["open_door_cell"])
    far = far_side(row)

    class Witness(RecoveryWalk):
        def step(self, *buttons):
            earlier = self.cell()
            super().step(*buttons)
            if earlier == door and self.cell() in far and self.first_crossing is None:
                self.first_crossing = self.clock.elapsed

    with make_env(row["path"], row["map_name"], episode_ticks=row["ticks"], render_buffers=False) as env:
        witness = Witness(env, row, door, 0)
        witness.first_crossing = None
        position = witness.position()
        assert all(isclose(a, b, abs_tol=1e-6) for a, b in zip(position, row["expected_start"], strict=True))
        assert not witness.state["keys"].any()
        assert witness.gap(door) == 124 and witness.state["door_lock"][witness.sectors[door]] == 0
        assert (witness.cell() in far) == (row["post_lock_spawn"] == "far-side")
        outcome = witness.finish()
        assert (witness.first_crossing is not None) == (row["post_lock_spawn"] == "approach-side")
        assert row["exit_type"] == "walkover" or "USE" in witness.last_buttons
        return dict(outcome, id=row["id"], split=row["split"], family=row["family"], mode=row["exit_type"],
                    sha256=row["sha256"], seed=0, start_side=row["post_lock_spawn"],
                    preopened_clearance=124, approach_crossing_tic=witness.first_crossing)


def prepare(output, seed, count):
    archive = read(ARCHIVE / "mixed-plan.json")["inputs"]
    archived = load_verified_maps(checked(archive["training"]).parent)
    if not all(row["split"] == "train" for row in archived):
        raise ValueError("The bootstrap is valid only for its verified training partition.")
    public = (checked(archive["development"]),
              ARCHIVE / "hard-maps" / "manifest.json",
              ARCHIVE / "lock-crossing-maps" / "manifest.json",
              ARCHIVE / "imitation" / "data" / "training-manifest.json",
              ARCHIVE.parent / "limit-campaign-20261127" / "original-120-panel.json")
    prior = [*archived, *(row for path in public if path.is_file() for row in read(path)["maps"])]
    geometry_seen = {row["geometry_sha256"] for row in prior if row.get("geometry_sha256")}
    hash_seen = {row["sha256"] for row in prior if row.get("sha256")}
    output.mkdir(parents=True, exist_ok=False)
    common = output / "retention"
    common.mkdir()
    rows, witnesses = [], []
    for row in archived:
        destination = common / f"{row['id']}.wad"
        shutil.copyfile(row["path"], destination)
        rows.append(dict(row, path=str(destination)))
    for split, families in FAMILIES.items():
        for family in families:
            rng = Random(f"{seed}:{split}:{family}")
            label = f"continuity-{family}"
            directory = output / split / label
            directory.mkdir(parents=True)
            for index in range(count if split == "train" else max(4, count // 2)):
                name = f"{label}-{index:03d}"
                before = index % 3 == 0
                mode = "walkover" if index % 4 == 3 else "switch"
                path = directory / f"{name}.wad"
                for _ in range(120):
                    design, door = geometry(rng, family, before)
                    identity = layout_digest(design[0], labeled=False)
                    if identity in geometry_seen:
                        continue
                    try:
                        row = write_map(path, split, label, name, design, mode, rng, task="conditional")
                        preopen(row, door)
                        assert row["sha256"] not in hash_seen
                        position = [int((cell + .5) * tile + offset)
                                    for cell, tile, offset in zip(spawn_cell(row), row["tile"], row["spawn_offset"])]
                        distance = distance_to_wall(position, row["exit_segment"]) if mode == "switch" else None
                        row.update(expected_start=position, post_lock_spawn="approach-side" if before else "far-side",
                                   initial_switch_distance=distance, starts_in_use_range=distance is not None and distance <= 64,
                                   curriculum_stage="conditional-post-lock", recovery_practice=True, level=3,
                                   continuity_post_lock=True)
                        receipt = verify_route(row)
                    except (AssertionError, RuntimeError, ValueError):
                        path.unlink(missing_ok=True)
                        continue
                    hash_seen.add(row["sha256"])
                    geometry_seen.add(identity)
                    rows.append(row)
                    witnesses.append(receipt)
                    break
                else:
                    raise ValueError(f"No engine-certified layout generated for {name}")
    manifest = output / "manifest.json"
    write(manifest, dict(seed=seed, maps=rows))
    load_verified_maps(output)
    counts = Counter(row["split"] for row in witnesses)
    write(output / "qa.json", dict(manifest=pin(manifest), verified=len(witnesses), witnesses=witnesses,
                                   counts=counts, sealed_final_access=False, authored_map_purpose="conditional post-lock"))
    print(f"Engine-certified {counts['train']} new training drills and {counts['validation']} distinct development drills.",
          flush=True)
