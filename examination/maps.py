"""Author and verify the separate examination PWADs."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from textwrap import dedent

from doom_env.mapinfo import LOCK_BLUE, LOCK_RED, LOCK_YELLOW, load_mapinfo
from doom_env.wad import Lump, list_maps, read_wad, write_wad

ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = Path(__file__).with_name("maps")
TIERS = ("Navigation", "Doors and combat", "Survival and resources", "Progression and gauntlet")
TILE = 144
DOORS = {"D": 0, "R": LOCK_RED, "B": LOCK_BLUE, "Y": LOCK_YELLOW}
THINGS = {"P": 1, "r": 13, "b": 5, "y": 6, "z": 3004, "i": 3001,
          "d": 3002, "h": 2011, "m": 2012, "a": 2007}


@dataclass(frozen=True)
class Challenge:
    slug: str
    name: str
    objective: str
    layout: str
    minimums: dict[str, int] = field(default_factory=dict)
    ticks: int = 1400
    heading: int = 0
    initial_health: int = 100
    initial_bullets: int = 50

    def cells(self):
        rows = dedent(self.layout).strip().splitlines()
        return {(x, len(rows) - row - 1): symbol for row, line in enumerate(rows)
                for x, symbol in enumerate(line) if symbol != "#"}


# Every challenge also requires a live exit within its tick budget.
CHALLENGES = (
    Challenge("straight-corridor", "Straight corridor navigation",
              "Traverse the corridor and reach the exit.", "P.......E", ticks=840),
    Challenge("multiple-turns", "Multiple turns",
              "Follow the bends and reach the exit.", """
              #...E
              #.###
              ...##
              .####
              P####
              """, ticks=1050, heading=90),
    Challenge("obstacle-avoidance", "Obstacle avoidance",
              "Go around the central obstruction and reach the exit.", """
              #......#
              P..##..E
              #......#
              """, ticks=1050),
    Challenge("dead-end-recovery", "Dead-end recovery",
              "Recover from a spawn facing a dead end and reach the exit.", """
              ####..E
              ####.##
              P....##
              """, ticks=1050, heading=180),
    Challenge("branching-corridors", "Branching corridors",
              "Visit at least two of the three marked branch checkpoints and reach the exit.", """
              ##*##*##
              ##.##.##
              P......E
              ###.####
              ###*####
              """, {"checkpoints": 2}),
    Challenge("single-door", "Single closed door",
              "Open the closed door with USE and reach the exit.", "P...D...E", {"doors": 1}),
    Challenge("connected-rooms", "Multiple connected rooms",
              "Pass through the three rooms, opening both doors, and reach the exit.", """
              ...#...#.....
              P..D...D....E
              ...#...#.....
              """, {"doors": 2}),
    Challenge("navigation-loop", "Circular navigation loop",
              "Visit all three marked checkpoints around the central wall and reach the exit.", """
              ##*...*##
              ##.###.##
              P..###..E
              ##.###.##
              ##..*..##
              """, {"checkpoints": 3}, ticks=1750),
    Challenge("single-enemy", "Single enemy encounter",
              "Defeat the zombieman and reach the exit.", """
              #.....#
              P..z..E
              #.....#
              """, {"kills": 1}),
    Challenge("multiple-enemies", "Multiple enemy encounters",
              "Defeat the enemy in each room and reach the exit.", """
              ######....E
              ######.z...
              ...###.####
              P.z....####
              ...########
              """, {"kills": 2}, ticks=1750),
    Challenge("low-health", "Low-health survival",
              "Start at 25 HP and reach the exit alive with an imp in the approach.", """
              #####....E
              #####.####
              P.....i###
              """, initial_health=25),
    Challenge("limited-ammunition", "Limited ammunition",
              "Defeat the imp and reach the exit with 12 starting bullets and no ammo pickup.", """
              #......#
              P..i...E
              #......#
              """, {"kills": 1}, initial_bullets=12),
    Challenge("health-collection", "Health resource collection",
              "Start at 40 HP, find the stimpack in the side passage, and reach the exit.", """
              ##..h###
              ##.#####
              P......E
              """, {"health_gained": 10}, initial_health=40),
    Challenge("ammunition-collection", "Ammunition collection",
              "Start with five bullets, collect ammunition from the side passage, and reach the exit.", """
              #####a#
              #####.#
              P......E
              """, {"ammo_gained": 1}, initial_bullets=5),
    Challenge("keycard-discovery", "Keycard discovery",
              "Find the red keycard in the side wing and reach the exit with it.", """
              ###..r##
              ###.####
              P......E
              """, {"keycards": 1}),
    Challenge("locked-door-backtracking", "Locked-door backtracking",
              "Leave the locked doorway, find the red keycard behind the spawn, then unlock the door and exit.", """
              ##..r
              ##.##
              ....P.R....E
              """, {"keycards": 1, "locks": 1}, ticks=1750),
    Challenge("keycard-dependencies", "Multiple keycard dependencies",
              "Unlock the red door to reach the blue key, then unlock the blue door and exit.", """
              ##.r#####..b
              ##.######.##
              P...R......B...E
              """, {"keycards": 2, "locks": 2}, ticks=2100),
    Challenge("complex-maze", "Complex maze exploration",
              "Find all three marked checkpoints in the maze and reach the exit.", """
              P....#...*
              .###.#.#.#
              ...#...#.#
              ##.###.#.#
              *..#...#.#
              .###.###.#
              .....#...E
              ##.#...#.#
              ##.*######
              """, {"checkpoints": 3}, ticks=2100),
    Challenge("combat-key-progression", "Combined combat and keycard progression",
              "Defeat the guard, collect the red key, unlock its door and reach the exit.", """
              ###..r
              ###.##
              P..z..R....E
              ##...#######
              """, {"kills": 1, "keycards": 1, "locks": 1}, ticks=2100),
    Challenge("full-gauntlet", "Full gauntlet",
              "Defeat at least two enemies, open all three doors using both keys, and reach the exit.", """
              ##.r#######..b
              ##.########.#
              P...D.z.R....i.B..z.E
              .a..#m..#......#.....
              """, {"kills": 2, "doors": 3, "keycards": 2, "locks": 2}, ticks=2100, initial_health=80),
)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def build_lumps(challenge, *, tile=(TILE, TILE), spawn_offset=(0, 0), door_wait=150, exit_switch=False):
    """Build engine lumps for a new grid using the wrapper's supported UDMF specials."""
    cells = challenge.cells()
    if (list(cells.values()).count("P") != 1 or list(cells.values()).count("E") != 1
            or set(cells.values()) - (THINGS.keys() | DOORS.keys() | {".", "*", "E"})):
        raise ValueError(f"Invalid layout glyphs, spawn or exit in {challenge.slug}")
    sectors = {cell: i for i, cell in enumerate(sorted(cells))}
    vertices, sides, lines, sector_text, things = [], [], [], [], []
    vertex_ids, drawn = {}, set()
    switch_drawn = False

    def vertex(point):
        if point not in vertex_ids:
            vertex_ids[point] = len(vertices)
            vertices.append(f"vertex {{ x = {point[0] * tile[0]}; y = {point[1] * tile[1]}; }}")
        return vertex_ids[point]

    def side(cell, middle):
        sides.append(f'sidedef {{ sector = {sectors[cell]}; texturetop = "BIGDOOR2"; '
                     f'texturemiddle = "{middle}"; texturebottom = "STARTAN2"; }}')
        return len(sides) - 1

    for x, y in sorted(cells):
        cell = (x, y)
        symbol = cells[cell]
        height = 0 if symbol in DOORS else 128
        floor = "FLAT14" if symbol == "*" else "FLOOR0_1"
        light = 224 if symbol in ("*", "E") else 168 + 8 * ((x + y) % 5)
        sector_text.append(f'sector {{ heightfloor = 0; heightceiling = {height}; '
                           f'texturefloor = "{floor}"; textureceiling = "CEIL1_1"; lightlevel = {light}; }}')
        if symbol in THINGS:
            angle = challenge.heading if symbol == "P" else 180
            dx, dy = spawn_offset if symbol == "P" else (0, 0)
            things.append(f"thing {{ x = {int((x + .5) * tile[0] + dx)}; y = {int((y + .5) * tile[1] + dy)}; "
                          f"type = {THINGS[symbol]}; angle = {angle}; "
                          "skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; }")
        # The current cell lies on the right of each clockwise edge.
        for a, b, neighbor in (((x, y), (x, y+1), (x-1, y)), ((x, y+1), (x+1, y+1), (x, y+1)),
                               ((x+1, y+1), (x+1, y), (x+1, y)), ((x+1, y), (x, y), (x, y-1))):
            edge = tuple(sorted((a, b)))
            if edge in drawn:
                continue
            drawn.add(edge)
            front, back = cell, neighbor
            if symbol in DOORS and back in cells:
                front, back, a, b = back, front, b, a
            front_symbol, back_symbol = cells[front], cells.get(back)
            texture = "-" if back_symbol else "EXITSIGN" if front_symbol == "E" else "STARTAN2"
            switch = exit_switch and back_symbol is None and front_symbol == "E" and not switch_drawn
            if switch:
                texture = "SW1EXIT"
            props = f"v1 = {vertex(a)}; v2 = {vertex(b)}; sidefront = {side(front, texture)}; "
            if back_symbol is None:
                props += "blocking = true;"
                if switch:
                    props += " special = 243; playeruse = true;"
                    switch_drawn = True
            else:
                props += f"sideback = {side(back, '-')}; twosided = true; "
                if back_symbol in DOORS:
                    lock = DOORS[back_symbol]
                    props += (f"special = {13 if lock else 12}; arg1 = 16; arg2 = {door_wait}; arg3 = {lock}; "
                              "playeruse = true; repeatspecial = true;")
                elif not exit_switch and "E" in (front_symbol, back_symbol):
                    props += "special = 243; playercross = true;"
            lines.append(f"linedef {{ {props} }}")
    if exit_switch and not switch_drawn:
        raise ValueError(f"Switch exit has no solid wall in {challenge.slug}")
    text = "\n".join(['namespace = "zdoom";', *vertices, *sector_text, *sides, *lines, *things])
    lumps = [Lump("MAP01", b""), Lump("TEXTMAP", text.encode()), Lump("ENDMAP", b"")]
    if (challenge.initial_health, challenge.initial_bullets) != (100, 50):
        patch = ("Patch File for DeHackEd v3.0\nDoom version = 19\nPatch format = 6\n\n"
                 f"Misc 0\nInitial Health = {challenge.initial_health}\n"
                 f"Initial Bullets = {challenge.initial_bullets}\n")
        lumps.append(Lump("DEHACKED", patch.encode()))
    return lumps


def filename(index, challenge):
    return f"{index:02d}-{challenge.slug}.wad"


def map_record(index, challenge, path):
    return dict(id=index, file=path.name, map_name="MAP01", sha256=digest(path),
                name=challenge.name, tier=(index - 1) // 5 + 1, objective=challenge.objective,
                tick_budget=challenge.ticks, requires_exit_alive=True, minimums=challenge.minimums,
                initial_health=challenge.initial_health, initial_bullets=challenge.initial_bullets)


def build():
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, challenge in enumerate(CHALLENGES, 1):
        path = MAP_DIR / filename(index, challenge)
        write_wad(path, build_lumps(challenge))
        rows.append(map_record(index, challenge, path))
    manifest = {"split": "diagnostic", "license": "CC0-1.0 map geometry; Freedoom supplies game assets",
                "maps": rows}
    (MAP_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verified_maps():
    manifest = json.loads((MAP_DIR / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["maps"]
    if manifest["split"] != "diagnostic" or len(rows) != len(CHALLENGES):
        raise ValueError("Unexpected diagnostic map manifest")
    hashes = set()
    for index, (row, challenge) in enumerate(zip(rows, CHALLENGES), 1):
        path = MAP_DIR / filename(index, challenge)
        lumps = read_wad(path)
        if (row != map_record(index, challenge, path) or row["sha256"] in hashes
                or list_maps(lumps) != ["MAP01"] or lumps != build_lumps(challenge)):
            raise ValueError(f"Diagnostic WAD does not match its catalog and manifest: {path}")
        info = load_mapinfo(path, "MAP01")
        expected_locks = sorted(DOORS[s] for s in challenge.cells().values() if s in DOORS)
        if info.num_sectors != len(challenge.cells()) or sorted(info.door_sectors.values()) != expected_locks:
            raise ValueError(f"Wrapper geometry does not match the layout: {path}")
        hashes.add(row["sha256"])
    return rows


if __name__ == "__main__":
    build()
    print(f"Generated and verified {len(verified_maps())} WADs in {MAP_DIR}")
