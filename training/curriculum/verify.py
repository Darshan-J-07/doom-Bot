"""Smoke-test curriculum WADs and solve representative maps in ViZDoom.

The route driver uses authored geometry and diagnostic engine state. It never
supplies training data or actions to the submitted agent.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import json
import math
from pathlib import Path

import numpy as np

from doom_env import make_env, spaces
from doom_env.mapinfo import LOCK_BLUE, LOCK_RED, LOCK_YELLOW
from training.maps import load_verified_maps
from training.rewards import EpisodeMetrics

from .generate import build_layout


NEIGHBORS = ((1, 0), (0, 1), (-1, 0), (0, -1))
LOCKS = {"R": (LOCK_RED, 2), "B": (LOCK_BLUE, 0), "Y": (LOCK_YELLOW, 1)}
DOORS = {"D", *LOCKS}
KEYS = {role.lower() for role in LOCKS}
BUTTON_INDEX = {name: index for index, name in enumerate(spaces.BUTTONS)}
PROBE_LAYOUTS = (
    "first-steps", "single-target", "first-door", "key-before-lock",
    "blue-key-detour", "yellow-key-detour", "two-lock-practice",
)


def action(button=None):
    result = np.zeros(len(spaces.BUTTONS), dtype=np.int8)
    if button is not None:
        result[BUTTON_INDEX[button]] = 1
    return result


def destination(tiles, role):
    return next(cell for cell, value in tiles.items() if value == role)


def route(tiles, target, collect_keys=True):
    """Find a diagnostic route from spawn with key inventory in the graph."""
    start = destination(tiles, "S")
    pending = deque([(start, frozenset(), (start,))])
    visited = {(start, frozenset())}
    while pending:
        cell, held, path = pending.popleft()
        if cell == target:
            return path
        for dx, dy in NEIGHBORS:
            neighbor = (cell[0] + dx, cell[1] + dy)
            role = tiles.get(neighbor)
            if role is None or (role == "G" and neighbor != target):
                continue
            if role in LOCKS and role.lower() not in held:
                continue
            if role in KEYS and not collect_keys:
                continue
            keys = held | {role} if role in KEYS else held
            node = (neighbor, keys)
            if node not in visited:
                visited.add(node)
                pending.append((neighbor, keys, (*path, neighbor)))
    return None


def smoke(row):
    env = make_env(row["path"], row.get("map_name", "MAP01"),
                   episode_ticks=40, render_buffers=False)
    try:
        obs, info = env.reset(seed=row["seed"])
        if not env.observation_space.contains(obs):
            raise RuntimeError("Initial observation is outside the official space")
        initial_tic = info["tic"]
        decisions = 0
        while True:
            obs, reward, terminated, truncated, info = env.step(action())
            decisions += 1
            if not env.observation_space.contains(obs) or not math.isfinite(reward):
                raise RuntimeError("Step returned an invalid observation or reward")
            if terminated or truncated:
                break
            if decisions > 11:
                raise RuntimeError("Short engine episode exceeded its tick cap")
        if info["tic"] <= initial_tic:
            raise RuntimeError("Engine time did not advance")
        return decisions
    finally:
        env.close()


class RouteProbe:
    def __init__(self, env, tiles, size, seed):
        self.env = env
        self.tiles = tiles
        self.size = size
        self.sectors = {cell: index for index, cell in enumerate(sorted(tiles))}
        env.reset(seed=seed)
        self.state = env.privileged_state()
        self.decisions = 0
        self.metrics = EpisodeMetrics()
        self.metrics.reset(self.state)

    def step(self, button=None):
        self.env.step(action(button))
        self.decisions += 1
        self.state = self.env.privileged_state()
        self.metrics.step(self.state)
        if self.state["dead"] or self.state["timeout"]:
            raise RuntimeError(f"Diagnostic route terminated at tic {self.state['tic']}")

    def shoot_test_target(self):
        for _ in range(64):
            self.step("ATTACK")
            if self.state["kills"] == 1:
                return
        raise RuntimeError("The corridor target was not defeated")

    def center(self, cell):
        return tuple((coordinate + 0.5) * self.size for coordinate in cell)

    def steering(self, heading):
        error = (heading - self.state["angle"] + 180) % 360 - 180
        if abs(error) <= 12:
            return None
        return "TURN_LEFT" if error > 0 else "TURN_RIGHT"

    def move_to(self, target, tolerance=18):
        for _ in range(120):
            if self.state["exit_reached"]:
                return
            dx = target[0] - self.state["x"]
            dy = target[1] - self.state["y"]
            if math.hypot(dx, dy) <= tolerance:
                return
            heading = math.degrees(math.atan2(dy, dx))
            self.step(self.steering(heading) or "MOVE_FORWARD")
        raise RuntimeError(f"Could not reach {target} from {(self.state['x'], self.state['y'])}")

    def gap(self, cell):
        index = self.sectors[cell]
        return self.state["sector_ceiling"][index] - self.state["sector_floor"][index]

    def use_door(self, approach, door, should_open):
        role = self.tiles[door]
        expected_lock, key_index = LOCKS.get(role, (0, None))
        observed_lock = self.state["door_lock"][self.sectors[door]]
        if observed_lock != expected_lock:
            raise RuntimeError(f"Door {door} has lock {observed_lock}, expected {expected_lock}")
        if key_index is not None and bool(self.state["keys"][key_index]) != should_open:
            raise RuntimeError(f"Inventory did not match the expected lock probe at {door}")
        if should_open and self.gap(door) >= 56:
            return
        dx, dy = (door[0] - approach[0], door[1] - approach[1])
        x, y = self.center(approach)
        offset = self.size / 2 - 28
        self.move_to((x + dx * offset, y + dy * offset), tolerance=12)
        heading = math.degrees(math.atan2(dy, dx))
        for _ in range(40):
            turn = self.steering(heading)
            if turn is None:
                break
            self.step(turn)
        else:
            raise RuntimeError(f"Could not face door {door}")
        self.step("USE")
        for _ in range(24):
            if should_open and self.gap(door) >= 56:
                return
            self.step()
        if should_open:
            raise RuntimeError(f"USE did not open door {door}")
        if self.gap(door) != 0:
            raise RuntimeError(f"Door {door} moved without its key")

    def follow(self, path):
        for previous, cell in zip(path, path[1:]):
            role = self.tiles[cell]
            if role in DOORS:
                self.use_door(previous, cell, should_open=True)
            self.move_to(self.center(cell))
            if role in KEYS and not self.state["keys"][LOCKS[role.upper()][1]]:
                raise RuntimeError(f"Key at {cell} was not collected")


def probe(row, tiles, size, prove_denial=False):
    path = route(tiles, destination(tiles, "G"))
    if path is None:
        raise RuntimeError("Layout has no solvable route with its key placements")
    env = make_env(row["path"], row.get("map_name", "MAP01"),
                   episode_ticks=4200, render_buffers=False)
    try:
        denied = False
        if prove_denial:
            for door, role in tiles.items():
                if role not in LOCKS:
                    continue
                for dx, dy in NEIGHBORS:
                    face = (door[0] + dx, door[1] + dy)
                    before_pickup = route(tiles, face, collect_keys=False)
                    if before_pickup is not None:
                        driver = RouteProbe(env, tiles, size, row["seed"])
                        driver.follow(before_pickup)
                        driver.use_door(face, door, should_open=False)
                        denied = True
                        break
                if denied:
                    break
            if not denied:
                raise RuntimeError("Could not approach the lock before picking up a key")
        driver = RouteProbe(env, tiles, size, row["seed"])
        if row["layout"] == "single-target":
            driver.shoot_test_target()
        driver.follow(path)
        metrics = driver.metrics.summary()
        expected_doors = {cell for cell in path if tiles[cell] in DOORS}
        expected_locks = {cell for cell in path if tiles[cell] in LOCKS}
        if not metrics["exit"] or not driver.state["finished"]:
            raise RuntimeError("Diagnostic route did not reach a real game exit")
        if (metrics["doors"], metrics["locks"]) != (len(expected_doors), len(expected_locks)):
            raise RuntimeError(f"Door and lock counts did not match route: {metrics}")
        fields = ("exit", "survived", "kills", "keys", "doors", "locks", "unique_cells", "health", "damage")
        return {"path": row["path"], "locked_before_pickup": denied, "decisions": driver.decisions,
                **{field: metrics[field] for field in fields}}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-root", default="training/curriculum")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    rows = load_verified_maps(args.map_root)
    decisions = 0
    for row in rows:
        try:
            decisions += smoke(row)
        except Exception as exc:
            raise RuntimeError(f"Engine smoke failed on {row['path']}: {exc}") from exc
    report = {"maps": len(rows), "split_counts": dict(Counter(row["split"] for row in rows)),
              "engine_smoke_decisions": decisions, "route_probes": {}}
    if not args.smoke_only:
        for layout in PROBE_LAYOUTS:
            row = next(row for row in rows if row["split"] == "train" and row["layout"] == layout)
            tiles, tile_size, _heading = build_layout(layout, row["seed"])
            try:
                report["route_probes"][layout] = probe(row, tiles, tile_size, prove_denial=layout.endswith("-detour"))
            except RuntimeError as exc:
                raise RuntimeError(f"{layout} probe on {row['path']}: {exc}") from exc
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
