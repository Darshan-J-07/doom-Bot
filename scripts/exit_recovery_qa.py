"""Check exit-recovery maps with legal diagnostic routes in the standard engine."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import atan2, degrees, dist, isclose
from pathlib import Path

from doom_env import make_env
from doom_env.spaces import FRAME_SKIP
from scripts.campaign_map_qa import route
from scripts.exit_switch_qa import SwitchWalk, midline, run_fixture, static_check
from training.exit_recovery_maps import distance_to_wall, prior_identities
from training.universe import neighbors


class RecoveryWalk(SwitchWalk):
    """Record clearance and crossings observed during the fixture route."""

    def __init__(self, env, row, door, seed):
        super().__init__(env, row, seed)
        self.checked_door = door
        self.crossed = False
        self.maximum_gap = self.gap(door) if door is not None else None

    def step(self, *buttons):
        super().step(*buttons)
        if self.checked_door is not None:
            self.crossed |= self.cell() == self.checked_door
            self.maximum_gap = max(self.maximum_gap, self.gap(self.checked_door))

    def move(self, position, tolerance=18):
        if self.row["exit_type"] == "walkover" and position == self.center(self.row["exit_cell"]):
            tolerance = min(tolerance, self.width / 3, self.height / 3)
        super().move(position, tolerance)

    def face_point(self, point):
        x, y = self.position()
        self.face(degrees(atan2(point[1] - y, point[0] - x)))

    def use(self, earlier, target, *, locked=False):
        if self.gap(target) >= 56:
            return
        first, second = self.center(earlier), self.center(target)
        boundary = tuple((a + b) / 2 for a, b in zip(first, second))
        for _ in range(3):
            self.approach_boundary(earlier, target)
            self.settled()
            self.face_point(boundary)
            self.step("USE")
            for _ in range(20):
                if self.gap(target) >= 56:
                    if locked:
                        raise ValueError("Lock moved before key collection")
                    return
                self.step()
            if locked and self.gap(target) == 0:
                return
        raise ValueError("Expected a fresh USE edge to open the approach door")

    def finish(self):
        goals = [tuple(goal) for goal in self.row["drill"]["targets"]]
        for index, target in enumerate(goals):
            self.follow(route(self.layout, self.cell(), target, goals[:index + 1]))
        if self.row["exit_type"] == "switch":
            wall = midline(self.row["exit_segment"])
            approach = tuple(self.row["exit_approach"])
            for _ in range(4):
                self.move(approach, 12)
                self.settled()
                if dist(self.position(), wall) <= 62:
                    self.face_point(wall)
                    self.step("USE")
                    if self.state["exit_reached"]:
                        break
        self.check_exit()
        return self.receipt()


def check_route(row, seed=0):
    open_door = row.get("open_door_cell")
    full = row["task"] in {"keycard", "gauntlet"}
    door = tuple(row["drill"]["targets"][-2]) if full else (tuple(open_door) if open_door else None)
    with make_env(row["path"], row["map_name"], episode_ticks=row["ticks"], render_buffers=False) as env:
        walker = RecoveryWalk(env, row, door, seed)
        start = walker.position()
        assert all(isclose(a, b, abs_tol=1e-6) for a, b in zip(start, row["expected_start"], strict=True))
        assert not walker.state["keys"].any()
        adjacent = sum(bool(walker.layout.get(cell)) for cell in neighbors(walker.cell()))
        if row["level"] == 2:
            assert adjacent == 1, "Recovery exercise must start in a dead end"
        distance = distance_to_wall(start, row["exit_segment"]) if row["exit_segment"] else None
        if distance is not None:
            assert isclose(distance, row["initial_switch_distance"], abs_tol=1e-6)
            assert bool(distance <= 64) == row["starts_in_use_range"] == (row["level"] == 0)

        clearance, lock = None, None
        if door is not None:
            clearance = float(walker.gap(door))
            lock = int(walker.state["door_lock"][walker.sectors[door]])
            if full:
                assert lock > 0 and clearance == 0, "Full episode needs an initially closed keyed lock"
            else:
                assert lock == 0 and clearance == 124, "Conditional post-lock passage is not pre-open and unlocked"
                assert walker.layout[door] == "." and row["post_lock"]

        initial_angle = float(walker.state["angle"])
        receipt = walker.finish()
        if door is not None:
            assert walker.crossed and walker.maximum_gap >= 56, "The legal route did not cross the expected doorway"
        if full:
            assert receipt["keycards"] >= 1
        if row["exit_type"] == "switch":
            assert "USE" in walker.last_buttons
        return dict(receipt, engine_seed=seed, start_position=list(start), start_angle=initial_angle,
                    initial_switch_distance=distance, starts_in_use_range=bool(distance is not None and distance <= 64),
                    start_adjacent_tiles=adjacent, door_initial_clearance=clearance, door_initial_lock=lock,
                    door_crossed=walker.crossed if door is not None else None,
                    door_maximum_clearance=walker.maximum_gap)


def guarded_route(row):
    if row["task"] != "gauntlet":
        return check_route(row)
    for seed in (0, 11, 23, 37):
        try:
            return check_route(row, seed)
        except RuntimeError as failure:
            if "health" not in str(failure) or seed == 37:
                raise


def audit(root, output):
    names = ("fixtures.json", "manifest.json", "final.json")
    source = {name: (root / name).read_bytes() for name in names}
    rows = {name: json.loads(blob)["maps"] for name, blob in source.items()}
    scored = rows["manifest.json"] + rows["final.json"]
    old_geometry, old_wads, _ = prior_identities()
    for key, old in (("geometry_sha256", old_geometry), ("sha256", old_wads)):
        identities = {row[key] for row in scored}
        assert len(identities) == len(scored) and not identities & old
    ownership = {}
    for row in scored:
        assert ownership.setdefault(row["family"], row["split"]) == row["split"]

    errors, fixtures, routes = [], {}, []
    for row in rows["fixtures.json"] + scored:
        try:
            static_check(row)
            if row.get("fixture"):
                fixtures[row["id"]] = run_fixture(row)
                continue
            routes.append(dict(id=row["id"], family=row["family"], level=row["level"], split=row["split"],
                               task=row["task"], exit_type=row["exit_type"], **guarded_route(row)))
        except (AssertionError, RuntimeError, ValueError) as error:
            errors.append(dict(id=row["id"], exception=type(error).__name__, detail=str(error)))
            print(f"FAIL {row['id']}: {type(error).__name__}: {error}", flush=True)

    result = dict(approved=not errors, expected_frame_skip=FRAME_SKIP,
                  manifests={name: sha256(blob).hexdigest() for name, blob in source.items()},
                  maps_reached=dict(Counter(row["split"] for row in routes)),
                  fixtures=fixtures, reference_routes=routes, errors=errors)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("approved", "expected_frame_skip", "manifests",
                                                    "maps_reached", "errors")}), flush=True)
    return not errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if audit(args.maps, args.out) else 1)
