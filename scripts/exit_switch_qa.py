"""Exercise real exit fixtures with legal, map-specific harness controls."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import atan2, degrees, dist
from pathlib import Path

import numpy as np

from doom_env import make_env
from doom_env.mapinfo import parse_udmf
from doom_env.spaces import BUTTONS, FRAME_SKIP
from doom_env.wad import map_lumps, read_wad
from scripts.campaign_map_qa import ReferenceWalk, route
from training.exit_switch_maps import historical_geometry, wall_segments
from training.universe import layout_digest, verify_layout


def vector(buttons):
    return np.array([button in buttons for button in BUTTONS], dtype=np.int8)


def midline(segment):
    return tuple((a + b) / 2 for a, b in zip(*segment))


def wall_pose(segment, margin):
    (x1, y1), (x2, y2) = segment
    x, y = midline(segment)
    length = dist((x1, y1), (x2, y2))
    dx, dy = (y1 - y2) / length, (x2 - x1) / length
    return (x - margin * dx, y - margin * dy), degrees(atan2(dy, dx))


class SwitchWalk(ReferenceWalk):
    def __init__(self, env, row, seed=0):
        super().__init__(env, row, seed)
        self.last_buttons = ()
        self.last_result = None

    def step(self, *buttons):
        action = vector(buttons)
        assert self.env.action_space.contains(action)
        previous = self.state
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        assert self.env.observation_space.contains(self.obs)
        self.state = self.env.privileged_state()
        self.clock.advance(self.state["tic"], exit_reached=self.state["exit_reached"])
        self.decisions += 1
        self.last_buttons = buttons
        self.last_result = dict(reward=reward, terminated=terminated, truncated=truncated,
                                wrapper_exit=info["exit_reached"], state_exit=self.state["exit_reached"])
        if not self.state["finished"]:
            assert self.state["tic"] - previous["tic"] == FRAME_SKIP
            assert self.env._game.get_last_action() == action.tolist()
        if self.state["timeout"] or self.state["dead"]:
            raise RuntimeError(f"Diagnostic route ended at tic {self.clock.elapsed}, health {self.state['health']}")

    def settled(self):
        for _ in range(4):
            self.step()

    def finish(self):
        goals = [tuple(xy) for xy in self.row["drill"]["targets"]]
        for index, target in enumerate(goals):
            self.follow(route(self.layout, self.cell(), target, goals[:index + 1]))
        if self.row["exit_type"] == "switch":
            approach = tuple(self.row["exit_approach"])
            _, angle = wall_pose(self.row["exit_segment"], 26)
            if not self.state["exit_reached"]:
                self.move(approach, 12)
                self.face(angle)
                if "USE" in self.last_buttons:
                    self.step()
                self.step("USE")
        self.check_exit()
        return self.receipt()

    def check_exit(self):
        assert self.state["exit_reached"] and not self.state["timeout"] and not self.state["dead"]
        assert self.last_result["terminated"] and not self.last_result["truncated"]
        assert self.last_result["wrapper_exit"] and self.last_result["state_exit"]
        assert self.last_result["reward"] >= 1.

    def receipt(self):
        return dict(exit=True, decisions=self.decisions, tics=self.clock.elapsed, estimated_tics=self.clock.estimated,
                    keycards=int(self.state["keys"].sum()), health=self.state["health"], damage=self.state["damage_taken"],
                    terminal=self.last_result)


def static_check(row):
    path = Path(row["path"])
    assert sha256(path.read_bytes()).hexdigest() == row["sha256"]
    cells = {(x, y): glyph for x, y, glyph in row["drill"]["cells"]}
    assert layout_digest(cells, labeled=False) == row["geometry_sha256"]
    verify_layout(cells, [tuple(xy) for xy in row["drill"]["targets"]], row["exit_type"] == "switch")
    blocks = parse_udmf(map_lumps(read_wad(path), row["map_name"])["TEXTMAP"].decode())
    exit_lines = [line for line in blocks["linedef"] if line.get("special") == 243]
    if row["exit_type"] == "walkover":
        assert exit_lines and all(line.get("playercross") and not line.get("playeruse") for line in exit_lines)
    else:
        assert len(exit_lines) == 1 and exit_lines[0]["playeruse"] and "sideback" not in exit_lines[0]
        xys = [(v["x"], v["y"]) for v in blocks["vertex"]]
        assert list(map(tuple, row["exit_segment"])) == [xys[exit_lines[0][f"v{i}"]] for i in (1, 2)]
    return len(exit_lines)


def approach_wall(driver, segment, margin=26):
    lookup = {edge: cell for cell, _, edge in wall_segments(driver.layout, driver.row["tile"])}
    target = lookup[tuple(map(tuple, segment))]
    driver.follow(route(driver.layout, driver.cell(), target, []))
    position, angle = wall_pose(segment, margin)
    driver.move(position, 10)
    driver.face(angle)
    driver.settled()


def distance_and_facing(env, row):
    driver = SwitchWalk(env, row)
    segment = row["exit_segment"]
    center = midline(segment)
    _, angle = wall_pose(segment, 26)
    driver.face(angle)
    far = dist(driver.position(), center)
    assert far > 64
    driver.step("USE")
    assert not driver.state["finished"]

    driver.follow(route(driver.layout, driver.cell(), tuple(row["exit_cell"]), []))
    position, _ = wall_pose(segment, 26)
    driver.move(position, 10)
    driver.face(angle + 180)
    driver.settled()
    near = dist(driver.position(), center)
    assert near < 64
    driver.step("USE")
    assert not driver.state["finished"]
    driver.face(angle)
    driver.settled()
    assert not driver.state["finished"]
    facing = driver.state["angle"]
    driver.step("USE")
    driver.check_exit()
    return dict(far_facing_use=dict(distance=far, exited=False),
                near_wrong_facing_use=dict(distance=near, exited=False),
                near_correct_facing_use=dict(distance=dist(driver.position(), center), facing=facing,
                                             exited=True, expected_facing=angle),
                entered_exit_cell_without_use=True, route=driver.receipt())


def obstruction_and_use_hold(env, row):
    driver = SwitchWalk(env, row)
    driver.settled()
    door = next(cell for cell, glyph in driver.layout.items() if glyph == "D")
    center = midline(row["exit_segment"])
    distance = dist(driver.position(), center)
    gap_before = float(driver.gap(door))
    assert distance <= 64.001 and gap_before == 0
    driver.step("USE")
    obstruction = dict(distance=distance, before_gap=gap_before, after_gap=float(driver.gap(door)),
                       no_exit_on_door_use=not driver.state["finished"])
    assert 0 < obstruction["after_gap"] < 56 and obstruction["no_exit_on_door_use"]
    gaps = [obstruction["after_gap"]]
    for _ in range(20):
        if gaps[-1] >= 56:
            break
        driver.step("USE")
        gaps.append(float(driver.gap(door)))
    assert gaps[-1] >= 56 and gaps == sorted(set(gaps))
    for _ in range(20):
        if driver.position()[0] > center[0] - 28:
            break
        driver.step("MOVE_FORWARD", "USE")
        assert not driver.state["finished"]
    held_distance = dist(driver.position(), center)
    assert held_distance < 30 and not driver.state["finished"]
    driver.step()
    assert not driver.state["finished"]
    driver.step("USE")
    driver.check_exit()
    return dict(obstruction=obstruction, held_use_opening_gaps=gaps, held_use_distance=held_distance,
                held_use_no_exit=True, release_repress_exit=True, route=driver.receipt())


def distractor_response(env, row):
    segment = row["distractor_segments"][0]
    frames = {}
    for button in ((), ("USE",)):
        driver = SwitchWalk(env, row)
        approach_wall(driver, segment)
        driver.step(*button)
        assert not driver.state["finished"]
        frames[bool(button)] = driver.obs["rgb"]
    change = np.count_nonzero(frames[True] != frames[False])
    assert change > 0
    return dict(use_does_not_exit=True, changed_pixel_components=int(change))


def run_fixture(row):
    with make_env(row["path"], episode_ticks=row["ticks"], render_buffers=True) as env:
        name = row["fixture"]
        if name == "front-switch":
            checks = distance_and_facing(env, row)
            checks["distractor"] = distractor_response(env, row)
            return checks
        if name == "use-edge":
            return obstruction_and_use_hold(env, row)
        driver = SwitchWalk(env, row)
        receipt = driver.finish()
        if name == "walkover":
            assert all(glyph not in "DRBY" for glyph in driver.layout.values())
            assert "USE" not in driver.last_buttons
        return dict(route=receipt, last_buttons=list(driver.last_buttons))


def audit(root, out, fixtures_only=False):
    manifests = ("fixtures.json",) if fixtures_only else ("fixtures.json", "manifest.json", "final.json")
    parts = {name: json.loads((root / name).read_text())["maps"] for name in manifests}
    unseen = historical_geometry()
    generated = [row for name, rows in parts.items() if name != "fixtures.json" for row in rows]
    structural = {row["geometry_sha256"] for row in generated}
    assert len(structural) == len(generated) and not structural & unseen
    assert all(len({row["split"] for row in generated if row["family"] == family}) == 1
               for family in {row["family"] for row in generated})
    fixture_results, routes, errors = {}, [], []
    for row in (row for rows in parts.values() for row in rows):
        static_check(row)
        name = row["id"]
        try:
            if row.get("fixture"):
                fixture_results[name] = run_fixture(row)
            else:
                with make_env(row["path"], episode_ticks=row["ticks"], render_buffers=False) as env:
                    routes.append(dict(id=name, split=row["split"], family=row["family"], task=row["task"],
                                       **SwitchWalk(env, row).finish()))
        except (AssertionError, RuntimeError, ValueError) as error:
            errors.append(dict(id=name, message=str(error), exception=type(error).__name__))
            print(f"FAIL {name}: {type(error).__name__}: {error}", flush=True)
    result = dict(approved=not errors, expected_frame_skip=FRAME_SKIP,
                  manifests={name: sha256((root / name).read_bytes()).hexdigest() for name in manifests},
                  fixtures=fixture_results, maps_reached=dict(Counter(row["split"] for row in routes)),
                  reference_routes=routes, errors=errors)
    with out.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("fixtures", "reference_routes")}, indent=2), flush=True)
    return not errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixtures-only", action="store_true")
    args = parser.parse_args()
    raise SystemExit(0 if audit(args.maps, args.out, args.fixtures_only) else 1)
