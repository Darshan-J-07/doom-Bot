"""Check course topology and legal engine controls, separately from learned agents."""
from __future__ import annotations

import argparse
from collections import deque
import json
from math import atan2, degrees, dist, floor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from doom_env import make_env
from doom_env.spaces import BUTTONS, FRAME_SKIP, VITALS_INDEX
from doom_env.wad import write_wad
from examination.maps import Challenge, build_lumps
from vizdoom import GameVariable as GV

from training.episode_clock import EpisodeClock
from training.skills import KEY_INDEX
from training.universe import neighbors, verify_layout


def route(layout, start, goal, allowed, avoid_keys=False):
    parents = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if cell == goal:
            path = []
            while parents[cell] is not None:
                path.append(cell)
                cell = parents[cell]
            return path[::-1]
        for other in neighbors(cell):
            glyph = layout.get(other)
            if other in parents or glyph is None or glyph == "E" and other != goal:
                continue
            if ((glyph in "DRBY" and other not in allowed) or (avoid_keys and glyph in KEY_INDEX)):
                continue
            parents[other] = cell
            queue.append(other)
    raise ValueError(f"No reference path to {goal}")


class ReferenceWalk:
    """Uses authored waypoints solely to check expected WAD events in the engine."""

    def __init__(self, env, row, seed):
        self.env = env
        self.obs, _ = env.reset(seed=seed)
        self.state = env.privileged_state()
        self.clock = EpisodeClock(self.state["tic"], env.episode_ticks)
        self.decisions = 0
        self.row = row
        self.width, self.height = row["tile"]
        self.layout = {(x,y): glyph for x,y,glyph in row["drill"]["cells"]}
        self.sectors = {cell: index for index, cell in enumerate(sorted(self.layout))}

    def position(self):
        return self.state["x"], self.state["y"]

    def cell(self):
        x,y = self.position()
        return floor(x / self.width), floor(y / self.height)

    def center(self, cell):
        return (cell[0] + .5) * self.width, (cell[1] + .5) * self.height

    def step(self, *buttons):
        action = np.zeros(len(BUTTONS), dtype=np.int8)
        for button in buttons:
            action[BUTTONS.index(button)] = 1
        assert self.env.action_space.contains(action)
        self.obs, _, _, _, _ = self.env.step(action)
        assert self.env.observation_space.contains(self.obs)
        self.state = self.env.privileged_state()
        self.clock.advance(self.state["tic"], exit_reached=self.state["exit_reached"])
        self.decisions += 1
        if self.state["dead"] or self.state["timeout"]:
            raise RuntimeError(f"Reference route {'died' if self.state['dead'] else 'timed out'} at tic {self.clock.elapsed}")

    def turn(self, angle):
        error = (angle - self.state["angle"] + 180) % 360 - 180
        if abs(error) <= 13:
            return None
        return "TURN_LEFT" if error > 0 else "TURN_RIGHT"

    def face(self, angle):
        for _ in range(24):
            button = self.turn(angle)
            if button is None:
                return
            self.step(button)
        raise ValueError("Unable to face fixture boundary")

    def move(self, point, tolerance=18):
        for _ in range(90):
            if self.state["exit_reached"] or dist(self.position(), point) <= tolerance:
                return
            x, y = self.position()
            facing = degrees(atan2(point[1] - y, point[0] - x))
            self.step(self.turn(facing) or "MOVE_FORWARD")
        raise ValueError(f"Fixture could not reach {point}")

    def gap(self, cell):
        sector = self.sectors[cell]
        return self.state["sector_ceiling"][sector] - self.state["sector_floor"][sector]

    def approach_boundary(self, earlier, later):
        x,y = self.center(earlier)
        dx,dy = later[0]-earlier[0], later[1]-earlier[1]
        self.move((x+dx*(self.width/2-26),y+dy*(self.height/2-26)), 12)
        self.face(degrees(atan2(dy, dx)))

    def use(self, earlier, target, *, locked=False):
        if self.gap(target) >= 56:
            return
        self.approach_boundary(earlier, target)
        self.step("USE")
        for _ in range(20):
            if self.gap(target) >= 56:
                if locked:
                    raise ValueError("Lock moved before key collection")
                return
            self.step()
        if not locked or self.gap(target) > 0:
            raise ValueError("Unexpected door response to one USE edge")

    def follow(self, points):
        earlier = self.cell()
        for point in points:
            glyph = self.layout[point]
            if glyph in "DRBY":
                self.use(earlier, point)
            self.move(self.center(point))
            if glyph in KEY_INDEX and not self.state["keys"][KEY_INDEX[glyph]]:
                raise ValueError("Expected key did not enter inventory")
            earlier = point

    def finish(self):
        goals = [tuple(goal) for goal in self.row["drill"]["targets"]]
        for index, target in enumerate(goals):
            self.follow(route(self.layout, self.cell(), target, goals[:index+1]))
        if self.row["drill"]["exit_switch"] and not self.state["exit_reached"]:
            x, y = goals[-1]
            wall = next(cell for cell in ((x-1,y),(x,y+1),(x+1,y),(x,y-1)) if cell not in self.layout)
            self.approach_boundary(goals[-1], wall)
            self.step("USE")
        if not self.state["exit_reached"]:
            raise ValueError("Walkover or switch did not end the map")
        return dict(decisions=self.decisions, tics=self.clock.elapsed, estimate=self.clock.estimated,
                    keys=int(self.state["keys"].sum()), health=self.state["health"])


def controls():
    with TemporaryDirectory(prefix="doom-actions-") as folder:
        path = Path(folder) / "controls.wad"
        write_wad(path, build_lumps(Challenge("controls", "controls", "Open room with a centered spawn",
                                              "....E\n.....\n..P..\n.....\n.....")))
        result = measure_controls(path)
        with make_env(path, episode_ticks=200, render_buffers=False) as env:
            env.reset(seed=0)
            idle = np.zeros(9, np.int8)
            switch = idle.copy()
            switch[8] = 1
            for _ in range(5):
                env.step(idle)
            initial = int(env._game.get_game_variable(GV.SELECTED_WEAPON))
            cycles = []
            for _ in range(2):
                timeline = []
                for _ in range(12):
                    env.step(switch)
                    timeline.append(int(env._game.get_game_variable(GV.SELECTED_WEAPON)))
                cycles.append(timeline)
                env.step(idle)
            result["weapon_hold_test"] = dict(initial=initial, through_held_input=cycles,
                                              first_changed=cycles[0][-1] != initial,
                                              repress_changed=cycles[1][-1] != cycles[0][-1])
        return result


def measure_controls(path):
    with make_env(path, episode_ticks=60, render_buffers=False) as env:
        output = {}
        for button in BUTTONS[:6]:
            env.reset(seed=0)
            for _ in range(5):
                env.step(np.zeros(9, np.int8))
            before = env.privileged_state()
            action = np.zeros(9, np.int8)
            action[BUTTONS.index(button)] = 1
            obs, _, term, trunc, _ = env.step(action)
            after = env.privileged_state()
            assert not term and not trunc and after["tic"]-before["tic"] == FRAME_SKIP
            assert env._game.get_last_action() == action.tolist()
            output[button] = dict(forward=float(obs["vitals"][VITALS_INDEX["vel_forward"]]),
                                  right=float(obs["vitals"][VITALS_INDEX["vel_right"]]),
                                  turn=(after["angle"]-before["angle"]+180)%360-180, tics=FRAME_SKIP)
        assert output["MOVE_FORWARD"]["forward"] > 0 > output["MOVE_BACKWARD"]["forward"], output
        assert output["MOVE_RIGHT"]["right"] > 0 > output["MOVE_LEFT"]["right"], output
        assert output["TURN_LEFT"]["turn"] > 0 > output["TURN_RIGHT"]["turn"], output
        old = obs
        while not trunc:
            old = obs
            obs, _, term, trunc, _ = env.step(np.zeros(9,np.int8))
        assert obs is old
        return dict(buttons=output, timeout_returns_previous_observation=True)


def audit(root, output):
    manifest = json.loads((root / "manifest.json").read_text())["maps"]
    final = json.loads((root / "final.json").read_text())["maps"]
    probe = json.loads((root / "probe.json").read_text())["maps"]
    result = dict(controls=controls(), valid_static=0, engine_smoke=[], reference_routes=[])
    for row in (*manifest, *final, *probe):
        if "drill" in row:
            drill = row["drill"]
            verify_layout({(x,y):g for x,y,g in drill["cells"]}, [tuple(cell) for cell in drill["targets"]], drill["exit_switch"])
            result["valid_static"] += 1
    seen = set()
    for row in (*final, *probe, *manifest):
        family = (row["split"], row["family"])
        if row["split"] == "train" and family in seen:
            continue
        seen.add(family)
        with make_env(row["path"], row["map_name"], episode_ticks=80, render_buffers=False) as env:
            obs, _ = env.reset(seed=0)
            assert env.observation_space.contains(obs)
            obs, _, _, _, _ = env.step(np.zeros(9, np.int8))
            assert env.observation_space.contains(obs)
        result["engine_smoke"].append(dict(id=row["id"], split=row["split"], sha256=row["sha256"]))
        if "drill" in row and row["stage"] not in ("combat", "gauntlet"):
            with make_env(row["path"], row["map_name"], episode_ticks=4200, render_buffers=False) as env:
                driver = ReferenceWalk(env, row, 0)
                evidence = dict(id=row["id"], split=row["split"], sha256=row["sha256"])
                try:
                    if row["family"] == "single-door" and "door_use_edge" not in result:
                        door = next(point for point, glyph in driver.layout.items() if glyph == "D")
                        approaches = [point for point in neighbors(door) if point in driver.layout]
                        reachable = []
                        for approach in approaches:
                            try:
                                reachable.append((approach, route(driver.layout, driver.cell(), approach, [])))
                            except ValueError:
                                continue
                        if reachable:
                            approach, path = min(reachable, key=lambda item: len(item[1]))
                            driver.follow(path)
                            driver.approach_boundary(approach, door)
                            held = []
                            for _ in range(4):
                                driver.step("USE")
                                held.append(driver.gap(door))
                            driver.step()
                            before = driver.gap(door)
                            driver.step("USE")
                            result["door_use_edge"] = dict(held=held, before_repress=before, after_repress=driver.gap(door))
                            driver = ReferenceWalk(env, row, 0)
                    if row["family"] == "side-key" and result.get("lock_denial") is None:
                        lock = next(cell for cell,glyph in driver.layout.items() if glyph in "RBY")
                        for side in neighbors(lock):
                            if side not in driver.layout or driver.layout[side] in "DRBYrbyE":
                                continue
                            try:
                                before_key = route(driver.layout, driver.cell(), side, [], avoid_keys=True)
                            except ValueError:
                                continue
                            driver.follow(before_key)
                            driver.use(side, lock, locked=True)
                            result["lock_denial"] = dict(wad=row["path"], open_height=driver.gap(lock), has_key=False)
                            driver = ReferenceWalk(env, row, 0)
                            break
                    evidence.update(driver.finish(), status="exit")
                except (RuntimeError, ValueError) as error:
                    evidence.update(status="unwitnessed", error=str(error))
                result["reference_routes"].append(evidence)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"static": result["valid_static"], "smoke": len(result["engine_smoke"]),
                      "route_exits": sum(row["status"] == "exit" for row in result["reference_routes"]),
                      "routes": len(result["reference_routes"]), "lock_denial": result.get("lock_denial")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.map_root, args.out)
