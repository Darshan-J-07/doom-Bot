"""Training-only signals and rewards for short skill drills."""
from __future__ import annotations

from collections import deque
from functools import lru_cache
from math import floor, hypot

import numpy as np
from vizdoom import GameVariable as GV

from examination.maps import TILE
from training.rewards import EpisodeMetrics

NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
KEY_INDEX = {"r": 2, "b": 0, "y": 1}
AMMO_VARIABLES = (GV.AMMO2, GV.AMMO3, GV.AMMO5, GV.AMMO6)


def skill_state(env):
    state = env.privileged_state()
    engine = env._game
    state["damage_dealt"] = float(engine.get_game_variable(GV.DAMAGECOUNT))
    state["ammo_counts"] = tuple(int(engine.get_game_variable(variable)) for variable in AMMO_VARIABLES)
    return state


@lru_cache(maxsize=128)
def distance_field(cells, goal):
    """Tile distances used only to measure progress on authored training maps."""
    cells = set(cells)
    levels = {goal: 0}
    queue = deque([goal])
    while queue:
        x, y = queue.popleft()
        for dx, dy in NEIGHBORS:
            next_cell = (x + dx, y + dy)
            if next_cell in cells and next_cell not in levels:
                levels[next_cell] = levels[x, y] + 1
                queue.append(next_cell)
    return levels


class SkillReward(EpisodeMetrics):
    """Score learned rollouts using goal progress and actual engine events."""

    def __init__(self, spec):
        self.layout = {(x, y): kind for x, y, kind in spec["cells"]}
        cells = tuple(sorted(self.layout))
        self.objectives = [(cell, kind) for kind in "rbyha*E"
                           for cell, glyph in self.layout.items() if glyph == kind
                           and (kind != "h" or "health_gained" in spec["minimums"])
                           and (kind != "a" or "ammo_gained" in spec["minimums"])]
        self.fields = [distance_field(cells, cell) for cell, _ in self.objectives]
        self.minimums = spec["minimums"]
        self.kill_lesson = spec["kill_lesson"]

    def reset(self, state):
        super().reset(state)
        self.health_gained = self.ammo_gained = 0
        self.seen = {self.tile(state)}
        self.target_index = 0
        self.ref_distance = 0.
        self.ref_distance = self.distance(state)
        self.door_peaks = self.gaps(state)
        self.lesson_complete = self.objective_complete = False

    @staticmethod
    def tile(state):
        return floor(state["x"] / TILE), floor(state["y"] / TILE)

    @staticmethod
    def gaps(state):
        if state["door_lock"] is None:
            return np.zeros(0)
        doors = state["door_lock"] >= 0
        return np.maximum(0, state["sector_ceiling"][doors] - state["sector_floor"][doors])

    def distance(self, state):
        tile = self.tile(state)
        goal, _ = self.objectives[self.target_index]
        levels = self.fields[self.target_index]
        steps = levels.get(tile)
        if steps is None:
            return self.ref_distance
        if steps == 0:
            return hypot(state["x"] - (goal[0] + .5) * TILE, state["y"] - (goal[1] + .5) * TILE)
        x, y = tile
        return (steps - 1) * TILE + min(
            hypot(state["x"] - (x + dx + .5) * TILE, state["y"] - (y + dy + .5) * TILE)
            for dx, dy in NEIGHBORS if levels.get((x + dx, y + dy)) == steps - 1
        )

    def reached(self, state, cell, kind):
        if kind in KEY_INDEX:
            return state["keys"][KEY_INDEX[kind]]
        if kind == "h":
            return self.health_gained >= self.minimums.get("health_gained", 1)
        if kind == "a":
            return self.ammo_gained >= self.minimums.get("ammo_gained", 1)
        return cell in self.seen

    def step(self, state):
        previous = self.previous
        doors, locks = len(self.opened), len(self.unlocked)
        legacy_reward = super().step(state)
        new_doors, new_locks = len(self.opened) - doors, len(self.unlocked) - locks
        new_keys = max(0, int(np.sum(state["keys"]) - np.sum(previous["keys"])))
        kills = max(0, state["kills"] - previous["kills"])
        dealt = max(0, state["damage_dealt"] - previous["damage_dealt"])
        damage = max(0, state["damage_taken"] - previous["damage_taken"])
        gained_health = max(0, state["health"] - previous["health"])
        ammo_changes = [new - old for new, old in zip(state["ammo_counts"], previous["ammo_counts"])]
        gained_ammo = sum(max(0, change) for change in ammo_changes)
        used_ammo = sum(max(0, -change) for change in ammo_changes)
        self.health_gained += gained_health
        self.ammo_gained += gained_ammo
        tile = self.tile(state)
        novel = tile in self.layout and tile not in self.seen
        if novel:
            self.seen.add(tile)

        gap = self.gaps(state)
        opened_distance = float(np.maximum(0, gap - self.door_peaks).sum())
        self.door_peaks = np.maximum(self.door_peaks, gap)

        # Only a new best approach earns progress for each target.
        distance = self.distance(state)
        progress = max(0., self.ref_distance - distance)
        self.ref_distance = min(distance, self.ref_distance)
        while self.target_index < len(self.objectives) - 1 and self.reached(state, *self.objectives[self.target_index]):
            self.target_index += 1
            self.ref_distance = self.distance(state)

        values = dict(kills=state["kills"], keycards=int(np.sum(state["keys"])), doors=len(self.opened),
                      locks=len(self.unlocked), health_gained=self.health_gained, ammo_gained=self.ammo_gained,
                      checkpoints=sum(self.layout[cell] == "*" for cell in self.seen if cell in self.layout))
        complete = all(values[name] >= amount for name, amount in self.minimums.items())
        self.lesson_complete = self.kill_lesson and complete and not state["dead"]
        self.objective_complete = (self.lesson_complete or state["exit_reached"] and complete) and not state["dead"]
        terminal = 12 * self.objective_complete - 7 * state["dead"] - 6 * (state["exit_reached"] and not complete)
        progress_weight = .004 if state["kills"] < self.minimums.get("kills", 0) else .022
        reward = (terminal + progress_weight * min(60., progress) + .12 * novel + 5 * new_keys
                  + 1.5 * new_doors + 3 * new_locks + .012 * opened_distance
                  + 7 * kills + .04 * min(dealt, 100) - .045 * damage
                  + .16 * gained_health + .10 * gained_ammo - .035 * used_ammo - .016)
        self.total_reward += reward - legacy_reward
        return float(reward)

    def summary(self):
        return dict(**super().summary(), drill_success=bool(self.objective_complete),
                    health_gained=self.health_gained, ammo_gained=self.ammo_gained,
                    damage_dealt=self.previous["damage_dealt"])
