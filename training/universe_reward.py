"""Finite-horizon objective rewards for the procedural training maps."""
from __future__ import annotations

from functools import lru_cache
from math import exp, floor, hypot

from training.rewards import EpisodeMetrics
from training.skills import KEY_INDEX, NEIGHBORS, distance_field


@lru_cache(maxsize=256)
def route_fields(cells, targets):
    return [distance_field(cells, target) for target in targets]


class UniverseReward(EpisodeMetrics):
    def __init__(self, spec, gamma, *, potential_weight=1., novelty_weight=.06):
        self.layout = {(x, y): glyph for x, y, glyph in spec["drill"]["cells"]}
        self.targets = tuple(tuple(cell) for cell in spec["drill"]["targets"])
        self.fields = route_fields(tuple(sorted(self.layout)), self.targets)
        self.sectors = {cell: i for i, cell in enumerate(sorted(self.layout))}
        self.minimums = spec["drill"]["minimums"]
        self.kill_lesson = spec["drill"]["kill_lesson"]
        self.tile_width, self.tile_height = spec["tile"]
        self.tile_distance = (self.tile_width + self.tile_height) / 2
        self.gamma = gamma
        self.potential_weight = potential_weight
        self.novelty_weight = novelty_weight

    def cell(self, state):
        return floor(state["x"] / self.tile_width), floor(state["y"] / self.tile_height)

    def distance(self, state):
        cell = self.cell(state)
        field = self.fields[self.target]
        steps = field.get(cell)
        if steps is None:
            return 8 * self.tile_distance
        if steps == 0:
            choices = (cell,)
        else:
            x, y = cell
            choices = ((x + dx, y + dy) for dx, dy in NEIGHBORS if field.get((x + dx, y + dy)) == steps - 1)
        return max(0, steps - 1) * self.tile_distance + min(
            hypot(state["x"] - (x + .5) * self.tile_width, state["y"] - (y + .5) * self.tile_height)
            for x, y in choices)

    def potential(self, state):
        # Bounded distance term with a stage offset when a key or door milestone changes the target.
        return 3. * (self.target + exp(-self.distance(state) / (2 * self.tile_distance)))

    def reset(self, state):
        super().reset(state)
        self.target = 0
        self.last_potential = self.potential(state)
        self.stalled = 0
        self.max_kills = state["kills"]
        self.lesson_complete = self.objective_complete = False
        self.health_gained = self.ammo_gained = self.novelty_paid = 0

    def step(self, state, action):
        previous = self.previous
        old_doors, old_locks, old_cells = len(self.opened), len(self.unlocked), len(self.visited)
        old_score = super().step(state)
        new_doors, new_locks = len(self.opened) - old_doors, len(self.unlocked) - old_locks
        new_keys = max(0, int(state["keys"].sum()) - int(previous["keys"].sum()))
        kills = max(0, state["kills"] - self.max_kills)
        self.max_kills = max(self.max_kills, state["kills"])
        damage = max(0., state["damage_taken"] - previous["damage_taken"])
        dealt = max(0., state["damage_dealt"] - previous["damage_dealt"])
        self.health_gained += max(0., state["health"] - previous["health"])
        ammo_delta = [n - p for n, p in zip(state["ammo_counts"], previous["ammo_counts"])]
        self.ammo_gained += sum(max(0, value) for value in ammo_delta)
        spent = sum(max(0, -value) for value in ammo_delta)
        while self.target < len(self.targets) - 1:
            cell = self.targets[self.target]
            glyph = self.layout[cell]
            if glyph in KEY_INDEX:
                reached = state["keys"][KEY_INDEX[glyph]]
            else:
                reached = self.sectors[cell] in self.opened
            if not reached:
                break
            self.target += 1

        values = dict(kills=self.max_kills, keycards=int(state["keys"].sum()),
                      doors=len(self.opened), locks=len(self.unlocked))
        met = all(values[key] >= value for key, value in self.minimums.items())
        self.lesson_complete = self.kill_lesson and met and not state["dead"]
        self.objective_complete = bool(met and (state["exit_reached"] or self.lesson_complete) and not state["dead"])
        done = state["finished"] or self.lesson_complete
        current_potential = 0. if done else self.potential(state)
        shaping = self.gamma * current_potential - self.last_potential
        self.last_potential = current_potential

        new_tile = len(self.visited) > old_cells and self.cell(state) in self.layout
        novelty = new_tile and self.novelty_paid < 24
        self.novelty_paid += int(novelty)
        self.stalled = 0 if new_tile or new_keys or new_doors or dealt else self.stalled + 1
        stuck = any(action[:4]) and hypot(state["x"] - previous["x"], state["y"] - previous["y"]) < 2
        reward = (12 * self.objective_complete + 2 * (state["exit_reached"] and not met)
                  - 7 * state["dead"] - 2 * state["timeout"]
                  + 3 * new_keys + 2 * new_locks + 1.4 * new_doors + 5 * kills + .028 * min(dealt, 100)
                  - .04 * damage - .045 * spent - .012 * action[8] - .008
                  - .023 * (stuck and not new_doors) - .018 * (self.stalled > 42)
                  + self.novelty_weight * novelty + self.potential_weight * shaping)
        self.total_reward += reward - old_score
        return float(reward)

    def summary(self):
        progress = len(self.visited & self.layout.keys()) / len(self.layout)
        return dict(super().summary(), drill_success=bool(self.objective_complete),
                    kills=self.max_kills, timeout=bool(self.previous["timeout"]),
                    health_gained=self.health_gained, ammo_gained=self.ammo_gained,
                    coverage=progress, targets_achieved=self.target + int(self.previous["exit_reached"]))
