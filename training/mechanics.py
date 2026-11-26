"""Training-only credit for concrete mechanics on authored courses."""
from __future__ import annotations

from functools import lru_cache
from math import floor, hypot

import numpy as np

from training.rewards import CompetitionReward
from training.skills import KEY_INDEX, NEIGHBORS, distance_field


@lru_cache(maxsize=256)
def target_fields(tiles, goals):
    return tuple(distance_field(tuple(sorted(cell for cell, glyph in tiles
                                             if glyph not in "DRBY" or cell in goals[:index+1])), goal)
                 for index, goal in enumerate(goals))


class CourseReward(CompetitionReward):
    def __init__(self, spec, *, guided, gamma=.995, trigger_discipline=False):
        drill = spec["drill"]
        self.guided = guided
        self.gamma = gamma
        self.trigger_discipline = trigger_discipline
        self.layout = {(x, y): glyph for x, y, glyph in drill["cells"]}
        self.goals = tuple(tuple(point) for point in drill["targets"])
        self.fields = target_fields(tuple(sorted(self.layout.items())), self.goals) if guided else ()
        self.sectors = {cell: i for i, cell in enumerate(sorted(self.layout))}
        self.minimums = drill["minimums"]
        self.kill_lesson = drill["kill_lesson"]
        self.tile = spec["tile"]
        self.unit = sum(self.tile) / 2

    def cell(self, state):
        return floor(state["x"]/self.tile[0]), floor(state["y"]/self.tile[1])

    def center_distance(self, state, cell):
        return hypot(state["x"] - (cell[0] + .5) * self.tile[0], state["y"] - (cell[1] + .5) * self.tile[1])

    def distance(self, state):
        cell = self.cell(state)
        field = self.fields[self.target]
        length = field.get(cell)
        if length is None:
            return float("inf")
        if length == 0:
            return self.center_distance(state, cell)
        x, y = cell
        choices = [(x+dx, y+dy) for dx, dy in NEIGHBORS if field.get((x+dx,y+dy)) == length-1]
        return (length - 1) * self.unit + min(self.center_distance(state, other) for other in choices)

    @staticmethod
    def door_gaps(state):
        if state["door_lock"] is None:
            return np.zeros(0, dtype=np.float32)
        gaps = state["sector_ceiling"] - state["sector_floor"]
        return np.where(state["door_lock"] >= 0, np.clip(gaps, 0, 56), 0)

    def reset(self, state):
        super().reset(state)
        self.target = 0
        self.best_distance = self.distance(state) if self.guided else 0.
        self.spend = self.dealt = self.guide_return = 0.
        self.prior = np.zeros(9, dtype=np.int8)
        self.peak_dealt = state["damage_dealt"]
        self.peak_gaps = self.door_gaps(state)
        self.credited_route = self.credited_gap = self.credited_damage = 0.
        self.lesson_complete = self.objective_complete = False
        self.no_progress = self.longest_stall = 0
        self.guide_components = dict(route=0., damage_dealt=0., door_motion=0., aim=0., completed_combat=0., wasted_inputs=0.)
        if self.trigger_discipline:
            self.guide_components["unseen_attacks"] = 0.

    def route_credit(self, state):
        if self.kill_lesson:
            return 0.
        goal = self.goals[self.target]
        glyph = self.layout[goal]
        current = self.distance(state)
        advance = max(0., self.best_distance - current) if np.isfinite(self.best_distance) else 0.
        self.best_distance = min(current, self.best_distance)
        delta = min(.22, .72 * advance / self.unit, 4. - self.credited_route)
        self.credited_route += delta
        if self.target < len(self.goals) - 1:
            reached = bool(state["keys"][KEY_INDEX[glyph]]) if glyph in KEY_INDEX else self.sectors[goal] in self.opened
            if reached:
                self.target += 1
                self.best_distance = self.distance(state)
                self.credited_route = 0.
        return delta

    @staticmethod
    def aim_potential(observation):
        entities = observation["visible_entities"]
        types = entities["entity_type"]
        visible = (types >= 1) & (types <= 3) & entities["mask"].astype(bool)
        if not np.any(visible):
            return 0.
        range_units, sine, *_ = entities["features"][np.flatnonzero(visible)[0]]
        # Bound the potential to a short-range, centered view from legal sensors.
        aiming = max(0., 1. - abs(float(sine)) / .70)
        return 2.5 * aiming / (1. + 2. * float(range_units))

    def step(self, state, action, observation, following):
        if self.finished:
            return 0.
        before = self.previous
        cells_before, doors_before = len(self.visited), len(self.opened)
        base = super().step(state)
        extra = dict.fromkeys(self.guide_components, 0.)
        net_ammo = sum(max(0, a-b) for a,b in zip(before["ammo_counts"], state["ammo_counts"]))
        self.spend += net_ammo
        dealt = max(0., state["damage_dealt"] - self.peak_dealt)
        self.peak_dealt += dealt
        self.dealt += dealt
        new_weapon_press = bool(action[8] and not self.prior[8])
        progress = (len(self.visited) > cells_before or len(self.opened) > doors_before or dealt > 0
                    or np.any(state["keys"] > before["keys"]))
        self.no_progress = 0 if progress else self.no_progress + 1
        self.longest_stall = max(self.longest_stall, self.no_progress)
        required_kills = self.minimums.get("kills", 0)
        values = dict(kills=self.max_kills, keycards=int(self.key_flags.sum()), locks=len(self.unlocked), doors=len(self.opened))
        met = all(values[name] >= minimum for name, minimum in self.minimums.items())
        self.lesson_complete = bool(self.kill_lesson and met and not state["dead"])
        self.objective_complete = bool(met and (self.lesson_complete or state["exit_reached"]) and not state["dead"])
        # A dedicated kill drill ends on a kill. Bypassing the opponent is a failed drill.
        drill_award = 8. * self.lesson_complete
        if self.kill_lesson and state["exit_reached"]:
            base -= self.components["completion"]
            self.components["completion"] = 0.
            drill_award -= 4. * (not self.lesson_complete)
        if self.guided:
            gaps = self.door_gaps(state)
            delta = float(np.maximum(0., gaps-self.peak_gaps).sum())
            self.peak_gaps = np.maximum(gaps,self.peak_gaps)
            extra["door_motion"] = min(.017*delta, 3. - self.credited_gap)
            self.credited_gap += extra["door_motion"]
            extra["route"] = self.route_credit(state)
            extra["damage_dealt"] = min(.064 * dealt, 12. - self.credited_damage)
            self.credited_damage += extra["damage_dealt"]
            if self.kill_lesson:
                potential = 0. if self.finished or self.lesson_complete else self.aim_potential(following)
                extra["aim"] = self.gamma * potential - self.aim_potential(observation)
            extra["completed_combat"] = 6. * (required_kills > 0 and self.objective_complete)
            if state["exit_reached"] and required_kills and self.max_kills < required_kills:
                extra["completed_combat"] -= 8.
            entities = observation["visible_entities"]
            types = entities["entity_type"]
            enemy_seen = np.any((types >= 1) & (types <= 3) & entities["mask"].astype(bool))
            # Small costs for ineffective inputs. Genuine damage and successful interactions dominate.
            useless = (.10 * new_weapon_press * (observation["vitals"][3] > 0)
                       + .008 * action[7] * self.prior[7]
                       + .018 * (net_ammo > 0 and not enemy_seen and not dealt)
                       + .010 * (self.no_progress > 46))
            extra["wasted_inputs"] = -min(useless, max(0., 3. + self.guide_components["wasted_inputs"]))
            if self.trigger_discipline and not enemy_seen and not dealt:
                # Test sustained trigger discipline with a separate episode budget.
                # An enemy can enter the shot during the four held tics.
                trigger_cost = .035 * bool(action[6]) + .08 * (net_ammo > 0)
                remaining_cost = max(0., 4. + self.guide_components["unseen_attacks"])
                extra["unseen_attacks"] = -min(trigger_cost, remaining_cost)
        self.prior = np.array(action, copy=True)
        for key, value in extra.items():
            self.guide_components[key] += float(value)
        difference = drill_award + sum(extra.values())
        self.guide_return += float(difference)
        self.total_reward = float(sum(self.components.values())) + self.guide_return
        self.finished = self.finished or self.lesson_complete
        return float(base + difference)

    def summary(self):
        result = super().summary()
        return dict(result, drill_success=self.objective_complete, coverage=len(self.visited & self.layout.keys())/len(self.layout),
                    guide=self.guide_components, damage_dealt=self.dealt, rounds_spent_lower_bound=self.spend,
                    longest_no_progress=self.longest_stall)
