"""Engine-side episode measurements. No state is passed to the policy."""
from __future__ import annotations

from collections import Counter, deque
from math import dist, floor

import numpy as np
from doom_env.spaces import BUTTONS, FRAME_SKIP
from vizdoom import GameVariable as GV

from training.rewards import EpisodeMetrics


AMMO_VARS = (GV.AMMO2, GV.AMMO3, GV.AMMO5, GV.AMMO6)
LOCK_KEYS = {1: (2, 5), 2: (0, 3), 3: (1, 4)}
FRONT = [62, 63, 0, 1, 2]


def engine_stats(env):
    game = env._game
    return {
        **env.privileged_state(),
        "armor": game.get_game_variable(GV.ARMOR),
        "dealt": game.get_game_variable(GV.DAMAGECOUNT),
        "hits": game.get_game_variable(GV.HITCOUNT),
        "weapon": int(game.get_game_variable(GV.SELECTED_WEAPON)),
        "weapon_ammo": int(game.get_game_variable(GV.SELECTED_WEAPON_AMMO)),
        "ammo": tuple(int(game.get_game_variable(v)) for v in AMMO_VARS),
    }


def door_extents(env):
    extents = {}
    locks = env.mapinfo.door_sectors
    for sector, lines in env.sector_polygons():
        if sector in locks and len(lines):
            extents[sector] = lines
    return extents


def in_polygon(lines, x, y):
    x1, y1, x2, y2 = lines.T
    mask = (y1 > y) != (y2 > y)
    return bool(np.count_nonzero(x < x1[mask] + (y - y1[mask]) * (x2[mask] - x1[mask]) / (y2[mask] - y1[mask])) % 2)


class GameplayObserver(EpisodeMetrics):
    def __init__(self, state, *, door_polygons, spec=None):
        self.reset(state)
        self.door_polygons = door_polygons
        self.actions, self.counts = Counter(), Counter()
        self.buttons = dict.fromkeys(BUTTONS, 0)
        self.maxima = {key: state[key] for key in ("kills", "damage_taken", "dealt", "hits")}
        self.keys = np.asarray(state["keys"], dtype=bool)
        self.key_times = {int(key): 0 for key in np.flatnonzero(self.keys)}
        self.unlock_times = []
        self.traversed = set()
        self.current_door = None
        self.entry = None
        self.previous_action = None
        self.same = self.max_same = self.stalled = self.max_stall = 0
        self.decisions = 0
        self.rotation = deque(maxlen=12)
        self.first_stall = None
        self.recovery_tics = []
        self.first_visited = {self.cell(state): 0}
        if spec and spec.get("tile") and spec.get("drill"):
            symbols = {(x, y): glyph for x, y, glyph in spec["drill"]["cells"]}
            self.dead_ends = {p for p, glyph in symbols.items() if glyph == "." and sum(
                (p[0]+dx, p[1]+dy) in symbols for dx, dy in ((-1,0),(1,0),(0,-1),(0,1))) == 1}
            self.tile = spec["tile"]
        else:
            self.dead_ends = self.tile = None
        self.trapped_since = None

    def track_traversal(self, old, new):
        here = new["x"], new["y"]
        before = old["x"], old["y"]
        sector = next((i for i, lines in self.door_polygons.items() if in_polygon(lines, *here)), None)
        if self.current_door is not None and sector != self.current_door:
            # A full crossing ends on the opposite side of the sector's center.
            vertices = self.door_polygons[self.current_door].reshape(-1, 2)
            middle = (vertices.min(0) + vertices.max(0)) / 2
            if np.dot(np.asarray(here) - middle, self.entry - middle) < 0:
                self.traversed.add(self.current_door)
        if sector is not None and sector != self.current_door:
            self.entry = np.asarray(before)
        self.current_door = sector

    def record(self, old_obs, action, state):
        old = self.previous
        unlocked = self.unlocked.copy()
        old_openings = len(self.opened)
        was_new = self.cell(state) not in self.visited
        reward = super().step(state)
        self.decisions += 1
        stamp = self.clock.elapsed
        keys = np.asarray(state["keys"], dtype=bool)
        acquired = keys & ~self.keys
        for key in np.flatnonzero(acquired):
            self.key_times[int(key)] = stamp
        self.keys |= keys
        for sector in sorted(self.unlocked - unlocked):
            color = int(state["door_lock"][sector])
            relevant = [self.key_times[k] for k in LOCK_KEYS.get(color, range(6)) if k in self.key_times]
            self.unlock_times.append(dict(sector=sector, lock=color, tic=stamp,
                                          since_key=stamp-max(relevant) if relevant else None))
        self.track_traversal(old, state)
        events = {key: max(0, state[key] - value) for key, value in self.maxima.items()}
        for key, increase in events.items():
            self.maxima[key] += increase
        movement = dist((old["x"], old["y"]), (state["x"], state["y"]))
        turn = abs((state["angle"]-old["angle"]+180) % 360-180)
        prior = self.previous_action or (0,) * 9
        action = tuple(int(bit) for bit in action)
        bits = "".join(map(str, action))
        self.same = self.same + 1 if action == prior and self.previous_action is not None else 1
        self.max_same = max(self.max_same, self.same)
        self.actions[bits] += 1
        self.previous_action = action
        self.counts["rounds_spent_lower_bound"] += sum(max(0, a-b) for a, b in zip(old["ammo"], state["ammo"]))
        self.counts["weapon_changes"] += old["weapon"] != state["weapon"]
        self.counts["weapon_press_edges"] += action[8] and not prior[8]
        self.counts["weapon_press_while_loaded"] += action[8] and not prior[8] and old["weapon_ammo"] > 0
        self.counts["use_press_edges"] += action[7] and not prior[7]
        self.counts["use_held"] += action[7] and prior[7]
        self.counts["attack_empty_ranged"] += action[6] and old["weapon"] >= 2 and old["weapon_ammo"] == 0
        self.counts["melee_attack"] += action[6] and old["weapon"] == 1
        self.counts["blocked_move"] += any(action[:4]) and movement < 2
        self.counts["forward_blocked"] += action[0] and movement < 2
        self.counts["path_units"] += movement
        self.counts["yaw_degrees"] += turn
        hit_types = old_obs["rays"]["hit_type"][FRONT]
        close = old_obs["rays"]["distance"][FRONT] * 1024 <= 64
        key_mask = (hit_types == 3) & close
        door_mask = (hit_types == 2) & close
        opportunities = {"door": np.any(door_mask), "lock": np.any(key_mask), "lock_with_key": any(self.keys) and np.any(key_mask)}
        for label, seen in opportunities.items():
            self.counts[f"{label}_range_steps"] += int(seen)
            self.counts[f"{label}_use_edges_in_range"] += int(seen and action[7] and not prior[7])
        visible = old_obs["visible_entities"]
        types = visible["entity_type"][visible["mask"].astype(bool)]
        enemy = bool(np.any((types >= 1) & (types <= 3)))
        self.counts["enemy_visible_steps"] += enemy
        self.counts["attack_without_visible_enemy"] += action[6] and not enemy
        self.counts["key_visible_steps"] += np.any(types == 9)
        for name, pressed in zip(BUTTONS, action):
            self.buttons[name] += pressed
        moved_cell = self.cell(old) != self.cell(state)
        self.counts["area_revisits"] += moved_cell and not was_new
        self.counts["key_held_revisits"] += moved_cell and not was_new and any(self.keys)
        progress = bool(was_new or acquired.any() or len(self.opened) > old_openings or events["dealt"] > 0 or state["exit_reached"])
        self.first_visited.setdefault(self.cell(state), stamp)
        self.stalled = 0 if progress else self.stalled + 1
        self.max_stall = max(self.stalled, self.max_stall)
        if self.stalled == 50:
            self.first_stall = stamp
            self.counts["stalls"] += 1
        if progress and self.first_stall is not None:
            self.recovery_tics.append(stamp - self.first_stall)
            self.first_stall = None
        self.rotation.append((movement, turn, progress))
        if len(self.rotation) == self.rotation.maxlen:
            travel, rotation, advancing = zip(*self.rotation)
            self.counts["spin_window_steps"] += sum(travel) < 48 and sum(rotation) >= 120 and not any(advancing)
        if self.dead_ends is not None:
            tile = (floor(state["x"]/self.tile[0]), floor(state["y"]/self.tile[1]))
            if tile in self.dead_ends and self.trapped_since is None:
                self.trapped_since = stamp
                self.counts["dead_end_entries"] += 1
            elif tile not in self.dead_ends and self.trapped_since is not None:
                self.counts["dead_end_departures"] += stamp - self.trapped_since <= 140
                self.trapped_since = None
        return dict(tic=stamp, raw_tic=state["tic"], x=state["x"], y=state["y"], angle=state["angle"],
                    health=state["health"], armor=state["armor"], keys=np.flatnonzero(self.keys).tolist(),
                    action=bits, weapon=state["weapon"], ammo=state["ammo"], reward=reward,
                    new_area=was_new, doors=len(self.opened), locks=len(self.unlocked),
                    enemy_visible=enemy, key_visible=bool(np.any(types == 9)), **events)

    def report(self):
        n = self.decisions
        counts = {key: value.item() if isinstance(value, np.generic) else value for key, value in self.counts.items()}
        if self.dead_ends is None:
            counts.update(dead_end_entries=None, dead_end_departures=None)
        return dict(self.summary(), decisions=n, armor=self.previous["armor"], kills=int(self.maxima["kills"]),
                    damage=self.maxima["damage_taken"], hits=int(self.maxima["hits"]), damage_dealt=self.maxima["dealt"],
                    shots=None, doors_traversed=len(self.traversed), key_pickup_tics=self.key_times,
                    unlocks=self.unlock_times, work=counts, progress_recoveries=self.recovery_tics,
                    longest_no_progress=self.max_stall, longest_identical_action=self.max_same,
                    distinct_actions=len(self.actions), top_action_share=max(self.actions.values(), default=0)/max(n, 1),
                    buttons=self.buttons, top_actions=self.actions.most_common(8))
