"""Training-only shaping and explicitly LOCAL proxy metrics, never organizer scores."""
from __future__ import annotations
import numpy as np


class EpisodeMetrics:
    def reset(self, state):
        self.previous = state
        self.visited = {self.cell(state)}
        self.opened, self.unlocked = set(), set()
        self.total_reward = 0.
        self.start_health = state["health"]

    @staticmethod
    def cell(state):
        return (int(np.floor(state["x"] / 96)), int(np.floor(state["y"] / 96)))

    def step(self, state):
        previous = self.previous
        cell = self.cell(state)
        novel = cell not in self.visited
        self.visited.add(cell)
        keys = max(0, int(np.sum(state["keys"]) - np.sum(previous["keys"])))
        kills = max(0, state["kills"] - previous["kills"])
        damage = max(0., state["damage_taken"] - previous["damage_taken"])
        new_doors = new_locks = 0
        if state["sector_ceiling"] is not None and previous["sector_ceiling"] is not None:
            opening = state["sector_ceiling"] - state["sector_floor"]
            was_opening = previous["sector_ceiling"] - previous["sector_floor"]
            crossed = np.flatnonzero((state["door_lock"] >= 0) & (opening >= 56) & (was_opening < 56))
            for sector in crossed:
                if int(sector) not in self.opened:
                    new_doors += 1
                    self.opened.add(int(sector))
                    if state["door_lock"][sector] > 0:
                        new_locks += 1; self.unlocked.add(int(sector))
        # Sparse success dominates one-time novelty; no survival-per-step reward.
        reward = (20. * state["exit_reached"] - 5. * state["dead"] + 2. * kills
                  + 4. * keys + 2. * new_doors + 3. * new_locks + .05 * novel
                  - .02 * damage - .002)
        self.total_reward += reward
        self.previous = state
        return reward

    def summary(self):
        s = self.previous
        # A transparent proxy; exact tier weights and scorer formula are private.
        proxy = (100 * s["exit_reached"] + 15 * int(np.sum(s["keys"]))
                 + 5 * len(self.opened) + 10 * len(self.unlocked) + 3 * s["kills"]
                 + .1 * len(self.visited) - .1 * s["damage_taken"] - 10 * s["dead"])
        return dict(proxy_score=float(proxy), exit=bool(s["exit_reached"]), dead=bool(s["dead"]),
                    survived=not s["dead"], kills=s["kills"], keys=int(np.sum(s["keys"])),
                    doors=len(self.opened), locks=len(self.unlocked), unique_cells=len(self.visited),
                    health=s["health"], damage=s["damage_taken"], ticks=s["tic"],
                    shaped_return=self.total_reward)


class ProgressReward(EpisodeMetrics):
    """Training-only shaping with one-time milestones and bounded stall costs."""

    def reset(self, state):
        super().reset(state)
        self.stall_steps = 0
        self.credited_keys = np.asarray(state["keys"], dtype=bool).copy()
        self.credited_terminal = False

    def step(self, state):
        if self.credited_terminal:
            return 0.0
        previous = self.previous
        visited, doors, locks = len(self.visited), len(self.opened), len(self.unlocked)
        self.credited_terminal = bool(state["exit_reached"] or state["dead"])
        existing_reward = super().step(state)
        keys = np.asarray(state["keys"], dtype=bool)
        new_keys = np.count_nonzero(keys & ~self.credited_keys)
        self.credited_keys |= keys
        new_doors, new_locks = len(self.opened) - doors, len(self.unlocked) - locks
        new_cell = len(self.visited) > visited
        kills = max(0, state["kills"] - previous["kills"])
        damage = max(0., state["damage_taken"] - previous["damage_taken"])
        progressed = new_cell or new_keys or new_doors or kills
        self.stall_steps = 0 if progressed else self.stall_steps + 1
        stopped = np.hypot(state["x"] - previous["x"], state["y"] - previous["y"]) < 2
        completion = (24 + 2 * np.clip(state["health"] / 100, 0, 1)) if state["exit_reached"] else 0
        reward = (completion - 6. * state["dead"] + 2.5 * kills + 5. * new_keys
                  + 2. * new_doors + 5. * new_locks + .10 * new_cell
                  - .035 * damage - .002 - .012 * (self.stall_steps > 24)
                  - .01 * (self.stall_steps > 6 and stopped))
        self.total_reward += reward - existing_reward
        return float(reward)
