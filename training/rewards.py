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
