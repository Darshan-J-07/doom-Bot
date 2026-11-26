"""Privileged training diagnostics for the post-lock funnel."""
from __future__ import annotations

from math import atan2, degrees

from doom_env.spaces import BUTTONS, FRAME_SKIP
from training.continuity_post_lock_maps import far_side
from training.exit_recovery_maps import distance_to_wall


USE = BUTTONS.index("USE")


def instrument(base_reward):
    """Keep the archived reward unchanged while recording the progression stages."""

    class PostLockMetrics(base_reward):
        def __init__(self, spec, **options):
            super().__init__(spec, **options)
            self.post_lock = spec.get("post_lock", False)
            if self.post_lock:
                self._post_lock_door = tuple(spec["open_door_cell"])
                self._post_lock_far = far_side(spec)
                self._post_lock_exit = tuple(spec["exit_cell"])
                self._post_lock_segment = spec["exit_segment"]
                self._post_lock_approach = spec["exit_approach"]

        def reset(self, state):
            value = super().reset(state)
            if self.post_lock:
                self._post_lock_step = 0
                self._post_lock_entered_from_near = False
                self._post_lock_held_use = False
                self._post_lock_starts_far = self.cell(state) in self._post_lock_far
                self._post_lock_stages = dict.fromkeys(("door_crossing", "exit_area_discovery",
                                                       "interaction_range", "alignment", "fresh_use", "exit"))
            return value

        def step(self, state, action, observation, following):
            if not self.post_lock or self.finished:
                return super().step(state, action, observation, following)
            previous = self.previous
            before, after = self.cell(previous), self.cell(state)
            stages = self._post_lock_stages
            tic = FRAME_SKIP * self._post_lock_step
            use = bool(action[USE])
            if self._post_lock_starts_far or stages["door_crossing"] is not None:
                if before == self._post_lock_exit and stages["exit_area_discovery"] is None:
                    stages["exit_area_discovery"] = tic
                if self._post_lock_segment and before in self._post_lock_far and stages["fresh_use"] is None:
                    self.observe_switch(previous, use, tic)
            self._post_lock_held_use = use
            self._post_lock_step += 1
            if after == self._post_lock_door and before != self._post_lock_door:
                self._post_lock_entered_from_near = before not in self._post_lock_far
            elif before == self._post_lock_door and after != before:
                if (after in self._post_lock_far and self._post_lock_entered_from_near
                        and stages["door_crossing"] is None):
                    stages["door_crossing"] = tic + FRAME_SKIP
                self._post_lock_entered_from_near = False
            if state["exit_reached"] and stages["exit"] is None:
                stages["exit"] = tic + FRAME_SKIP
            return super().step(state, action, observation, following)

        def observe_switch(self, state, use, tic):
            stages = self._post_lock_stages
            position = state["x"], state["y"]
            midpoint = tuple((a + b) / 2 for a, b in zip(*self._post_lock_segment))
            heading = degrees(atan2(midpoint[1] - position[1], midpoint[0] - position[0]))
            error = abs((heading - state["angle"] + 180) % 360 - 180)
            near = distance_to_wall(position, self._post_lock_segment) <= 64
            if not (near or stages["exit_area_discovery"] is None and error <= 45):
                return
            if not self._clear_segment(position, self._post_lock_approach):
                return
            if error <= 45 and stages["exit_area_discovery"] is None:
                stages["exit_area_discovery"] = tic
            if near and stages["exit_area_discovery"] is not None:
                if stages["interaction_range"] is None:
                    stages["interaction_range"] = tic
                if error <= 15:
                    if stages["alignment"] is None:
                        stages["alignment"] = tic
                    if use and not self._post_lock_held_use:
                        stages["fresh_use"] = tic

        def summary(self):
            result = super().summary()
            if self.post_lock:
                result["post_lock_funnel"] = dict(spawn_side="far-side" if self._post_lock_starts_far
                                                 else "approach-side",
                                                 stage_tics=dict(self._post_lock_stages),
                                                 interaction_events="four-tic action-boundary training geometry")
            return result

    return PostLockMetrics
