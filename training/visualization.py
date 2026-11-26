"""Native ViZDoom playback and local telemetry for the training and demo tools."""
from __future__ import annotations

import argparse
import math
from threading import Lock, get_ident
import time
from unittest.mock import patch

import vizdoom
from doom_env import make_env, wrapper
from doom_env.spaces import BUTTONS, FRAME_SKIP


WINDOW_EXIT_ERRORS = (vizdoom.ViZDoomUnexpectedExitException, vizdoom.ViZDoomIsNotRunningException)


class _ObservationState:
    """Expose standard-sized buffers while the native window renders at 640x480."""

    def __init__(self, state):
        self._state = state

    def __getattr__(self, name):
        value = getattr(self._state, name)
        if name in ("screen_buffer", "depth_buffer"):
            return value[::4, ::4]
        return value


class _WindowedGame(vizdoom.DoomGame):
    def __init__(self, render_all_frames, contract_pixels):
        super().__init__()
        self.render_all_frames = render_all_frames
        self.contract_pixels = contract_pixels

    def init(self):
        self.set_window_visible(True)
        if not self.contract_pixels:
            self.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
            self.set_render_hud(True)
        self.set_render_all_frames(self.render_all_frames)
        return super().init()

    def get_state(self):
        state = super().get_state()
        if self.contract_pixels or state is None:
            return state
        return _ObservationState(state)


class _WindowedBindings:
    def __init__(self, render_all_frames, contract_pixels):
        self.owner = get_ident()
        self.render_all_frames = render_all_frames
        self.contract_pixels = contract_pixels

    def DoomGame(self):
        if get_ident() == self.owner:
            return _WindowedGame(self.render_all_frames, self.contract_pixels)
        return vizdoom.DoomGame()

    def __getattr__(self, name):
        return getattr(vizdoom, name)


_window_init_lock = Lock()


def make_visible_env(wad, map_name="MAP01", *, render_all_frames=True, contract_pixels=False, **kwargs):
    # The pinned wrapper sets headless mode during construction. Override only
    # this thread's game factory until init() finishes, then restore the binding.
    with _window_init_lock, patch.object(wrapper, "vd", _WindowedBindings(render_all_frames, contract_pixels)):
        return make_env(wad, map_name, **kwargs)


def add_playback_arguments(parser, *, visible=False):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=visible,
                       help="open the native ViZDoom window")
    group.add_argument("--headless", dest="visualize", action="store_false",
                       help="run without a window")
    parser.add_argument("--speed", type=float, default=1.,
                        help="visible playback speed relative to 35 tics/sec (0 is uncapped)")
    parser.add_argument("--telemetry-every", type=int, default=8,
                        help="show a status line every N visible agent decisions")


def validate_playback_arguments(parser, args):
    if not math.isfinite(args.speed) or args.speed < 0:
        parser.error("--speed must be a finite nonnegative number")
    if args.telemetry_every < 1:
        parser.error("--telemetry-every must be positive")


class Playback:
    def __init__(self, label, speed, telemetry_every):
        self.label = label
        self.speed = speed
        self.telemetry_every = telemetry_every
        self.interval = FRAME_SKIP / 35 / speed if speed else 0.
        self.last_step = None
        self.decisions = 0

    def start(self, description):
        self.last_step = None
        self.decisions = 0
        pace = f"{self.speed:g}x" if self.speed else "uncapped"
        print(f"[{self.label}] {description} | {pace}", flush=True)

    def wait(self, stop_requested=None):
        if self.interval and self.last_step is not None:
            deadline = self.last_step + self.interval
            while (remaining := deadline - time.monotonic()) > 0:
                if stop_requested and stop_requested():
                    break
                time.sleep(min(remaining, .1) if stop_requested else remaining)
        self.last_step = time.monotonic()

    def step(self, env, action, tracker, state, reward, *, update=None, log_prob=None, global_step=None):
        self.decisions += 1
        finished = state["finished"]
        if self.decisions != 1 and self.decisions % self.telemetry_every and not finished:
            return
        ammo = int(env._game.get_game_variable(vizdoom.GameVariable.SELECTED_WEAPON_AMMO))
        decision = " + ".join(button for button, pressed in zip(BUTTONS, action) if pressed) or "IDLE"
        probability = f" (p={math.exp(float(log_prob)):.2%})" if log_prob is not None else ""
        progress = ""
        if finished:
            outcome = ("drill complete" if state.get("lesson_complete") else "exit" if state["exit_reached"]
                       else "death" if state["dead"] else "timeout")
            progress = f" | episode ended ({outcome})"
        training = f" | training decisions {global_step:,} | PPO updates {update:,}" if global_step is not None else ""
        clock = getattr(tracker, "clock", None)
        elapsed = clock.elapsed if clock is not None else state["tic"]
        approximate = " (estimated)" if clock is not None and clock.estimated else ""
        print(f"[{self.label}] Decision {self.decisions:,} | elapsed {elapsed}/{state['episode_ticks']} tics{approximate}"
              f"{training} | {decision}{probability}{progress}\n"
              f"  HP {state['health']:g} | weapon ammo {ammo} | shaped reward {reward:+.3f}"
              f" | episode return {tracker.total_reward:+.2f} | cells {len(tracker.visited)}"
              f" | keys {int(state['keys'].sum())} | doors {len(tracker.opened)}"
              f" | unlocks {len(tracker.unlocked)} | kills {state['kills']}", flush=True)
