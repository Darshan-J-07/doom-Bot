"""Elapsed in-game time across ViZDoom's terminal map-clock reset."""
from doom_env.spaces import FRAME_SKIP


class EpisodeClock:
    def __init__(self, tic, limit=None):
        self.previous = int(tic)
        self.limit = None if limit is None else int(limit)
        self.elapsed = 0
        self.estimated = False

    def advance(self, tic, *, exit_reached=False):
        tic = int(tic)
        delta = tic - self.previous
        if delta < 0 or (exit_reached and tic == 0):
            self.estimated = True
            delta = FRAME_SKIP
        self.elapsed += delta
        if self.limit is not None:
            self.elapsed = min(self.limit, self.elapsed)
        self.previous = tic
