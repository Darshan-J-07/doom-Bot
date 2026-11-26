"""Optional spawn-based engine workers for machines with enough RAM."""
from __future__ import annotations
import multiprocessing as mp
import traceback


def _serve(pipe, wad, ticks):
    from doom_env import make_env
    env = None
    try:
        env = make_env(wad, episode_ticks=ticks, render_buffers=False)
        while True:
            command, payload = pipe.recv()
            if command == "close": break
            result = env.reset(seed=payload) if command == "reset" else env.step(payload)
            pipe.send((True, result, env.privileged_state()))
    except BaseException:
        try: pipe.send((False, traceback.format_exc(), None))
        except (EOFError,OSError): pass
    finally:
        if env is not None: env.close()
        pipe.close()


class WorkerEnv:
    def __init__(self, wad, episode_ticks):
        from doom_env import action_space, observation_space
        self.action_space, self.observation_space = action_space(), observation_space()
        context = mp.get_context("spawn")
        self.pipe, child = context.Pipe()
        self.process = context.Process(target=_serve, args=(child,wad,episode_ticks), daemon=True)
        self.process.start(); child.close()
        self.state = None

    def send(self, command, value):
        self.pipe.send((command,value))

    def receive(self):
        if not self.pipe.poll(60): raise TimeoutError("Training worker did not respond within 60 s")
        ok, result, self.state = self.pipe.recv()
        if not ok: raise RuntimeError(result)
        return result

    def reset(self, seed=None):
        self.send("reset",seed); return self.receive()

    def step(self, action):
        self.send("step",action); return self.receive()

    def privileged_state(self): return self.state

    def close(self):
        if self.process.is_alive():
            try: self.send("close",None)
            except (BrokenPipeError,OSError): pass
            self.process.join(3)
            if self.process.is_alive(): self.process.terminate(); self.process.join(3)
        self.pipe.close()
