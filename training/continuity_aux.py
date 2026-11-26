"""Bounded exploration bonuses for metered training experiments."""
from __future__ import annotations

import torch

from doom_env.spaces import NUM_RAYS, RAY_HIT_TYPES, VITALS_INDEX


def count_reward(base):
    class CountReward(base):
        def reset(self, state):
            super().reset(state)
            self.coverage_credit = 0.

        def step(self, state, action, observation, following):
            new_tile = self.cell(state) in self.layout and self.cell(state) not in self.visited
            reward = super().step(state, action, observation, following)
            bonus = min(.06, 1.2 - self.coverage_credit) if new_tile and not self.finished else 0.
            self.coverage_credit += bonus
            self.total_reward += self.coverage_credit
            return reward + bonus

        def summary(self):
            return dict(super().summary(), count_based_credit=self.coverage_credit)

    return CountReward


class Novelty:
    """Random network distillation over legal ray geometry during training only."""

    def __init__(self, seed, state=None):
        width = NUM_RAYS * (1 + len(RAY_HIT_TYPES))

        def network():
            return torch.nn.Sequential(torch.nn.Linear(width, 48), torch.nn.Tanh(), torch.nn.Linear(48, 24))

        # Do not advance action or minibatch sampling RNG while building the RND networks.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 350117)
            self.target = network().eval().requires_grad_(False)
            self.predictor = network()
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=.001)
        self.running_error = None
        if state is not None:
            self.target.load_state_dict(state["target"], strict=True)
            self.predictor.load_state_dict(state["predictor"], strict=True)
            self.optimizer.load_state_dict(state["optimizer"])
            self.running_error = state["running_error"]

    def state_dict(self):
        return dict(target=self.target.state_dict(), predictor=self.predictor.state_dict(),
                    optimizer=self.optimizer.state_dict(), running_error=self.running_error)

    def update(self, update_ppo, model, optimizer, data, **kwargs):
        obs = data["obs"]
        hit = torch.nn.functional.one_hot(obs["hit_type"], len(RAY_HIT_TYPES)).float()
        features = torch.cat((obs["distance"].clamp(0, 1).unsqueeze(-1), hit), dim=-1).flatten(-2)
        with torch.no_grad():
            target = self.target(features)
            error = (self.predictor(features) - target).square().mean(-1)
            mean = float(error.mean())
            scale = max(self.running_error if self.running_error is not None else mean, .00001)
            motion = (obs["vitals"][..., VITALS_INDEX["vel_forward"]].abs()
                      + obs["vitals"][..., VITALS_INDEX["vel_right"]].abs())
            eligible = ~(data["terminated"] | data["truncated"]) & (motion > .12)
            bonus = .012 * (error / scale - .7).clamp(0, 1) * eligible
        stats = update_ppo(model, optimizer, {**data, "rewards": data["rewards"] + bonus}, **kwargs)
        self.optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(self.predictor(features), target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite RND prediction loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), .5, error_if_nonfinite=True)
        self.optimizer.step()
        self.running_error = .99 * scale + .01 * mean
        return dict(stats, rnd_predictor_loss=float(loss.detach()), rnd_bonus_mean=float(bonus.mean()),
                    rnd_bonus_max=float(bonus.max()), rnd_error_scale=self.running_error)
