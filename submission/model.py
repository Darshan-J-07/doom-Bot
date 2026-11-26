"""Custom spatial actor critic. All button decisions come from neural logits."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

GROUP_SIZES = (3, 3, 3, 2, 2, 2)


def buttons(groups: torch.Tensor) -> torch.Tensor:
    """Factorized action encoding, not a gameplay controller.

    Each opposing button pair is a single categorical choice: idle / first / second.
    Attack, use and next-weapon are independent categorical binary choices.
    """
    return torch.stack([groups[..., 0] == 1, groups[..., 0] == 2,
                        groups[..., 1] == 1, groups[..., 1] == 2,
                        groups[..., 2] == 1, groups[..., 2] == 2,
                        groups[..., 3] == 1, groups[..., 4] == 1,
                        groups[..., 5] == 1], dim=-1).to(torch.int8)


def _choose_groups(distributions, deterministic, generator):
    return torch.stack([
        d.logits.argmax(-1) if deterministic else
        torch.multinomial(d.probs, 1, generator=generator).squeeze(-1)
        for d in distributions], -1)


def action_groups(logits, deterministic=False, generator=None):
    """Choose the same PPO policy actions without computing training statistics."""
    distributions = [Categorical(logits=group) for group in logits.split(GROUP_SIZES, -1)]
    return _choose_groups(distributions, deterministic, generator)


def policy_stats(logits, actions=None, deterministic=False, generator=None):
    distributions = [Categorical(logits=x) for x in logits.split(GROUP_SIZES, -1)]
    if actions is None:
        actions = _choose_groups(distributions, deterministic, generator)
    log_prob = torch.stack([d.log_prob(actions[..., i])
                            for i, d in enumerate(distributions)], -1).sum(-1)
    entropy = torch.stack([d.entropy() for d in distributions], -1).sum(-1)
    return actions, log_prob, entropy


class SpatialPolicy(nn.Module):
    def __init__(self, hidden_size=128, recurrent=True, entity_mode="pool"):
        super().__init__()
        self.config = dict(hidden_size=hidden_size, recurrent=recurrent, entity_mode=entity_mode)
        self.hidden_size, self.recurrent, self.entity_mode = hidden_size, recurrent, entity_mode
        self.hit_embed = nn.Embedding(6, 4)
        self.entity_embed = nn.Embedding(12, 8, padding_idx=0)
        self.ray_conv = nn.Sequential(
            nn.Conv1d(7, 24, 5, padding=2, padding_mode="circular"), nn.Tanh(),
            nn.Conv1d(24, 32, 5, stride=4, padding=2, padding_mode="circular"), nn.Tanh(),
            nn.Flatten(), nn.Linear(32 * 16, 96), nn.Tanh())
        # Fixed sensor bearing channels preserve orientation after convolution.
        bearing = torch.arange(64) * (2 * torch.pi / 64)
        self.register_buffer("bearing", torch.stack([bearing.sin(), bearing.cos()], -1))
        self.entity_mlp = nn.Sequential(nn.Linear(13, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        self.vitals_mlp = nn.Sequential(nn.Linear(17, 32), nn.Tanh())
        self.fusion = nn.Sequential(nn.Linear(96 + 65 + 32 + 9, hidden_size), nn.Tanh())
        if recurrent:
            self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, sum(GROUP_SIZES))
        self.critic = nn.Linear(hidden_size, 1)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        if entity_mode == "attention":
            # Start at uniform valid-entity weights and preserve the baseline RNG stream.
            with torch.random.fork_rng(devices=[]):
                self.entity_query = nn.Linear(32, 32)
            nn.init.zeros_(self.entity_query.weight)
            nn.init.zeros_(self.entity_query.bias)

    def initial_state(self, batch_size=1, device=None):
        return torch.zeros(batch_size, self.hidden_size,
                           device=device or next(self.parameters()).device)

    def forward(self, obs, hidden, previous_action, episode_start):
        b = obs["distance"].shape[0]
        ray = torch.cat([obs["distance"].unsqueeze(-1), self.hit_embed(obs["hit_type"]),
                         self.bearing.expand(b, -1, -1)], -1)
        spatial = self.ray_conv(ray.transpose(1, 2))
        entity = self.entity_mlp(torch.cat([obs["entities"], self.entity_embed(obs["entity_type"])], -1))
        mask = obs["mask"].unsqueeze(-1)
        count = mask.sum(1)
        vitals = self.vitals_mlp(obs["vitals"])
        if self.entity_mode == "attention":
            query = self.entity_query(vitals).unsqueeze(1)
            scores = (entity * query).sum(-1, keepdim=True) / (32 ** .5)
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
            weights = scores.softmax(dim=1) * mask
            mean = (entity * weights).sum(1)
        else:
            mean = (entity * mask).sum(1) / count.clamp_min(1)
        maximum = entity.masked_fill(mask == 0, -1e4).amax(1)
        maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
        pooled = torch.cat([mean, maximum, count / 32], -1)
        if self.entity_mode == "none":
            pooled = pooled * 0  # explicit ablation with the same parameter count
        hidden = hidden * (1 - episode_start.float()).unsqueeze(-1)
        previous_action = previous_action * (1 - episode_start.float()).unsqueeze(-1)
        latent = self.fusion(torch.cat([spatial, pooled, vitals,
                                       previous_action.float()], -1))
        next_hidden = self.gru(latent, hidden) if self.recurrent else latent
        return self.actor(next_hidden), self.critic(next_hidden).squeeze(-1), next_hidden
