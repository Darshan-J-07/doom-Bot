"""Offline, read-only inference entry point for contract 1.0.0-draft."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from submission.model import SpatialPolicy, buttons, policy_stats
from submission.preprocessing import tensor_batch


class DoomAgent:
    def __init__(self, observation_space, action_space, device):
        torch.set_num_threads(1)
        self.device = torch.device(device)
        self.observation_space, self.action_space = observation_space, action_space
        if tuple(action_space.shape) != (9,) or tuple(observation_space["vitals"].shape) != (17,):
            raise ValueError("DoomAgent requires the official 1.0.0-draft spaces")
        checkpoint = torch.load(Path(__file__).parent / "weights" / "best_model.pt",
                                map_location="cpu", weights_only=True)
        if checkpoint.get("trained_steps", 0) <= 0 or checkpoint.get("updates", 0) <= 0:
            raise ValueError("A genuinely trained checkpoint is required")
        if checkpoint.get("contract_version") != "1.0.0-draft":
            raise ValueError("Checkpoint contract version mismatch")
        self.model = SpatialPolicy(**checkpoint["model_config"]).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.deterministic = checkpoint.get("deterministic", False)
        self.reset()

    def reset(self):
        self.hidden = self.model.initial_state(1, self.device)
        self.previous_action = torch.zeros(1, 9, device=self.device)
        self.episode_start = torch.ones(1, device=self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(0)

    @torch.inference_mode()
    def act(self, observation):
        logits, _, self.hidden = self.model(tensor_batch([observation], self.device),
            self.hidden, self.previous_action, self.episode_start)
        groups, _, _ = policy_stats(logits, deterministic=self.deterministic, generator=self.generator)
        action = buttons(groups)
        self.previous_action = action.float()
        self.episode_start.zero_()
        return action[0].cpu().numpy().astype(np.int8, copy=False)
