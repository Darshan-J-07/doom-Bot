"""Compact neural memories driven by the official egocentric inputs."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SlotMemory(nn.Module):
    def __init__(self, width, slots=4, slot_width=24):
        super().__init__()
        self.slots = slots
        self.slot_width = slot_width
        self.state_size = slots * slot_width
        self.address = nn.Parameter(torch.empty(slots, slot_width).normal_(std=.15))
        self.query = nn.Linear(width, slot_width)
        self.write = nn.Linear(width, slot_width)
        self.gate = nn.Linear(width, slots)
        self.read = nn.Linear(slot_width, width)

    def forward(self, latent, hidden):
        memory = hidden.reshape(-1, self.slots, self.slot_width)
        addresses = memory + self.address
        weights = ((self.query(latent).unsqueeze(1) * addresses).sum(-1) / self.slot_width ** .5).softmax(-1)
        retrieved = (weights.unsqueeze(-1) * memory).sum(1)
        write_rate = (weights * self.gate(latent).sigmoid()).unsqueeze(-1)
        next_memory = memory.lerp(self.write(latent).tanh().unsqueeze(1), write_rate)
        return latent + self.read(retrieved).tanh(), next_memory.flatten(1)


class OccupancyMemory(nn.Module):
    """Differentiable local occupancy, with learned egomotion and update gates."""

    def __init__(self, width, side=9, spacing=48.):
        super().__init__()
        self.side = side
        self.spacing = spacing
        self.state_size = side * side
        coordinates = (torch.arange(side) - side // 2) * spacing
        forward, right = torch.meshgrid(coordinates, coordinates, indexing="ij")
        bins = torch.atan2(-right, forward) * (64 / (2 * torch.pi))
        self.register_buffer("ray_bins", bins.round().long().remainder(64).flatten())
        self.register_buffer("radius", torch.hypot(right, forward).flatten())
        self.egomotion = nn.Linear(4, 3, bias=False)
        self.fuse = nn.Sequential(nn.Linear(side * side, width), nn.Tanh())
        self.write_gate = nn.Linear(width, side * side)

    def initialize_egomotion(self):
        # Initialization uses four elapsed tics and allows the sampled policy to learn corrections.
        with torch.no_grad():
            scale = 80 / (self.spacing * (self.side - 1) / 2)
            self.egomotion.weight.copy_(self.egomotion.weight.new_tensor(
                [[0, scale, 0, 0], [scale, 0, 0, 0], [0, 0, .25, -.25]]))

    def forward(self, latent, hidden, distance, hit_type, motion):
        shift_right, shift_forward, yaw = self.egomotion(motion).unbind(-1)
        cos, sin = yaw.cos(), yaw.sin()
        transform = torch.stack((cos, -sin, shift_right, sin, cos, shift_forward), -1).reshape(-1, 2, 3)
        old = hidden.reshape(-1, 1, self.side, self.side)
        grid = F.affine_grid(transform, old.shape, align_corners=True)
        prior = F.grid_sample(old, grid, align_corners=True).flatten(1)
        difference = distance[:, self.ray_bins] * 1024 - self.radius
        hit = hit_type[:, self.ray_bins]
        free = difference > self.spacing / 2
        edge = (difference.abs() <= self.spacing / 2) & (hit >= 1) & (hit <= 3)
        evidence = edge.float() - free.float()
        confidence = (free | edge) * self.write_gate(latent).sigmoid()
        updated = prior.lerp(evidence, confidence)
        return latent + self.fuse(updated), updated
