"""Experimental PPO with a frozen policy anchor on fresh collector states."""

from __future__ import annotations

import torch

from submission.model import GROUP_SIZES, policy_stats
from training.ppo import _minibatch, generalized_advantage


def forward_kl(frozen_log_probs, student_logits):
    """Joint teacher→student KL for the product of six categorical heads."""
    groups = student_logits.split(GROUP_SIZES, -1)
    return sum((teacher.exp() * (teacher - student.log_softmax(-1))).sum(-1)
               for teacher, student in zip(frozen_log_probs, groups, strict=True))


def update_anchored(model, optimizer, data, *, reference, anchor_coefficient, epochs=3,
                    sequence_length=32, sequences_per_batch=4, clip=.2, value_coef=.5,
                    entropy_coef=.01, max_grad_norm=.5, generator=None, gamma=.99, lam=.95,
                    bootstrap_truncations=True, audit_replay=False, burn_in=0):
    if model.recurrent or reference.recurrent or burn_in or model.config != reference.config:
        raise ValueError("The first anchored arm requires an architecture-matched feedforward reference")
    if not 0 <= anchor_coefficient < float("inf"):
        raise ValueError("The anchor coefficient must be finite and nonnegative")
    advantage, returns = generalized_advantage(data["rewards"], data["values"], data["next_values"],
                                                data["terminated"], data["truncated"], gamma=gamma, lam=lam,
                                                bootstrap_truncations=bootstrap_truncations)
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
    time_steps, envs = advantage.shape
    sequences = [(env, start, min(start + sequence_length, time_steps))
                 for env in range(envs) for start in range(0, time_steps, sequence_length)]
    with torch.no_grad():
        observations = {key: value.flatten(0, 1) for key, value in data["obs"].items()}
        logits, _, _ = reference(observations, data["hidden"].flatten(0, 1),
                                 data["previous_action"].flatten(0, 1), data["starts"].flatten())
        frozen = tuple(group.log_softmax(-1).reshape(time_steps, envs, -1)
                       for group in logits.split(GROUP_SIZES, -1))
    replay_error = None
    if audit_replay:
        with torch.no_grad():
            index, logits, _ = _minibatch(model, data, sequences)
            _, current, _ = policy_stats(logits, data["actions"][index])
            replay_error = (current - data["log_probs"][index]).abs().max().item()
        if replay_error > 1e-4:
            raise ValueError(f"PPO replay changed pre-update action log-probabilities by {replay_error:g}")

    stats, sizes = [], []
    for _ in range(epochs):
        order = torch.randperm(len(sequences), generator=generator).tolist()
        for start in range(0, len(order), sequences_per_batch):
            batch = [sequences[index] for index in order[start:start + sequences_per_batch]]
            index, logits, values = _minibatch(model, data, batch)
            _, log_prob, entropy = policy_stats(logits, data["actions"][index])
            log_ratio = log_prob - data["log_probs"][index]
            ratio = log_ratio.exp()
            adv = advantage[index]
            policy_loss = -torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            old_value = data["values"][index]
            clipped_value = old_value + (values - old_value).clamp(-clip, clip)
            target = returns[index]
            value_loss = .5 * torch.maximum((values-target).square(), (clipped_value-target).square()).mean()
            entropy = entropy.mean()
            anchor_kl = forward_kl(tuple(group[index] for group in frozen), logits).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy + anchor_coefficient * anchor_kl
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite anchored PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            collector_kl = ((ratio - 1) - log_ratio).mean()
            clip_fraction = (torch.abs(ratio - 1) > clip).float().mean()
            metrics = (loss, policy_loss, value_loss, entropy, collector_kl, clip_fraction, grad, anchor_kl)
            stats.append(torch.stack(metrics).detach())
            sizes.append(values.numel())
    stats = torch.stack(stats)
    if not torch.isfinite(stats).all():
        raise FloatingPointError("Nonfinite anchored PPO learning metrics")
    counts = stats.new_tensor(sizes)
    result = ((stats * counts[:, None]).sum(0) / counts.sum()).tolist()
    variance = returns.var(unbiased=False).item()
    explained = 1 - (returns - data["values"]).var(unbiased=False).item() / variance if variance > 1e-8 else None
    names = ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm", "anchor_kl")
    output = dict(zip(names, result, strict=True), explained_variance=explained, anchor_coefficient=anchor_coefficient)
    if replay_error is not None:
        output["replay_log_prob_max_error"] = replay_error
    return output
