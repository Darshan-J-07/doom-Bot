"""Sequence-preserving PPO with distinct bootstrap and episode-continuation masks."""
from __future__ import annotations
import torch
from submission.model import policy_stats


def generalized_advantage(rewards, values, next_values, terminated, truncated, gamma=.99, lam=.95):
    advantage = torch.zeros_like(rewards)
    tail = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        # Time limits bootstrap the FINAL observation, never the reset observation.
        delta = rewards[t] + gamma * next_values[t] * (~terminated[t]) - values[t]
        # Neither kind of episode boundary propagates GAE into the next episode.
        tail = delta + gamma * lam * (~(terminated[t] | truncated[t])) * tail
        advantage[t] = tail
    return advantage, advantage + values


def update(model, optimizer, data, *, epochs=3, sequence_length=32,
           sequences_per_batch=4, clip=.2, value_coef=.5, entropy_coef=.01,
           max_grad_norm=.5, generator=None):
    advantage, returns = generalized_advantage(data["rewards"], data["values"],
        data["next_values"], data["terminated"], data["truncated"])
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
    time_steps, envs = advantage.shape
    sequences = [(e, s, min(s + sequence_length, time_steps))
                 for e in range(envs) for s in range(0, time_steps, sequence_length)]
    stats = []
    for _ in range(epochs):
        order = torch.randperm(len(sequences), generator=generator).tolist()
        for start in range(0, len(order), sequences_per_batch):
            losses, value_losses, entropies, kls, counts = [], [], [], [], []
            for idx in order[start:start + sequences_per_batch]:
                env, left, right = sequences[idx]
                hidden = data["hidden"][left, env:env + 1].detach()
                logits_seq, value_seq = [], []
                for t in range(left, right):
                    obs = {k: v[t, env:env + 1] for k, v in data["obs"].items()}
                    logits, value, hidden = model(obs, hidden,
                        data["previous_action"][t, env:env + 1], data["starts"][t, env:env + 1])
                    logits_seq.append(logits[0]); value_seq.append(value[0])
                logits, value = torch.stack(logits_seq), torch.stack(value_seq)
                _, log_prob, entropy = policy_stats(logits, data["actions"][left:right, env])
                old_log_prob = data["log_probs"][left:right, env]
                ratio = (log_prob - old_log_prob).exp()
                adv = advantage[left:right, env]
                losses.append(-torch.minimum(ratio * adv, ratio.clamp(1-clip, 1+clip) * adv).mean())
                old_value = data["values"][left:right, env]
                clipped_value = old_value + (value - old_value).clamp(-clip, clip)
                target = returns[left:right, env]
                value_losses.append(.5 * torch.maximum((value-target).square(), (clipped_value-target).square()).mean())
                entropies.append(entropy.mean())
                kls.append(((ratio - 1) - (log_prob - old_log_prob)).mean().detach())
                counts.append(right-left)
            weights = torch.tensor(counts, device=advantage.device, dtype=torch.float32)
            weights /= weights.sum()
            policy_loss = (torch.stack(losses) * weights).sum()
            value_loss = (torch.stack(value_losses) * weights).sum()
            entropy = (torch.stack(entropies) * weights).sum()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            stats.append([float(loss.detach()), float(value_loss.detach()), float(entropy.detach()),
                          float(torch.stack(kls).mean()), float(grad)])
    result = torch.tensor(stats).mean(0).tolist()
    return dict(zip(("loss", "value_loss", "entropy", "approx_kl", "grad_norm"), result))
