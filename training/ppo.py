"""Sequence-preserving PPO with distinct bootstrap and episode-continuation masks."""
from __future__ import annotations
import torch
from submission.model import policy_stats


def generalized_advantage(rewards, values, next_values, terminated, truncated, gamma=.99, lam=.95):
    advantage = torch.zeros_like(rewards)
    tail = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        # ViZDoom's wrapper supplies its last available observation on timeout.
        delta = rewards[t] + gamma * next_values[t] * (~terminated[t]) - values[t]
        # Neither kind of episode boundary propagates GAE into the next episode.
        tail = delta + gamma * lam * (~(terminated[t] | truncated[t])) * tail
        advantage[t] = tail
    return advantage, advantage + values


def _minibatch(model, data, sequences):
    device = data["values"].device
    env = torch.tensor([entry[0] for entry in sequences], device=device)
    left = torch.tensor([entry[1] for entry in sequences], device=device)
    right = torch.tensor([entry[2] for entry in sequences], device=device)
    length = max(end - begin for _, begin, end in sequences)
    time = torch.arange(length, device=device)[:, None] + left
    valid = time < right
    time = torch.minimum(time, right - 1)
    env = env.expand_as(time)

    if not model.recurrent:
        index = (time[valid], env[valid])
        obs = {key: value[index] for key, value in data["obs"].items()}
        logits, values, _ = model(obs, data["hidden"][index],
                                   data["previous_action"][index], data["starts"][index])
        return index, logits, values

    hidden = data["hidden"][left, env[0]].detach()
    logits_seq, values_seq = [], []
    for t, e in zip(time, env):
        obs = {key: value[t, e] for key, value in data["obs"].items()}
        logits, values, hidden = model(obs, hidden, data["previous_action"][t, e],
                                        data["starts"][t, e])
        logits_seq.append(logits)
        values_seq.append(values)
    # Padded samples follow the valid prefix and contribute no objective or gradient.
    return (time[valid], env[valid]), torch.stack(logits_seq)[valid], torch.stack(values_seq)[valid]


def update(model, optimizer, data, *, epochs=3, sequence_length=32,
           sequences_per_batch=4, clip=.2, value_coef=.5, entropy_coef=.01,
           max_grad_norm=.5, generator=None, gamma=.99):
    advantage, returns = generalized_advantage(data["rewards"], data["values"],
        data["next_values"], data["terminated"], data["truncated"], gamma=gamma)
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
    time_steps, envs = advantage.shape
    sequences = [(e, s, min(s + sequence_length, time_steps))
                 for e in range(envs) for s in range(0, time_steps, sequence_length)]
    stats = []
    for _ in range(epochs):
        order = torch.randperm(len(sequences), generator=generator).tolist()
        for start in range(0, len(order), sequences_per_batch):
            batch = [sequences[i] for i in order[start:start + sequences_per_batch]]
            index, logits, values = _minibatch(model, data, batch)
            _, log_prob, entropy = policy_stats(logits, data["actions"][index])
            old_log_prob = data["log_probs"][index]
            log_ratio = log_prob - old_log_prob
            ratio = log_ratio.exp()
            adv = advantage[index]
            policy_loss = -torch.minimum(ratio * adv, ratio.clamp(1-clip, 1+clip) * adv).mean()
            old_value = data["values"][index]
            clipped_value = old_value + (values - old_value).clamp(-clip, clip)
            target = returns[index]
            value_loss = .5 * torch.maximum((values-target).square(), (clipped_value-target).square()).mean()
            entropy = entropy.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            kl = ((ratio - 1) - log_ratio).mean()
            stats.append(torch.stack((loss, value_loss, entropy, kl, grad)).detach())
    result = torch.stack(stats).mean(0).tolist()
    return dict(zip(("loss", "value_loss", "entropy", "approx_kl", "grad_norm"), result))
