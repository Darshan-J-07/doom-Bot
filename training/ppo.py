"""Sequence-preserving PPO with distinct bootstrap and episode-continuation masks."""
from __future__ import annotations
import torch
from submission.model import policy_stats


def generalized_advantage(rewards, values, next_values, terminated, truncated, gamma=.99, lam=.95,
                          bootstrap_truncations=True):
    advantage = torch.zeros_like(rewards)
    tail = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        episode_end = terminated[t] | truncated[t]
        bootstrap = ~terminated[t] if bootstrap_truncations else ~episode_end
        delta = rewards[t] + gamma * next_values[t] * bootstrap - values[t]
        tail = delta + gamma * lam * (~episode_end) * tail
        advantage[t] = tail
    return advantage, advantage + values


def _minibatch(model, data, sequences, burn_in=0):
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

    beginning = (left - burn_in).clamp_min(0)
    hidden = data["hidden"][beginning, env[0]].detach()
    warmup = min(burn_in, max(begin for _, begin, _ in sequences))
    if warmup:
        with torch.no_grad():
            raw_time = torch.arange(-warmup, 0, device=device)[:, None] + left
            past = raw_time.clamp_min(0)
            index = past, env[0].expand_as(past)
            past_starts = data["starts"][index]
            observations = {key: value[index].flatten(0, 1) for key, value in data["obs"].items()}
            encoded = model.encode(observations, data["previous_action"][index].flatten(0, 1),
                                   past_starts.flatten()).reshape(*past.shape, -1)
            for latent, start, valid_past in zip(encoded, past_starts, raw_time >= 0):
                _, _, warmed = model.advance(latent, hidden, start)
                hidden = torch.where(valid_past.unsqueeze(-1), warmed, hidden)
    starts = data["starts"][time, env]
    obs = {key: value[time, env].flatten(0, 1) for key, value in data["obs"].items()}
    encoded = model.encode(obs, data["previous_action"][time, env].flatten(0, 1), starts.flatten()).reshape(
        *time.shape, -1)
    logits_seq, values_seq = [], []
    for latent, start in zip(encoded, starts):
        logits, values, hidden = model.advance(latent, hidden, start)
        logits_seq.append(logits)
        values_seq.append(values)
    # Padded samples follow the valid prefix and contribute no objective or gradient.
    return (time[valid], env[valid]), torch.stack(logits_seq)[valid], torch.stack(values_seq)[valid]


def update(model, optimizer, data, *, epochs=3, sequence_length=32,
           sequences_per_batch=4, clip=.2, value_coef=.5, entropy_coef=.01,
           max_grad_norm=.5, generator=None, gamma=.99, lam=.95,
           bootstrap_truncations=True, audit_replay=False, burn_in=0):
    advantage, returns = generalized_advantage(data["rewards"], data["values"],
        data["next_values"], data["terminated"], data["truncated"], gamma=gamma, lam=lam,
        bootstrap_truncations=bootstrap_truncations)
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
    time_steps, envs = advantage.shape
    sequences = [(e, s, min(s + sequence_length, time_steps))
                 for e in range(envs) for s in range(0, time_steps, sequence_length)]
    replay_error = None
    if audit_replay:
        with torch.no_grad():
            index, logits, _ = _minibatch(model, data, sequences, burn_in)
            _, probabilities, _ = policy_stats(logits, data["actions"][index])
            replay_error = (probabilities - data["log_probs"][index]).abs().max().item()
        if replay_error > 1e-4:
            raise ValueError(f"PPO replay changed pre-update action log-probabilities by {replay_error:g}")
    stats = []
    weights = []
    for _ in range(epochs):
        order = torch.randperm(len(sequences), generator=generator).tolist()
        for start in range(0, len(order), sequences_per_batch):
            batch = [sequences[i] for i in order[start:start + sequences_per_batch]]
            index, logits, values = _minibatch(model, data, batch, burn_in)
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
            clip_fraction = (torch.abs(ratio - 1) > clip).float().mean()
            stats.append(torch.stack((loss, policy_loss, value_loss, entropy, kl, clip_fraction, grad)).detach())
            weights.append(values.numel())
    stats = torch.stack(stats)
    counts = stats.new_tensor(weights)
    result = ((stats * counts[:, None]).sum(0) / counts.sum()).tolist()
    variance = returns.var(unbiased=False).item()
    explained = 1 - (returns - data["values"]).var(unbiased=False).item() / variance if variance > 1e-8 else None
    output = dict(zip(("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm"), result))
    output["explained_variance"] = explained
    if replay_error is not None:
        output["replay_log_prob_max_error"] = replay_error
    return output
