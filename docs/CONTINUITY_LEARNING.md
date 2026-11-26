# Continuity campaign learner profiles

The independent controller schedules these profiles through `scripts.continuity_learning`.
They write only to the new campaign directory. The official observation contract,
factorized action head, engine wrapper and published weights stay as archived.

## Prepare the mastery map pool

```bash
cd /home/darshan/projects/doom-nexus
py=~/doom-test/.venv/bin/python
run=runs/continuity-campaign-20261127
"$py" -m scripts.continuity_learning prepare --out "$run" --map-root "$run/post-lock-maps"
```

Preparation combines the approved legacy, guarded-progression, closed-lock and new
post-lock training maps. It checks WAD identities and disjoint map hashes, geometry
and families against the familiar and switch development partitions. The resulting
pool has 190 unique training WADs. Its manifest and the exact source hashes are in
`continuity-profile.json`. No reserved final-assessment manifest is used.

## Profile comparisons

| Profile | Change from its control | Optimizer on initialization |
|---|---|---|
| `mixed` | Fixed mixed practice control | Exact matching PPO Adam |
| `retention-low` | Half learning rate | Exact matching PPO Adam |
| `retention-anchor` | Forward KL to the frozen approach policy at coefficient 0.06 | Exact matching PPO Adam |
| `retention-distill` | Same KL, with the BC-PPO teacher's USE group supplying targets near a visible lock when a key is held | Exact matching PPO Adam |
| `retention-safe` | Quarter learning rate with the same 0.06 warm-policy KL | Exact matching PPO Adam |
| `memory-control` | Matched MLP control for memory migrations | Fresh Adam on the first block start |
| `gru` | Residual GRU and a zero initial memory readout | Fresh Adam after migration |
| `occupancy` | Ego-centric occupancy state with a zero initial readout | Fresh Adam after migration |
| `count` | First-visit map-cell shaping bonus capped at 1.2 per episode | Exact matching PPO Adam |
| `rnd` | Bounded random-network-distillation bonus on legal ray geometry | Exact matching PPO Adam, separate predictor optimizer |
| `mastery` | Family-level adaptive sampling in fixed skill lanes, at 0.55 of the original learning rate | Exact matching PPO Adam |

For comparisons, match seeds, parent, map root and rollout size. Use `memory-control`
for the architecture comparison. Run both exploration variants against `mixed`.
The choice of training teacher affects only a loss. The exported student generates
actions from official observations using the same decoder as the archived policy.
RND parameters and adaptive sampling history are checkpointed for training and
excluded from the inference export. The controller, evaluation protocol and paired
results determine advancement.

## Metered learner and recovery

```bash
"$py" -m scripts.continuity_learning learner --out "$run" \
  --label retention-59 --meter-id retention-59 --profile retention-anchor --seed 59 \
  --source runs/limit-campaign-20261127/training/approach-41/step-314496.pt \
  --map-root "$run/post-lock-maps" --active-seconds 600
```

This is a single-block example. The controller owns the campaign's time and
resources. A recovery attempt uses a new `--label`, the previous saved `latest.pt`
as `--source` and the same `--meter-id` and cumulative `--active-seconds`. Read the
atomically replaced `training-progress.json` and matching optimizer checkpoint to
confirm committed learning. The profile defaults to the fastest measured stable
eight-environment CPU rollout from the previous hardware comparison. Override
`--rollout` consistently for a matched workload if benchmarking requires it.

Pass `--visualize --speed 1` with a display-capable Ubuntu terminal for optional
live PPO rollouts. This opens one native training environment. Headless training
uses no window. The separate checkpoint viewer leaves the training policy and
optimizer alone.

## Supervised optimization

```bash
"$py" -m scripts.continuity_learning collect --out "$run" \
  --map-root "$run/mastery-training" --seed 3159 --maps 48 --seconds 210
"$py" -m scripts.continuity_learning clone --out "$run" \
  --label imitation-59 --meter-id imitation-59 --seed 59 --max-updates 240 \
  --source runs/limit-campaign-20261127/training/approach-41/step-314496.pt
```

Collection keeps successful native-engine demonstrations and successful
recoveries after learner-controlled prefixes. It earns zero active-time credit.
Cloning counts each saved supervised optimizer update using monotonic time. The
warm reference and the trajectory weights retain the original mix of movement,
strafe, turn and use targets. The PPO checkpoint preserves all unchanged Adam
moments and resets the actor moments after changing that head. Feed the clone
checkpoint to a matching PPO profile for the continuation. The earlier pinned
successful trajectories are the fallback if the new dataset was never collected.
