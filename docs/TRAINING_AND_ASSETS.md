# Training, maps and reward accounting

`training/generate_maps.py` authors connected grid-based UDMF geometry. Generated
geometry is dedicated under CC0-1.0. No third-party community maps or commercial
DOOM WADs are downloaded. Runtime textures, monsters and sounds come from the
ViZDoom-bundled Freedoom 2 IWAD and retain its upstream licenses. ViZDoom and
Freedoom acknowledgments: https://github.com/Farama-Foundation/ViZDoom and
https://freedoom.github.io/. The official template's files retain their upstream
status; no license for the whole template is invented where none is supplied.

There are five stages: short corridors; small combat arenas; connected branching
layouts and ordinary doors; a red-card side branch before a red-locked door;
and combat plus the key dependency. Seeds, rotation, tile width, spawn jitter,
heading, monster type/placement and item placement vary. Each split has separate
seeds, and exact WAD hashes must be disjoint. These small maps are a starting
curriculum, not a substitute for broad community-map diversity. Initial health,
ammo and enemy counts are not broadly randomized in this first baseline.

`auto` initially samples movement maps and gradually admits later stages based on
the requested lifetime step target. `mixed` admits every stage immediately.
Training budget is checked between rollouts; one rollout/update can overrun the
requested wall-clock budget. Checkpointing occurs after each completed update.
Workers may be serial or spawned independently with `--parallel`. Benchmark before
increasing workers on this RAM-limited host. Spawned engines step concurrently;
network inference remains batched in the parent. Worker restart overhead matters
for short episodes.

The actual reward is transparent:

| Event | Training reward |
|---|---:|
| Exit | +20 |
| Death | -5 |
| Kill | +2 |
| New key | +4 |
| First opening of a door sector | +2 |
| First opening of a locked door sector | additional +3 |
| First visit to a 96-unit spatial cell | +0.05 |
| Damage | -0.02 per point |
| Every decision | -0.002 |

The reward tracker alone uses privileged state. It pays novelty and each door
sector once per episode, so stationary survival and repeated door operation do
not farm reward. Sector opening can be caused by an enemy; attribution to the
agent is not guaranteed. Position-cell novelty can still favor broad wandering
over task completion on large maps. Weights are initial hyperparameters, not
experimentally optimal values. No privileged feature enters the model.

Local proxy score is `100*exit +15*keys +5*doors +10*locks +3*kills
+0.1*unique_cells -0.1*damage -10*death`. It is neither the official score nor
normalized map coverage. Unique cell count is reported as a count, not a coverage
percentage. Survival includes episodes that time out; it does not imply success.
There are no invented official tier weights.

Validation maps select checkpoints. Test maps report generalization after selection
and are never used by the promotion command. Fixed evaluation seeds are recorded.
One short run per architecture cannot establish statistical superiority, especially
when its map exposure or step count differs. Check the recorded experiment budget
before treating comparisons as controlled ablations.

Training-only scripts may read maps and privileged state, launch engine processes
and write checkpoints. The submission modules cannot import these scripts.
