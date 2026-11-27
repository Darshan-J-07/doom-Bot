# DOOM NEXUS campaign findings

The campaign completed three hours of saved active learning. The 12 experimental pairs produced 24 exports. None cleared the two-seed advancement gate. The published submission stays in place, and all 48 reserved layouts remain sealed.

## Development results

Each familiar tier has six layouts and five engine seeds. Episode counts are out of 30 per tier and 120 overall. The published and exploration baselines are archived results on the same paired familiar panel.

| Artifact | T1 combat | T2 exploration | T3 keys and locks | T4 gauntlet | Total | T4 deaths |
|---|---:|---:|---:|---:|---:|---:|
| Published | 1 | 5 | 0 | 0 | 6 | 15 |
| Frozen exploration | 1 | 15 | 0 | 0 | 16 | 25 |
| Starting policy | 5 | 25 | 0 | 0 | 30 | 20 |
| BC + PPO, seed 59 | 5 | 25 | 15 | 0 | 45 | 21 |
| BC + PPO, seed 83 | 10 | 15 | 0 | 0 | 25 | 21 |
| GRU, seed 59 | 8 | 10 | 5 | 0 | 23 | 22 |
| GRU, seed 83 | 6 | 15 | 5 | 1 | 27 | 20 |
| RND, seed 59 | 11 | 20 | 0 | 0 | 31 | 20 |
| RND, seed 83 | 13 | 20 | 0 | 0 | 33 | 20 |

The 24 exports completed 6,048 development episodes across 68 independent layouts. The switch panel used 32 layouts in four families with three seeds per layout. The familiar panel used 24 layouts in 12 families and five seeds. The separate conditional post-lock panel used 12 layouts in two families and three seeds. Its warm baseline added 36 newly run episodes. Archived reference rows are also included in the CSV.

## Switch approach and use

The far-switch panel consists of 18 layouts from three families with three engine seeds each.

| Policy | Exit area | In range | Faced in range | Attempt | Engine exit | Near-start exits | Walkovers |
|---|---:|---:|---:|---:|---:|---:|---:|
| Starting policy | 39/54 | 21/54 | 21/54 | 18/54 | 18/54 | 12/18 | 9/24 |
| RND, seed 59 | 51/54 | 33/54 | 27/54 | 21/54 | 21/54 | 12/18 | 12/24 |
| RND, seed 83 | 42/54 | 36/54 | 27/54 | 15/54 | 15/54 | 3/18 | 21/24 |

Each logged far-switch attempt ended in an engine-confirmed exit. RND improved range entry on four and five layouts, while the fraction of range entries converted to an exit fell from 18/21 to 21/33 and 15/36. The family bootstrap intervals for the range improvement are 0 to +50 percentage points in each seed, based on only three families. The paired familiar gain of the BC seed 59 averaged +7.5 points across families, with an interval of −5.8 to +26.7 points.

RND seed 59 passed its individual gate. Seed 83 lost near-switch alignment and far-switch completion. Conditional practice starts with an unlocked door and has a separate endpoint. Seed 59 finished 15/30 conditional post-lock switches, compared with 9/30 for the warm baseline, while conditional walkovers fell from 3/6 to 0/6.

## Progression and survival

- BC seed 59 collected both keys and crossed both relevant locks in all 15 Tier 3 wins. Those wins used walkover exits and came from three layouts in one family. Seed 83 completed none. Both BC seeds lost half the warm policy's near-switch completions.
- Both GRU seeds completed the key-to-exit chain on `early-lock-002` across its five engine seeds. The two feedforward memory controls completed zero Tier 3 episodes. The same feedforward controls went 16/30 and 14/30 on Tier 1, compared with 5/30 for the warm policy, with lower incoming damage and more kills. Their observed ammunition decreases rose, and that measurement is a lower bound.
- The only full gauntlet switch-chain success came from `p2-03-retention-low-59` on `hub-vault-000`, engine seed 3. It crossed matching locks at tics 1152 and 2080, pressed fresh USE, and exited at tic 2592 with 88 HP. On seed 0 the same agent crossed both doors and died at tic 2443 after taking 102 damage. Its other three seeds on the layout died too. The GRU and mastery gauntlet wins were walkovers on the same `loop-vault-001` seed.
- The report records unknown traversal for archived episodes whose original record lacked a trace. Fifteen preserved warm replays match the frozen policy, WAD and MAP lump, engine seed, action hash and archived outcomes. Rechecking trace SHA-256s and sampling the doorway crossings resolves the warm count to 5/30 in Tier 3 and 5/30 in Tier 4. No warm episode completed either chain. The replay receipt is local at `runs/adaptive-campaign-20261127/warm-lock-replays/index.json`.

## Experiment interpretation

The comparisons used matched seeds and stayed close to 450 active seconds per PPO block. Decision counts vary with throughput. The GRU and occupancy comparisons have fresh-Adam feedforward controls. The post-lock learning-rate comparison shares its map pool. Count-based novelty and RND share the mastery map pool, fixed skill lanes, and learning rate. The unshaped mastery profile uses different sampling and learning rate, so the data cannot attribute the replicated range gain to RND alone. The earlier mixed-versus-sequential results remain archived and were not rerun in this campaign.

The next test should compare RND on and off with the same map pool, sampler, decision budget and paired training seeds. Use fresh development families and hold the existing retention gates. Measure the fraction of range entries that end in a confirmed switch exit as a primary endpoint.

## Runtime and audit

The saved active duration was 10,800.023 seconds, with 3,505,152 decisions, 3,423 PPO updates and 480 supervised updates. UTC timestamps span 3h 19m 35s through report generation. The controller's steady clock recorded 3h 10m 32s across the same interval on this WSL host. All learning credit was reconciled against numbered checkpoint counters. The evaluation process ran concurrently with PPO for most of the campaign.

The learner used eight headless CPU environments on the Core i5-12450HX. Earlier matched profiling on the same laptop measured 325.7 decisions/s for that configuration and 198.5 decisions/s for the CUDA hybrid. The campaign averaged 324.8 PPO decisions per credited PPO second. The local GPU was an RTX 2050.

The ranked development checkpoint was `p3-02-gru-83`. All 150 tests and exact-artifact parity passed. The unchanged validator passed twice per device in the local offline, read-only organizer-recipe container. The campaign also recorded one passing host validator run per device. The background service lacked the WSL `nvidia-smi` directory on its PATH, so its host CUDA timing entry is unverified in the original compliance receipt. The follow-up used the same packaged source and weight hashes and passed all five fresh CUDA runs, with a 1.52-second maximum first action and a 9.30 ms mean decision round trip. No recorded new development episode had an invalid action, inference timeout, engine error or crash.

The published weight SHA-256 before and after the campaign was `fe14608eaa2728ebe2389b7940144713ab6c0ce39da28a1aba24f576005a463f`.

## Artifacts

[HTML report](report.html) · [Machine-readable report](report.json) · [Episode records](episodes.csv) · [Compliance](compliance.json) · [Experiments](experiment_manifest.json) · [Checkpoints](checkpoint_manifest.json) · [Learning curves](learning_curves.json) · [Reproduction and native gameplay](reproduction_commands.md) · [Map manifests](map_manifests.json) · [Host CUDA follow-up](host-cuda-recheck.json)

The checkpoint manifest contains source and weight hashes. Run-local engine maps, checkpoints, raw trajectories, and optimizer states remain in the training workspace. The commands use those archived inputs, and the supervised collection and cloning commands are in the repository's [learner documentation](../../docs/CONTINUITY_LEARNING.md).
