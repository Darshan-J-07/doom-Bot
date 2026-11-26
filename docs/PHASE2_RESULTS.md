# DOOM NEXUS Phase 2 results

**BASELINE READY.** The committed release uses the MLP at 200,096 cumulative decisions and 878 PPO updates. The original inference artifact is recoverable at Git tag `phase2-baseline` and in `runs/phase2/baseline/best_model.pt`.

Release SHA-256 `fe14608eaa2728ebe2389b7940144713ab6c0ce39da28a1aba24f576005a463f`.

## Training and selection

| Run | Cumulative decisions | Updates | Decisions/sec during invocation | Five-seed validation proxy |
|---|---:|---:|---:|---:|
| milestone-100k | 100,000 | 487 | 295.9 | 30.60 |
| milestone-250k | 250,000 | 1,073 | 281.5 | 30.87 |
| Selected periodic snapshot | 200,096 | 878 | Periodic checkpoint | 38.88 |

The original baseline scored 26.47 on the five-seed validation panel. Training stabilized at about 282 decisions/sec across the 150,000-decision continuation. The 250,000-decision checkpoint scored below the earlier snapshot and still completed no Tier 4 episodes. Another 250,000 decisions would take about 14.8 minutes at the measured rate. The next compute should test a targeted progression or combat change, starting with a matched budget.

The short matched control favored the GRU on the two-seed panel and the zero-memory intervention removed its observed Tier 3 unlocks. The same GRU scored 17.79 on the five-seed panel, below the preserved MLP. The evidence supports useful recurrent state in some situations. Robust gains across different seeds remain unverified. The progress reward control also scored below the legacy control, so both milestones used legacy shaping.

## Frozen comparison on 100 unseen episodes per policy

Five new procedural maps per tier and five engine seeds per map. Both policies used the fixed deployment reset seed. The tier-weighted score is a local proxy.

| Tier | Exit count, baseline → release | Survived, baseline → release | Keys, baseline → release | Doors, baseline → release | Locks, baseline → release |
|---|---:|---:|---:|---:|---:|
| Combat | 11 → 23 | 11 → 23 | 0 → 0 | 0 → 0 | 0 → 0 |
| Exploration | 20 → 20 | 25 → 25 | 0 → 0 | 20 → 15 | 0 → 0 |
| Keys | 0 → 0 | 25 → 25 | 15 → 20 | 0 → 0 | 0 → 0 |
| Gauntlet | 0 → 0 | 2 → 0 | 1 → 5 | 0 → 1 | 0 → 1 |

| Tier | Kills/episode, baseline → release | Damage/episode, baseline → release | Unique cells/episode, baseline → release |
|---|---:|---:|---:|
| Combat | 0.00 → 0.00 | 94.56 → 50.52 | 12.16 → 9.56 |
| Exploration | 0.00 → 0.00 | 0.00 → 0.00 | 15.60 → 18.20 |
| Keys | 0.00 → 0.00 | 0.00 → 0.00 | 18.80 → 16.60 |
| Gauntlet | 0.44 → 0.44 | 109.56 → 112.44 | 7.84 → 8.48 |

The weighted proxy increased from 16.41 to 23.58. Exit count increased from 31 to 43, and survival from 63 to 73. Total kills were 11 for both. The approximate paired map-bootstrap interval for the proxy difference was -4.95 to 19.20. Five maps per tier leave substantial map-selection uncertainty.

All of the extra exits came from Tier 1. Tier 3 key pickup improved without any final-suite unlocks. Tier 4 produced one unlock, zero exits and zero surviving challenger episodes. The 85–95% exit target and lasting hard-tier robustness remain unmet.

## Throughput on this laptop

| PPO configuration | Decisions/sec | Peak summed process-tree RSS | PyTorch reserved VRAM |
|---|---:|---:|---:|
| cpu-1 | 217.6 | 700 MiB | 0 MiB |
| cpu-2 | 285.3 | 720 MiB | 0 MiB |
| cpu-4 | 337.8 | 759 MiB | 0 MiB |
| cpu-4-workers | 83.5 | 2898 MiB | 0 MiB |
| cpu-4-long | 334.0 | 762 MiB | 0 MiB |
| cuda-4 | 155.0 | 1905 MiB | 282 MiB |
| cpu-4-large-batch | 339.2 | 762 MiB | 0 MiB |

Sum of RSS can count shared pages more than once. PyTorch reserved VRAM omits other GPU allocations. The CPU minibatch test used four serial engine environments, and the submission still runs on one CPU thread.

## Reproducible evidence

- `benchmarks/phase2/release-decision.json` and `final-comparison.json` store the release decision and both sets of episodes.
- `benchmarks/phase2/checkpoint-freeze.json` stores the validation ranking and frozen training digest.
- `recordings/phase2/dashboard-final/index.html` plays paired recorded runs on an illustrative validation scenario. The original LMP files and per-decision traces are in the adjoining replay directories.
- `PHASE2_FAILURE_ANALYSIS.md`, `PHASE2_PROTOCOL.md`, and `PHASE2_DEMO.md` describe the failure evidence, controlled runs and demo.
