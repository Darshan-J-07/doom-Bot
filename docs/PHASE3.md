# Phase 3 research plan implementation

The project already ships a compact custom PPO MLP with circular ray convolutions, masked entity pooling, vitals and previous actions. The visible demo streams real ViZDoom frames. Training and validation keep their headless execution. The official contract and Phase 2 results take precedence over the two research documents' provisional template and hardware assumptions.

| Research recommendation | Phase 3 decision |
|---|---|
| Custom structured PPO and an auditable fallback | Keep the existing implementation and committed release. Preserve the previous baseline tag and the current release hash. |
| Interleaved curriculum with harder progression | Compare a matched two-lock curriculum with the original training sampler. Keep category counts and the original validation set fixed. |
| Longer temporal credit assignment | Compare 0.995 with the shipped training default of 0.99 using the same model, reward and maps. The discount changes training only. |
| Vitals-conditioned masked entity attention | Add optional attention that starts with uniform weights on valid entities. Compare with the existing mean/max encoder from the same initialization and seed. |
| RGB for clues absent from structured observations | Audit map cues and real rendered observations before changing the encoder. |
| GRU, episodic exploration, prioritization and larger networks | Keep the measured GRU and reward results from Phase 2. The existing hard-tier evidence favors isolating the curriculum and attention effects before adding a compound architecture. |
| Asymmetric critic, privileged auxiliary labels and scripted teachers | The reports flag these for organizer clarification. The experiments keep privileged state within the existing training reward and diagnostics. |
| New dependencies and a framework migration | The current CPU trainer is measured and validated. The experiments use the existing Python, PyTorch, ViZDoom and official contract. |
| Validation, final benchmark, demo and submission | Use the existing validation split and a fresh two-lock development panel. The released policy's final comparison is frozen. Keep the visible demo, replay artifacts, two-page brief, and official CPU/CUDA validation. |

## Reproduce the warm-start comparison

The exact run resumes the local optimizer and RNG checkpoint `runs/phase2/frozen/mlp-200096/latest.pt`. The Git LFS release contains inference weights. The experimental output directory is intentionally ignored by Git.

```bash
py="$HOME/doom-test/.venv/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p runs/phase3-repro
"$py" -m scripts.prepare_expanded_curriculum --replace-gauntlets 3 --out runs/phase3-repro/matched-maps
for arm in control two-lock longer-horizon; do
  maps=training/maps
  gamma=0.99
  if [[ "$arm" == two-lock ]]; then maps=runs/phase3-repro/matched-maps; fi
  if [[ "$arm" == longer-horizon ]]; then gamma=0.995; fi
  "$py" -m training.train --model mlp --stage mixed --device cpu --envs 4 \
    --ticks 1400 --rollout 64 --sequence 16 --sequences-per-batch 8 --epochs 2 \
    --lr 0.0001 --gamma "$gamma" --reward legacy --map-root "$maps" \
    --min-available-mib 1536 --steps 216480 --seconds 180 --checkpoint-every 8000 \
    --resume runs/phase2/frozen/mlp-200096/latest.pt --out "runs/phase3-repro/$arm"
done
"$py" -m scripts.compare_phase2 --checkpoint runs/phase3-repro/control/latest.pt \
  --checkpoint runs/phase3-repro/two-lock/latest.pt \
  --checkpoint runs/phase3-repro/longer-horizon/latest.pt \
  --checkpoint submission/weights/best_model.pt --seeds 0 1 2 --policy-seed 0 \
  --out runs/phase3-repro/validation.json
```

Each warm-start arm added 16,384 decisions and 64 updates. The lifetime learning-rate schedule kept their effective rate at the existing 10% floor. The two-lock arm replaces three of six train gauntlets at their original sampling slots. The other 27 train maps and all validation and test files are byte-identical. Different policy trajectories can still change which episodes finish during the budget.

The fresh attention and pooling controls use the original map split, the same PPO settings and 16,384 decisions from seed 7. Use `--entity-mode attention` or `--entity-mode pool`, remove the resume argument, use the default base learning rate, and set `--steps 16384`. The attention query is conditioned on vitals. It adds 1,056 parameters and has zero initial logits for uniform valid-entity weights. Tests cover all-padding, permutation invariance, initialization equivalence and RNG parity.

The four fresh two-lock maps were generated from seeds 510000–510003. Their hashes were checked against the original and expanded curricula and the Phase 2 final suite. Each policy played all four with engine seeds 0–2 and fixed policy seed 0. These episodes are development evidence and provide no fresh final-release certification.

## Exit cues

The 50 original generated maps and three freshly generated exploration maps had 76 walkover exit linedefs. All 76 shared the same front/back texture signatures as ordinary portals. The official ray and entity type enumerations have no dedicated exit label. Real RGB captures confirmed the observation buffers carry nonzero scene information. The audit does not establish how unseen organizer maps mark switch-operated exits, and the current generated maps provide little evidence for adding RGB as an isolated change. See [exit audit](../benchmarks/phase3/exit-audit.json).

## Results and release decision

| Candidate | Broad validation proxy | Exits across 30 broad episodes | Fresh two-lock exits / survives / unlocks across 12 episodes |
|---|---:|---:|---:|
| Released MLP, 200,096 decisions | 50.86 | 21 | 0 / 1 / 2 |
| Original-map continuation | 29.79 | 17 | 0 / 2 / 1 |
| Matched two-lock continuation | 33.28 | 17 | 0 / 0 / 4 |
| Discount 0.995 continuation | 31.75 | 17 | 0 / 3 / 0 |
| Fresh mean/max pooling | 16.58 | 15 | Not evaluated |
| Fresh masked attention | 16.58 | 15 | Not evaluated |

The two-lock replacement increased unlocks on the fresh progression panel and lost survival. No policy collected both keys or opened both locks there. The longer discount increased survival in this sample and produced zero unlocks. Attention learned a nonzero query weight norm of 0.03738 and tied pooling on every recorded episode metric in its 30-episode panel. These single-seed training controls give no reason to promote a new checkpoint. The current release and its 100-episode final results stay in place.

The targeted 4/8-engine sweep ran 2,048 decisions twice per configuration with alternating order. Four serial engines had a median of 294.1 decisions/sec and about 764 MiB peak summed process RSS. Eight reached 308.8 decisions/sec with about 838 MiB peak summed RSS. Eight also showed more timing variance and a lower minimum available-memory sample. Four engines remains the conservative default for long runs.

Raw evidence is under [benchmarks/phase3](../benchmarks/phase3). It includes the training counters and checkpoint hashes, both development panels, the entity ablation, exit-cue audit and throughput sweep. The next curriculum experiment should isolate survival and the first lock-to-second-key transition before increasing the budget or combining other architecture changes.

## Verification

All 112 tests passed. The unchanged official validator passed on CPU and CUDA locally and inside the pinned organizer-recipe image with networking disabled and the repository mounted read-only. The Git LFS integrity checks passed for both the release and the recoverable baseline tag. The release weight hash matched the pre-experiment snapshot.
