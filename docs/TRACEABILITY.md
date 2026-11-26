# Continuation evidence

Status remains **NOT READY**. WSL status/list commands are denied; the user reports
WSL 2 support. Seven new regression tests and the combined 42-case unaffected unit
run passed. Five engine/checkpoint/CUDA cases were explicitly deselected and
remain blocked. All 50 original WAD hashes and 30/10/10 split separation were
rechecked. No training or official Linux validation has run. See FINAL_REPORT.md,
LINUX_CONTINUATION.md and the 42-entry INDIVIDUAL_TEST_AUDIT.md for current details.

# Requirements traceability - final evidence

Status: **NOT READY**. Sources: supplied Track 1 PDF pages 3-6 and official template
commit `63b90c441ca92bcdf043de9ee3930ac6370a44bc`, contract `1.0.0-draft`.
The template controls machine-readable requirements. No hidden tests are claimed.

| Requirement | Implementation / source | Test / evidence | Status | Actual evidence / limit |
|---|---|---|---|---|
| Official origin and exact contract | Git upstream; docs/CONTRACT.md; env/doom_env/spaces.py | source/integrity test | VERIFIED | upstream commit 63b90c4; 1.0.0-draft |
| Python 3.13 / ViZDoom 1.3.3 installation | scripts/setup.ps1; requirements-training.txt | hardware.json; pip check | PARTIAL | versions installed; engine IPC denied |
| Exact DoomAgent interface | submission/agent.py | official-validator.txt | INTERFACE PASS | constructor/reset/act signatures pass; init lacks weights |
| Ray/entity/vitals encoding; empty slots; finite extrema | submission/model.py; preprocessing.py | test_valid_spatial_observations; test_vitals_and_extremes; pooling | UNIT PASS | 24 spatial combinations, four vitals cases, permutation test |
| Learned valid actions only | submission/model.py | unit sampled actions; static import audit | PARTIAL | encoding valid; no trained parameters |
| Genuine weights loaded once | submission/agent.py; weights/ | test_checkpoint_load_once_and_readonly_act | BLOCKED | no checkpoint; no fallback weights |
| Recurrent memory and cross-map reset | submission/agent.py; model.py | test_memory_reset_and_independent_instances | IMPLEMENTED / BLOCKED | agent lifecycle test cannot construct without trained weights |
| Custom PPO and termination/truncation correctness | training/ppo.py | analytic GAE; MLP/GRU finite-gradient updates | UNIT PASS | synthetic inputs only, no DOOM learning claim |
| Checkpointing / resume / wall-clock budget | training/train.py | training-smoke.txt | IMPLEMENTED / BLOCKED | fails at engine initialization before first update |
| Five-stage legal curriculum and split isolation | training/generate_maps.py; manifest.json | test_map_splits_disjoint; engine map-load test | PARTIAL | 50 files generated; hashes disjoint; engine loading blocked |
| Reward shaping and one-time novelty/door rewards | training/rewards.py | test_reward_novelty_and_door_rewards_are_once_only | UNIT PASS | one-time reward accounting; actor attribution limitation disclosed |
| Combat / survival / navigation / doors / keys / backtracking / exits | training/evaluate.py; generated stages | real scenario metrics | UNMEASURED | engine blocked, no behavior evidence |
| Unseen-map generalization and ablations | validation/test splits; evaluator | held-out runs and promotion | IMPLEMENTED / UNMEASURED | no gameplay score or architectural improvement claim |
| CPU / CUDA options and parallel workers | training/train.py; workers.py | CPU unit tests; CUDA test | PARTIAL | CPU tested; CUDA skipped; worker throughput blocked |
| Offline / read-only / engine-free inference | submission modules | source import allowlist; readonly runtime test | SOURCE REVIEW / RUNTIME BLOCKED | runtime guard test requires missing trained weights |
| Official initialization / reset / act / map time budget | env/doom_env/limits.py; benchmark scripts | official validator; performance benchmark | UNVERIFIED | model-only timings are explicitly not agent timing |
| 2 GiB VRAM / 8 GiB RAM / one CPU | official limits; scripts/benchmark.py | resource benchmark / Linux validator | UNVERIFIED | only CPU model-process memory measured |
| Untouched wrapper and validator | env/doom_env/*.py; validate.py | test_official_files_unchanged | PASS | bytes match upstream apart from CRLF |
| Pinned extras / no submitted Dockerfile | submission/requirements.txt; docs/organizer-base-image.Dockerfile.txt | official requirements check / source audit | PASS | zero extra inference packages; original Dockerfile preserved as .txt |
| Repository <= 1 GiB including LFS | Git tracked payload / repository-size.json | final official validator size check | SIZE PASS ONLY | no model or LFS object to include yet |
| LFS trained weights | .gitattributes; submission/weights/ | git lfs ls-files; submission-audit.json | NOT SATISFIED | attributes ready, no real model to track |
| Professional README and complete pipeline source | README.md; training/; docs/ | source package inspection | DELIVERED | engine-facing pipeline not end-to-end verified |
| Two-page brief and technical defense | docs/project-brief.pdf; DEMO_AND_QA.md | page count and rendered visual inspection | DELIVERED | brief clearly states untrained / NOT READY |
| Reproducible successful demonstration | scripts/demo.py | recording and actual metrics | IMPLEMENTED / BLOCKED | no video or successful replay fabricated |
| Official validator passes | validate.py | official-validator-final.txt | FAIL | missing best_model.pt; exit 1 |
| Official Linux organizer image / RTX3090 | official image and evaluator | docker-validator.txt | BLOCKED / UNVERIFIED | Docker absent; CPU-only local Torch; versions differ |
| Public repository created during hackathon | user-owned GitHub template repository | public URL and event dates | OUTSTANDING | only local checkout; no publication authorized |
| Hidden score / Top 10 / Top 2 / first place | organizer scorer | organizer results | UNKNOWN | no hidden evaluations available |

The full suite ran with no organizer checks weakened: 47 passed, 29 failed, 12
setup errors, one CUDA skip. See FINAL_REPORT.md and raw benchmarks for the exact
failures. A unit-pass status never implies successful gameplay or readiness.
