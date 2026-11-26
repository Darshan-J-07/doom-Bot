# DOOM NEXUS

A custom spatial reinforcement-learning implementation for DevForge Track 1.
**NOT READY:** the user has verified Ubuntu engine/CUDA execution, but this Codex
session still cannot access WSL (`Wsl/E_ACCESSDENIED`). Project training, weights,
held-out results and successful official validation remain outstanding.
Start with [the measured final report](docs/FINAL_REPORT.md). The original starter
README is preserved in `docs/upstream-README.md`.

## Measured status

Latest user-verified environment: Ubuntu 24.04 / WSL 2, Python 3.13.18, ViZDoom
1.3.3 and Torch 2.11.0+cu128, with RTX 2050 CUDA computation and actual engine
actions passed. Reuse `~/doom-test/.venv`; do not downgrade this working stack.
Run the [verified-environment launcher](docs/VERIFIED_ENVIRONMENT_RUN.md) in the
normal Ubuntu terminal to copy this existing checkout into Linux home and start
the complete project workflow. It preserves the Windows original.

The latest local unaffected/control run has **51 passed, 5 explicitly deselected**.
The deselected cases still require actual engine/weights/CUDA access from this
session. CPU/CUDA device selection now benchmarks real PPO runs, and CPU
checkpoint saving avoids unnecessary CUDA initialization. None of these unit
results is evidence of genuine game training or held-out skill.

The remaining paragraphs and table below retain previous-run history:

Continuation: WSL exists, but this session receives `Wsl/E_ACCESSDENIED` for
status and distro enumeration. The user reports version 2 support. Ubuntu and
WSL CUDA are not yet verified. Start with the exact manual commands in
[Linux continuation](docs/LINUX_CONTINUATION.md).

The latest unaffected-unit run has **42 passed, 5 explicitly deselected**;
the five engine/checkpoint/CUDA cases remain blocked. Seven new regression
cases verify actual map-byte separation and saved decision-mode parity. All 50
maps were preserved and their hashes rechecked. Every previous failed, errored
and skipped case has an [individual audit](docs/INDIVIDUAL_TEST_AUDIT.md).

The following table records the **previous full execution**, before continuation:

| Check | Actual result |
|---|---|
| Local acceptance cases | 35 passed, 4 failed, 1 CUDA skip |
| Full suite including organizer cases | 47 passed, 29 failed, 12 errors, 1 skip |
| Real training / selected checkpoint | 0 decisions; no trained weights |
| Official validator | Exit 1: checkpoint missing; interface check passes |
| Docker validator | Exit 2: Docker not installed |
| Untrained model-only CPU mean latency | MLP 3.337 ms; GRU 5.575 ms; 512 synthetic inputs each |

The model-only timing is not submitted-agent or RTX3090 performance. Engine IPC,
missing weights and eight Linux-specific upstream cases account for all failing
or error cases. [Raw results](benchmarks/full-tests-utf8.txt) and
[failure categories](benchmarks/test-failure-categories.json) preserve the evidence.
No gameplay skill, learning curve or hidden-map score has been established.
All 50 generated maps passed static WAD metadata parsing; that does not establish
successful engine loading. Evidence: `benchmarks/static-map-check.json`.

New Linux entry points are `bash scripts/setup_linux.sh cuda` and
`bash scripts/run_linux.sh cuda`, executed manually inside Ubuntu. Use `cpu`
for both if CUDA compute checks fail. These scripts have not run in this sandbox;
they retain per-stage logs and stop at the first failure. The full guide covers
Ubuntu installation, Python 3.13.9, ViZDoom 1.3.3 and the existing checkout path.
That fresh-install route is historical. The current preferred launcher reuses
the verified 3.13.18 / Torch 2.11 environment and skips Torch installation.

This local checkout originates at the official template commit
`63b90c441ca92bcdf043de9ee3930ac6370a44bc` and implements contract `1.0.0-draft`.
The official wrapper, limits and validator are preserved unchanged. There is no
published team repository yet, and event-time eligibility must be checked by the
team. Hidden leaderboard performance is unknown.

## What the project implements

The implemented policy encodes 64 circular raycasts, masked visible entities and 17
vitals. A shared fusion layer receives previous action bits. An optional 128-unit
GRU supplies compressed history; a feed-forward ablation bypasses that memory.
Six learned categorical heads produce the nine official binary buttons, with no
opposing movement/turn commands. A critic supplies training value estimates.
There is no rule-based aiming, pathfinding, map lookup or engine access in the
submission. RGB and depth are not used in this configuration.

PPO is implemented directly: clipped policy and value objectives, GAE, entropy,
normalized advantages, finite-gradient checks, gradient clipping, learning-rate
decay and contiguous recurrent minibatches. Self-authored maps cover movement,
combat, exploration, red-card progression and a combined scenario. Training-only
privileged state supplies shaped reward; it never enters model features. Model
selection uses a validation split and test reporting uses a separate split.

These are implemented mechanisms, not claims of mastered gameplay. Only the
recorded episodes establish success rates. Read [architecture](docs/ARCHITECTURE.md),
[training and assets](docs/TRAINING_AND_ASSETS.md), [traceability](docs/TRACEABILITY.md)
and [demo/Q&A](docs/DEMO_AND_QA.md).

## Windows setup

From this repository in PowerShell, create an isolated environment:

```powershell
.\scripts\setup.ps1 -Python 'C:\Users\Darshan\anaconda3\python.exe'
$Python = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$env:PYTHONUTF8 = '1'
& $Python -m scripts.diagnose
& $Python -m training.generate_maps
```

The supplied setup script installs the tested CPU PyTorch 2.9.1 wheel and pinned
training packages. `-Cuda` instead requests its CUDA 12.8 wheel; CUDA support must
be verified on the installed driver before training. Do not count a CPU-only
test as a CUDA result. The official evaluation image currently locks newer
versions; it must be tested separately before submission. Inference imports only
PyTorch, NumPy and Python standard libraries, all supplied by that image, so
`submission/requirements.txt` requires no extra packages.

This build's existing environment is outside the deliverable repository. To use
it without reinstalling, from this exact workspace checkout:

```powershell
$Python = (Resolve-Path '..\..\work\.venv\Scripts\python.exe').Path
```

Linux setup uses Python 3.13, `python -m venv .venv`, installs the appropriate
PyTorch wheel, then `pip install -r requirements-training.txt -e env`. Evaluation
must use the organizer base image. The starter Dockerfile is preserved verbatim
as `docs/organizer-base-image.Dockerfile.txt`, not as a submitted Dockerfile.

## Train and resume

Benchmark workers before selecting their count. CPU threads are limited to one
by the training and inference code. The environment pool is serial by default;
`--parallel` enables separate spawned engine processes and costs more RAM.

```powershell
& $Python -m scripts.benchmark --throughput-only
& $Python -m training.train --config configs/baseline.json --out runs/mlp
& $Python -m training.train --config configs/recurrent.json --out runs/gru
& $Python -m training.train --model gru --stage auto --steps 2000000 --seconds 3600 --resume runs/gru/latest.pt --out runs/gru
```

`--steps` is the total lifetime target, including resumed steps. `--seconds` is a
per-invocation wall-clock budget, checked between rollouts. A resume starts fresh
episodes, restores model/optimizer/RNG state and preserves counters. Use `--envs`,
`--rollout`, `--sequence`, `--epochs`, `--ticks`, `--lr`, `--entropy`, `--seed` and
`--device cpu|cuda` to match hardware and time. Config JSON supplies defaults;
explicit command-line values override them. Checkpoints are replaced atomically
after every completed optimizer update. Interrupted in-progress rollouts are lost.

## Evaluate and choose weights

```powershell
& $Python -m training.evaluate --checkpoint runs/mlp/latest.pt --checkpoint runs/gru/latest.pt --split validation --seeds 0 1 --ticks 2100 --promote --out benchmarks/validation.json
& $Python -m training.evaluate --checkpoint submission/weights/best_model.pt --split test --seeds 0 1 --ticks 2100 --out benchmarks/heldout.json
& $Python -m scripts.benchmark --steps 512
```

Promotion exports a small inference-only `submission/weights/best_model.pt` from
the highest local proxy mean on validation maps. Test-split promotion is rejected.
The proxy formula is documented and is not an estimate of official leaderboard
points. Stochastic inference uses a private generator reset for each map;
`--deterministic` is an explicit evaluation/selection option, not a heuristic.
Additional training time should be spent on varied vanilla maps and multiple
seeds, with broad held-out progression results before optional architectural
experiments.

## Tests, validator and demonstration

```powershell
& $Python -m pytest tests/test_nexus.py tests/test_env.py -q -ra
& $Python validate.py
& $Python -m scripts.check_submission
& $Python -m scripts.demo --stage gauntlet --seed 0 --out recordings/demo
```

After the engine and training blockers are resolved, the demo command saves an
engine replay, raw RGB frames and a metric JSON. No replay or GIF was produced in
this build. Upstream Linux-only tests
may fail on Windows because they import `resource` or use `sched_getaffinity`;
the full suite result and platform limitations must be reported explicitly.
No official check is weakened. On an appropriately configured Linux GPU host:

```bash
python validate.py --docker --image ghcr.io/reinforce-sst/devforge-doom:latest
```

This checks the real organizer image, offline/read-only isolation and resource
limits. Local Windows CPU measurements alone cannot certify that container or
RTX3090 performance.

The two-page project brief and final report use actual saved measurements only.
No model is downloaded at inference, and a missing/untrained checkpoint is an
error rather than a silent random fallback. Git LFS attributes cover `.pt` files.
Train intermediates and replays are ignored; future selected weights and compact
evidence belong in the submitted tree. Public GitHub creation/push is a separate,
user-authorized step.

## Layout

```text
submission/       neural model, preprocessing, exact DoomAgent, selected LFS weights
training/         custom PPO, rewards, map generation, curricula, workers, evaluator
training/maps/    self-authored train/validation/test maps and hash manifest
configs/          baseline and recurrent defaults
tests/            original organizer tests plus local agent/training acceptance tests
scripts/          setup, hardware/resource benchmark, audit, demonstration
benchmarks/       measured training/evaluation/test/validator evidence
docs/             official rules and contract, design, traceability, report and Q&A
validate.py       untouched official validator
```

Dependencies retain their upstream licenses: PyTorch, NumPy, Gymnasium, ViZDoom,
Freedoom, pytest and psutil. Generated map geometry is CC0-1.0; runtime assets
retain Freedoom licensing. No commercial IWAD is distributed by this project.
