# DOOM NEXUS final report

**Classification: NOT READY.** Implementation and unit evidence are preserved,
but no real engine rollout, trained checkpoint or successful official validation
was obtained. The user explicitly chose to package with this blocker.

## 1. Completed features

Official-template checkout and contract extraction; exact DoomAgent signatures;
ray/vitals/entity preprocessing; custom spatial actor-critic with optional GRU;
learned factorized actions; custom sequence PPO/GAE; reward tracking; five-stage
procedural maps; disjoint splits; checkpoint/resume code; CPU/CUDA device options;
optional spawned engine workers; validation-only checkpoint promotion; proxy
evaluation; resource tools; source audit; demonstration capture; setup commands,
architecture notes, requirements traceability and this two-page project brief.

These features are implemented, with partial unit coverage. The engine-facing
training, resume, recurrent-agent lifecycle, gameplay evaluator and demo are NOT
end-to-end verified. Hierarchical control, attention and curiosity were deferred
because no valid trained baseline exists yet.

## 2. Repository and source of truth

Local project: `outputs/doom-nexus`. Upstream remote is named `upstream` to avoid
mistaking this clone for a publishable team repository. Inspected commit:
`63b90c441ca92bcdf043de9ee3930ac6370a44bc`; contract `1.0.0-draft`.
Official source: https://github.com/Reinforce-SST/devforge-doom-template.

`env/doom_env/*.py` and `validate.py` match that commit, allowing only checkout
line-ending differences; the integrity acceptance test passed. The original
README and Dockerfile are preserved as documentation references. No file named
Dockerfile remains in the deliverable tree. No public repository was created or
pushed. Actual event dates and event-time eligibility have not been established.

```text
submission/       DoomAgent, model, preprocessing; no trained checkpoint yet
training/         PPO, rewards, generator, workers, train/evaluate CLIs
training/maps/    30 training + 10 validation + 10 test maps, hash manifest
configs/          baseline MLP / recurrent GRU defaults
tests/            unmodified upstream tests + 40 local acceptance cases
scripts/          setup, diagnostics, benchmark, audit, demo
benchmarks/       raw logs, JUnit, packages, hardware and microbenchmark evidence
docs/             official contract/rules, design, traceability, report and Q&A
validate.py       unmodified official validator
```

## 3. Architecture

64 normalized ranges plus learned six-class hit embeddings and fixed sensor
bearings pass through circular 1D convolutions. The entity MLP uses shared weights,
masked mean/maximum and count over the official 32 padded slots. Seventeen scaled
vitals and nine previous-action bits join these features in a 202-to-128 fusion.
The optional GRU has 128 hidden units. Six categorical action groups emit the nine
official buttons. The MLP variant has 84,232 parameters; the GRU
variant has 183,304. Neither has trained parameters in this package.
No RGB, absolute coordinates, engine reference or training-only feature reaches
inference. See `docs/ARCHITECTURE.md` for the four-tick terminal-observation
bootstrap approximation and fresh-episode resume semantics.

## 4. Exact setup commands

In normal PowerShell, from the repository directory:

```powershell
.\scripts\setup.ps1 -Python 'C:\Users\Darshan\anaconda3\python.exe'
$Python = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$env:PYTHONUTF8 = '1'
& $Python -m scripts.diagnose
& $Python -m training.generate_maps
```

The already-installed environment for this exact workspace can instead be used
with `$Python = (Resolve-Path '..\..\work\.venv\Scripts\python.exe').Path`.
The environment is intentionally outside the distributable repository and is
not included in the ZIP. `pip check` passed. A clean project-local Python 3.13
environment was installed; full engine installation acceptance remains blocked.

## 5. Exact training and resume commands

Run outside the current IPC-restricted sandbox, or on a suitable Linux host:

```powershell
& $Python -m scripts.benchmark --throughput-only
& $Python -m training.train --config configs/baseline.json --envs 1 --out runs/mlp
& $Python -m training.train --config configs/recurrent.json --envs 1 --out runs/gru
& $Python -m training.train --model gru --stage auto --steps 2000000 --seconds 3600 --envs 1 --resume runs/gru/latest.pt --out runs/gru
```

Start with one environment because available RAM was low. The worker benchmark
could not run, so no optimal worker count or throughput is asserted. CUDA setup
is available via `setup.ps1 -Cuda`, followed by `--device cuda`; it is untested.

## 6. Exact evaluation and validation commands

After successful real training:

```powershell
& $Python -m training.evaluate --checkpoint runs/mlp/latest.pt --checkpoint runs/gru/latest.pt --split validation --seeds 0 1 --ticks 2100 --promote --out benchmarks/validation.json
& $Python -m training.evaluate --checkpoint submission/weights/best_model.pt --split test --seeds 0 1 --ticks 2100 --out benchmarks/heldout.json
& $Python -m pytest -q -ra
& $Python validate.py
& $Python -m scripts.benchmark --steps 512
& $Python -m scripts.check_submission
& $Python -m scripts.demo --stage gauntlet --seed 0 --out recordings/demo
```

On a Linux GPU host with Docker and the official image available:
`python validate.py --docker --image ghcr.io/reinforce-sst/devforge-doom:latest`.
All commands above are implemented interfaces; gameplay commands have not
completed successfully in this session.

## 7. Actual test results

Final UTF-8 full suite: **47 passed, 29 failed,
12 setup errors, 1 skipped** out of
89 cases, in 46.53 seconds; exit code 1.
Local project tests: **35 passed, 4 failed, 1 skipped** out of 40.
The four local failures are one blocked map-load check and three missing-weight
agent lifecycle/episode checks. The skipped test is CUDA, explicitly unavailable
in this CPU PyTorch build. It is not a claim that CUDA hardware is absent.

Across the full suite, all 41 failing/error cases are classified in
`benchmarks/test-failure-categories.json`: 29 engine IPC cases, four missing-weight
cases, and eight upstream Windows platform cases (seven import the Unix
`resource` module; one requires `os.sched_getaffinity`). No organizer test was
altered or silently omitted. The first run had 43 passes, 33 failures and 12
errors; setting `PYTHONUTF8=1` fixed four subprocess-output failures. Both raw
runs are retained. Analytical GAE boundary tests and finite PPO parameter-update
tests passed; these use synthetic contract-valid inputs and are not DOOM training.

Evidence: `benchmarks/full-tests-utf8.txt`, `full-tests-utf8.xml`,
`initial-tests.txt`, `engine-retry.txt`, `local-packages.txt`.

## 8. Official validator results

`python -X utf8 validate.py`: **exit 1**. The interface and requirements checks
pass. Initialization fails at `submission/weights/best_model.pt`, which does not
exist because engine training never started. The initial tracked-size check
passed but preceded staging all project additions; the final size evidence is
recorded separately. Final tracked size is also checked again after staging.
`validate.py --docker`: **exit 2**, Docker executable absent. Windows cannot
emulate the Linux CPU-affinity/RAM checks fully. The official image pins Torch
2.15.0, NumPy 2.5.5 and Gymnasium 1.4.0, whereas local versions differ; exact-image
compatibility is therefore unverified. No successful official validation is claimed.

## 9. Training progress

**0 real environment decisions, 0 completed training updates.** The smoke command
failed at the official wrapper's `g.init()` before reset/rollout. Exact exception:
`vizdoom.vizdoom.MessageQueueException: Failed to create message queues: Access is denied.`
No model weights were generated or fabricated. No learning curve exists.
`benchmarks/training-smoke.txt` contains the command's failure trace.

## 10. Best checkpoint

**None.** `submission/weights/best_model.pt` is the future required destination,
not an existing artifact. Git LFS is installed locally and the starter attributes
are retained; `git lfs ls-files` is empty because there are no trained weights.
The submission audit intentionally fails this requirement. The agent has no
random-weight fallback masquerading as a trained model.

## 11. Held-out and capability results

No held-out episodes could run. Survival, kills, damage, keys, doors, locks,
exploration, loop recovery, backtracking and exit success are **UNMEASURED**.
A map file existing or a unit test passing does not demonstrate these skills.
No completed demo or baseline-vs-GRU gameplay comparison exists. Hidden map
scores and ranking are unknown.

## 12. Latency measurements

Trained-agent initialization/reset/mean/peak inference latency: **UNMEASURED**.
The separate untrained CPU microbenchmark includes preprocessing, forward pass,
sampling and action conversion on a synthetic official observation, one thread,
512 measured decisions per model after 16 warm-ups:

| Model | Mean | P95 | Peak | Parameters |
|---|---:|---:|---:|---:|
| Untrained MLP | 3.337 ms | 5.786 ms | 18.115 ms | 84,232 |
| Untrained GRU | 5.575 ms | 10.210 ms | 43.385 ms | 183,304 |

This does NOT certify DoomAgent timing, the process protocol, real entities,
Linux sandbox behavior, or RTX3090 performance. No checkpoint was saved by this
microbenchmark. Raw evidence: `benchmarks/model-microbenchmark.json`.

## 13. Memory and hardware

Total RAM: 11.78 GiB; available at detection:
1.41 GiB. Detected GPU:
NVIDIA GeForce RTX 2050, 4096 MiB, 621.40. Python 3.13.9, ViZDoom 1.3.3, Torch 2.9.1+cpu, NumPy 2.3.5,
Gymnasium 1.2.3; 8 physical / 12 logical CPUs.
The sensor processor string is recorded rather than claiming a CIM-verified
commercial CPU model; CIM access was unavailable.

The microbenchmark process peak working set was
208.34 MiB.
This is a model-only process result, not peak submitted-agent RAM. Trained-agent
RAM and VRAM remain UNMEASURED. CUDA runtime is absent from this Torch build;
there is no CUDA inference or training measurement. The official 8 GiB/2 GiB
resource constraints have not been certified.

## 14. Repository size and Git LFS

See `benchmarks/repository-size.json` for final payload/Git bytes and
`benchmarks/official-validator-final.txt` for the final tracked-size check.
The archive excludes `.git`, Python caches, runtime configuration, virtualenv,
optimizer checkpoints and third-party source used during diagnosis. Its manifest
contains checksums for every payload file except the manifest itself. It includes
no model-weight artifact.

## 15. Remaining weaknesses

The engine is unavailable inside this sandbox, so training/evaluation cannot
establish any gameplay skill. Generated scenarios are untested in the engine.
PPO/resume/parallel workers are not verified end to end. The generated maps are
small and regular; broad-map generalization would need more diverse training.
Reward door events lack actor attribution; novelty may still encourage wandering.
GRU memory is not proof of successful spatial reasoning. The exact organizer
image, public repository, event eligibility, LFS weights and replay are outstanding.

## 16. Recommended next steps

1. Use normal unsandboxed PowerShell or a Linux host where unmodified ViZDoom can
   create its legitimate local IPC objects. No registry or global security
   settings were changed. A directory-specific permission grant did not suffice;
   Windows still denied creating `C:\ProgramData\boost_interprocess`.
2. Run map-load tests, then the short baseline training command. Correct any
   engine-facing defects before extending the curriculum. Save genuine weights.
3. Validate the first trained MLP checkpoint, then compare equal-step MLP and GRU
   runs with multiple seeds and held-out progression metrics. Keep the best.
4. Increase map variety and sustained training within hardware/time limits. Add
   optional attention, curiosity or hierarchy only after measured improvements.
5. Test inside the official image, commit LFS weights to an event-valid public
   team repository and verify the pushed artifact before submission.

## 17. Readiness and approval boundary

**NOT READY.** It cannot be submitted as a trained agent. Automatic approval
review rejected the bounded engine test outside the sandbox because this session
disables sandbox escalation. A separate approved write scope for the Boost queue
directory did not overcome the Windows denial. The user selected packaging with
the blocker. No escalation was bypassed, no project was published, and no
unmeasured result was presented as successful.
