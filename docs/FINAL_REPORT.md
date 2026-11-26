# Verified environment continuation - current status

**NOT READY.** The user has verified successful actual engine initialization,
game actions and CUDA tensor computation in WSL 2 / Ubuntu 24.04, using Python
3.13.18, ViZDoom 1.3.3, Torch 2.11.0+cu128 and RTX 2050 (4 GB). The working
virtualenv is `~/doom-test/.venv`. These environment checks are accepted as user
evidence; they are not project PPO, model-weight or held-out results.

## Remaining execution boundary

A fresh `wsl.exe --list --verbose` from this session still returns
`Wsl/E_ACCESSDENIED` with exit -1. Codex therefore did not copy files into Linux
home or execute project engine tests, training, evaluation or validation there.
No sandbox bypass or escalation was attempted. The real engine is now known to
work outside this session; the remaining blocker is access from Codex.

## Exact manual continuation

Run this single command in the already-working normal Ubuntu terminal:

```bash
bash '/mnt/c/Users/Darshan/Documents/Codex/2026-11-26/files-mentioned-by-the-user-yugaantar/outputs/doom-nexus/scripts/continue_verified.sh'
```

The launcher preserves the Windows checkout, copies it into a new Linux-home
directory including Git history, reuses the verified Python environment, keeps
Torch unchanged and installs only project support dependencies as needed. It
prints and saves the actual destination path. It starts work in the foreground;
both successful and failed runs export a timestamped archive of actual results
back into this chat's Windows `outputs/` directory. No successful Linux result
archive exists yet from this session. See `VERIFIED_ENVIRONMENT_RUN.md` for budget,
resume and evidence details. Old fresh-install instructions are historical.

## Project work completed here

- Reviewed the existing setup, run, smoke, PPO, checkpoint and test interfaces.
- Added a real PPO CPU-versus-CUDA benchmark with complete fixed-step jobs,
  repeated alternating device order, recorded failures and median-rate selection.
  No benchmark result has been fabricated or pre-selected.
- Added a staged runner: project engine smoke on the official map and all 50
  existing WADs; the entire test suite before training; an early real MLP
  baseline plus full tests/official validation; longer GRU curriculum; real
  optimizer/RNG resume; validation-only checkpoint selection; held-out report;
  full final tests; CPU/CUDA inference measurements; Git LFS and official checks.
- Pre-training failures can continue only when they are the explicitly known
  missing `best_model.pt` prerequisite. Their failed status remains in the log;
  every test is rerun after real training and must then succeed. Genuine
  implementation errors stop the workflow. No organizer check is modified.
- Fixed unnecessary CUDA RNG initialization when saving a CPU checkpoint on a
  CUDA-capable host. This avoids distorting CPU training timing and memory.
- Added a wall-clock curriculum option so short training budgets can introduce
  late progression stages, a configurable post-update idle interval, and a
  configurable CUDA allocator fraction. The verified workflow uses one engine,
  one CPU thread, 0.45 allocator fraction and 200 ms update pauses. No measured
  thermal/throughput benefit is claimed before Linux execution.
- Preserved all original map bytes, submission modules, organizer wrapper,
  upstream tests and validator. Existing trained parameters are not replaced;
  this checkout still contains no genuinely trained checkpoint.

## Actual new test results

The latest combined unaffected/control run recorded **51 passed, 5 explicitly
deselected**. It includes all nine new workflow control cases. The CPU RNG fix
was also rerun with a simulated CUDA-available flag and no actual CUDA calls;
all nine control cases passed. These are unit results, not GPU execution or
DOOM training. The five engine/checkpoint/CUDA acceptance cases remain blocked
from this session. Their required Linux reruns are included without deselection
in the launcher. The previous full-suite result remains historical, not a
successful fresh full-suite run.

Evidence: `benchmarks/continuation/verified-environment-control-tests.xml`,
`verified-workflow-unit-tests.xml` and `verified-environment-status.json`.

## Submission readiness

No genuine project training, trained weights, CPU/CUDA PPO timing comparison,
held-out performance, official validator pass or organizer sandbox compliance
has been measured by this session. The launcher will save these only if its
actual stages execute successfully. Even a successful local baseline workflow
requires review of held-out combat, navigation, exploration, doors and keys,
plus organizer-image/resource certification and event-valid public delivery
before a COMPETITION READY declaration. The updated brief reports this boundary.

The reports below preserve previous execution history.

---

# DOOM NEXUS continuation report

**NOT READY.** The existing implementation was preserved. WSL exists, but this
session receives `Wsl/E_ACCESSDENIED` for both status and distribution enumeration.
The user reports WSL version 2 / version 1 unsupported; installed Ubuntu and Linux
CUDA remain unverified. No sandbox bypass or escalation was attempted in this
continuation. No genuine training, new weights or held-out gameplay ran.

## WSL 2 and CUDA checks

Windows still detects RTX 2050, 4096 MiB, driver 621.40. The Windows environment
has Python 3.13.9, ViZDoom 1.3.3 and CPU-only Torch 2.9.1. This is not WSL CUDA
evidence. `benchmarks/continuation/host-probe.json` distinguishes observed command
failures from the user's version report. Follow `LINUX_CONTINUATION.md` in a
normal terminal to identify/install Ubuntu and verify real CUDA tensor execution.

## Individual test investigation

Every one of the previous 29 failed cases, 12 setup errors and the CUDA skip was
reviewed using its JUnit traceback and source function. The complete per-case
record, source lines, causes, remediation and exact selector commands are in
`INDIVIDUAL_TEST_AUDIT.md` and `benchmarks/continuation/individual-test-audit.json`.
Linux reruns are explicitly NOT EXECUTED. The original tests and validator are
unchanged; eight Unix-only failures should be rerun unchanged on Linux, not
removed or treated as successful Windows tests.

## Genuine implementation fixes and new test results

1. Training and evaluation now recalculate each WAD's SHA-256, inspect its actual
   map metadata and reject cross-split byte overlap before using the manifest.
   All existing 50 maps are preserved byte for byte, with 30/10/10 splits. Tests
   demonstrate rejection of tampered bytes and duplicate train/validation WADs.
2. Evaluation inherits the saved deterministic/stochastic mode by default and
   promotion preserves the mode actually evaluated. Explicit `--deterministic`
   and `--no-deterministic` overrides remain available. Four parity cases pass.
3. Both installers force Torch replacement when switching CPU/CUDA builds,
   avoiding an installed `2.9.1+cpu` satisfying a CUDA request for `2.9.1`.
   Installer execution in Linux is still unverified.
4. Resuming to a lifetime step target already reached now fails clearly, and
   reset failures close the newly created training environment. These engine-
   facing paths remain unverified without the real engine.
5. The worker benchmark can cap itself at one environment for this RAM-limited
   host. This does not claim that worker throughput was measured.

Seven new regression cases passed. The combined unaffected unit rerun recorded
**42 passed, 5 explicitly deselected**, in `benchmarks/continuation/unaffected-tests.txt`
and `.xml`. The five deselected cases are engine-load, three trained-agent
lifecycle/episode cases, and CUDA-agent execution; their previous failures/skip
remain outstanding, not passed. No fresh full-suite success is claimed. The
previous full-suite record remains 47 passed, 29 failed, 12 errors and one skip.
Static preflight verified Python/ViZDoom versions and the actual hashes/separation
of all 50 WADs; its `engine_requested=false` and empty episode list make clear
that it is not an engine smoke result.

## Manual Linux execution and exact remaining blocker

Use the existing checkout, not a newly generated project. `LINUX_CONTINUATION.md`
contains copyable PowerShell and Ubuntu commands. `scripts/setup_linux.sh`
creates an external Linux Python 3.13.9 environment, pins ViZDoom 1.3.3 and Torch
2.9.1 CUDA 12.8 or CPU, and records CUDA computation checks. `scripts/run_linux.sh`
then performs an actual complete official-map episode and 50 complete short
curriculum episodes, real PPO + resume, validation-only checkpoint selection,
held-out evaluation, all tests, performance measurements, Git LFS staging and
`python validate.py`. Each stage logs a real exit code and stops on failure.
These scripts were reviewed but could not be executed under the WSL restriction.

Run in normal PowerShell: `wsl --list --verbose`; use its Ubuntu distribution
name to enter Linux. If absent, install Ubuntu using Microsoft's supported WSL
procedure. Then follow the guide's `cd`, system dependency installation, and:

```bash
bash scripts/setup_linux.sh cuda
bash scripts/run_linux.sh cuda
```

The CPU fallback is the same two commands with `cpu`. No Linux GPU display driver
is installed by these scripts. Their first training budgets establish a baseline;
they are not proof of competitive skill or equal-budget statistical superiority.

Training, model weights, held-out combat/navigation/keycard results, official
resource compliance and official validator success remain BLOCKED until these
commands run in the user's normal Linux terminal. Preserve the timestamped
evidence folder and share it back for failure analysis and further optimization.

## Submission status

**NOT READY.** No COMPETITION READY claim is justified. The two-page brief now
includes this continuation's actual unit results and the WSL blocker. The
historical report below remains a record of the earlier run, not fresh results.

---

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
The Git shell hook also reports a Windows named-object access denial, so the LFS
upload/checkout workflow must be tested outside this sandbox before adding weights.
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
