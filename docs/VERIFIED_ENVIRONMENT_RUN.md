# Continue using the verified Ubuntu environment

The user reports actual successful engine initialization/actions and CUDA tensor
computation in Ubuntu 24.04 / WSL 2, Python 3.13.18, ViZDoom 1.3.3 and Torch
2.11.0+cu128 on RTX 2050 (4 GB). These are user-verified environment results,
not project-specific engine, PPO or held-out results from this session.
The session's fresh `wsl --list --verbose` attempt still returns
`Wsl/E_ACCESSDENIED` (exit -1). Linux execution therefore needs the normal Ubuntu
terminal. Do not rerun the old fresh-install procedure or downgrade Torch.

## Exact copy-and-run command

In the normal Ubuntu terminal, paste:

```bash
bash '/mnt/c/Users/Darshan/Documents/Codex/2026-11-26/files-mentioned-by-the-user-yugaantar/outputs/doom-nexus/scripts/continue_verified.sh'
```

This command checks the existing `~/doom-test/.venv/bin/python`, creates a new
`~/doom-nexus-<UTC timestamp>-<process id>` directory and copies the existing
checkout, including Git history, into it. It refuses to overwrite an existing
destination. The Windows original is preserved. It uses your current Python,
ViZDoom and Torch; only missing/incompatible project support dependencies and
the editable official `doom-env` package are installed. Installed compatible
versions are retained, and the actual package list is recorded. Torch is not
included in the install command.

If pip is absent, the script uses the verified Python's bundled `ensurepip`.
The default pip cache is reused. The environment must have access to package
indexes only when a required package/build dependency is missing from that cache.
No GPU driver or CUDA toolkit is reinstalled.

The default main GRU run budget is 1800 seconds, plus separate bounded device
benchmarks, a five-minute baseline, a three-minute resume check and evaluation.
For a shorter first attempt, paste this instead:

```bash
DOOM_NEXUS_TRAIN_SECONDS=600 bash '/mnt/c/Users/Darshan/Documents/Codex/2026-11-26/files-mentioned-by-the-user-yugaantar/outputs/doom-nexus/scripts/continue_verified.sh'
```

The script prints `Working Linux copy: ...` and saves that actual path to the
Windows outputs directory as `DOOM_NEXUS_Linux_Working_Copy.txt`. It launches work
in the foreground and records output as it runs. It does not publish the repo.

## Actual stages and stop conditions

1. Verify project versions, CUDA computation, real WAD hashes and 30/10/10 split
   separation. Run a complete official-map episode, then a complete short episode
   on each of the 50 original curriculum maps. Engine smoke uses random actions
   only as a diagnostic; it produces no trained weights or gameplay claims.
2. Run the **entire test suite** before training. If genuine weights are absent,
   only known `FileNotFoundError: ... best_model.pt` prerequisites may be deferred.
   These failures remain recorded as failures. Any other failure or setup error
   stops the workflow for investigation. No assertion, test or official validator
   is weakened, skipped or deleted by the workflow.
3. Compare two real fixed-step GRU PPO runs per device, alternating CPU/CUDA order,
   with one environment, one CPU thread and identical rollout/sequence settings.
   A benchmark must finish its requested decisions and updates to count.
   Select the device with the larger median decision throughput among completed
   candidates. Failed/partial runs remain in the report and cannot be reported
   as completed comparisons. The timed training includes collection, PPO updates
   and checkpointing; process startup wall time is recorded separately.
4. Train a genuine spatial MLP baseline, select it on validation maps, rerun the
   full test suite, then run the unchanged `validate.py`. This preserves the
   first valid trained submission before the longer recurrent experiment.
5. Train the existing GRU with the five-stage curriculum and a wall-clock schedule,
   so a short budget still introduces combat, exploration, doors and keycards.
   Earlier stages remain eligible as new ones are introduced. A separate resume
   invocation restores actual model/optimizer/RNG state and trains on mixed maps.
6. Compare MLP, GRU and resumed-GRU candidates on validation maps only. Export the
   winner to `submission/weights/best_model.pt`, then report a separate test split.
   No test-split promotion is allowed. Training budgets and actual counters are
   recorded; unequal budgets are not claimed to be a controlled architecture study.
7. Rerun every test. Measure CPU and CUDA inference, stage genuine weights with
   Git LFS, audit submission imports/provenance and run official validation on
   both devices. The organizer base-image/offline/read-only sandbox remains a
   separate final certification; local success is not automatically competition readiness.

Each stage saves a text log and exit code under
`benchmarks/verified-environment/<UTC timestamp>/`. Real checkpoints and learning
logs are under `runs/<same timestamp>/`. The workflow stops at the first genuine
failure and writes `blocked.json`. An actual successful run writes `completed.json`.
Neither file is pre-created with fabricated success by this Codex session.

## Laptop resource choices

Training starts with one engine and one PyTorch CPU thread. The CUDA allocator
fraction is 0.45, leaving headroom on the 4 GB GPU; inference still uses the exact
official limits. A 200 ms pause follows main-run optimizer updates. This is a
configurable idle interval, not a measured thermal limit or a guarantee about
temperature. Device benchmark jobs use no artificial pauses.

`latest.pt` is atomically replaced after every completed PPO update. Resume
starts fresh engine episodes while restoring optimizer/model and appropriate
device RNG state. CPU checkpoint saving now avoids initializing CUDA merely
to save irrelevant CUDA RNG. The verified-environment benchmark therefore does
not add that avoidable CUDA allocation to CPU training.

## Resume commands

To return to the actual copied project, read its saved path and inspect its runs:

```bash
DOOM_DEST="$(cat '/mnt/c/Users/Darshan/Documents/Codex/2026-11-26/files-mentioned-by-the-user-yugaantar/outputs/DOOM_NEXUS_Linux_Working_Copy.txt')"
cd -- "$DOOM_DEST"
find runs -name latest.pt -print
```

Resume the chosen checkpoint with a lifetime target larger
than its saved counter, for example after replacing the path with the real file:

```bash
~/doom-test/.venv/bin/python -m training.train \
  --model gru --stage mixed --device cpu --envs 1 \
  --steps 2000000 --seconds 1800 --cooldown-ms 200 \
  --resume runs/ACTUAL_RUN/gru/latest.pt --out runs/ACTUAL_RUN/gru
```

Use the selected benchmark device (`cpu` or `cuda`) and the checkpoint's real
architecture settings. The example run path is a placeholder, not an existing
checkpoint claim. Do not regenerate maps or overwrite the Windows source.

## Bring the evidence back

The Linux-home working copy is intentionally separate from the preserved Windows
original. On success or failure, the launcher exports a timestamped
`doom-nexus-...-results.tar.gz` beside the original Windows checkout, under this
chat's `outputs/` directory. It includes actual benchmark logs, training state,
weights when present, and the real launcher exit code. A failing run is not
marked successful merely because its evidence was archived.

The archive includes `blocked.json` and the failing stage log if available. Those real
outputs are required before diagnosing any new engine-specific failures or
claiming trained weights, learning improvement, held-out skill or validator success.
