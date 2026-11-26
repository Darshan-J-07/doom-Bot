# Continue the existing project in WSL 2 / Ubuntu

Current status: **NOT READY**. The existing project has been extended, not rebuilt.
This session cannot launch WSL: both `wsl --status` and `wsl --list --verbose`
return `Wsl/E_ACCESSDENIED`. The user reports version 2 with version 1 unsupported.
That supports WSL 2 availability but does not identify an installed Ubuntu distro.
Windows detects RTX 2050, 4096 MiB VRAM, driver 621.40. WSL CUDA is unverified.

## 1. Identify or install Ubuntu from a normal Windows terminal

Run these in your normal PowerShell window, outside Codex:

```powershell
wsl --status
wsl --list --verbose
```

If Ubuntu is listed with VERSION 2, use that exact distro name below. If no Ubuntu
distro exists, run the following installation commands in Administrator PowerShell;
restart Windows if requested, launch Ubuntu and create its Linux user/password:

```powershell
wsl --install -d Ubuntu-24.04
wsl --update
```

If an existing Ubuntu distro has VERSION 1, convert only that distro:

```powershell
wsl --set-version Ubuntu-24.04 2
```

Replace `Ubuntu-24.04` with the actual name when it differs. No deregistration,
distribution deletion, registry edits or sandbox changes are needed. This follows
[Microsoft's WSL setup and version checks](https://learn.microsoft.com/en-us/windows/wsl/install).

## 2. Enter the existing checkout

From normal PowerShell (replace the distro name when necessary):

```powershell
wsl -d Ubuntu-24.04
```

Inside Ubuntu, use the existing project files on the mounted Windows drive:

```bash
cd '/mnt/c/Users/Darshan/Documents/Codex/2026-11-26/files-mentioned-by-the-user-yugaantar/outputs/doom-nexus'
uname -r
cat /etc/os-release
```

The kernel should identify WSL 2 and the distro should identify Ubuntu. This route
preserves all existing maps, source, Git history and shared result files. Engine
IPC uses Linux mechanisms, rather than Windows ProgramData Boost queues. Do not
reuse the Windows Python environment from Linux.

## 3. Install Linux dependencies and the isolated Python 3.13 environment

Inside Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git git-lfs libgl1 libglib2.0-0 libgomp1
export PATH="/usr/lib/wsl/lib:$PATH"
nvidia-smi
bash scripts/setup_linux.sh cuda
```

The script downloads pinned uv 0.9.18 from its official release, installs managed
Python **3.13.9**, creates a separate Linux virtualenv under `../../work/linux`,
and installs **ViZDoom 1.3.3**, CUDA **PyTorch 2.9.1/cu128**, and the existing
pinned training dependencies. It runs a real CUDA matrix-compute check, verifies
actual bytes and separation of all 50 maps, and records `linux-setup.json`. It
does not create model weights. [uv supports managed Python installation](https://docs.astral.sh/uv/guides/install-python/)
and [PyTorch provides these pinned CUDA/CPU wheels](https://pytorch.org/get-started/previous-versions/).

CUDA in WSL requires the Windows NVIDIA driver. No Linux NVIDIA display driver
or `cuda-drivers` package should be installed here. See the
[NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
Installing the full CUDA toolkit is unnecessary for these prebuilt wheels.
If the setup reports CUDA unavailable or the compute check fails, preserve its
JSON and use the explicit CPU route below; do not describe CUDA as tested.

```bash
bash scripts/setup_linux.sh cpu
```

CPU and CUDA use the same external environment path. Switching modes intentionally
replaces the Torch build; rerun setup for the intended mode before execution.
The managed Python and cache paths stay outside the submission repository.
If you choose a Linux-native data path for faster dependency I/O, export the same
`DOOM_NEXUS_WORK_DIR` value before **both** setup and run scripts.

## 4. Real engine smoke, genuine training, resume and held-out evaluation

Inside the same Ubuntu checkout:

```bash
bash scripts/run_linux.sh cuda
```

Use `cpu` if that is the build you installed. The script stops at the first failing
stage and preserves its log and exit code in a timestamped folder under
`benchmarks/continuation/`. It performs, in order:

1. A complete 2100-tick-cap official-map episode, then a complete 40-tick-cap
   episode on **every one of the 50** current curriculum maps. Random actions
   here diagnose the engine; they are not the submitted agent or PPO training.
2. One-worker environment throughput measurement. The default keeps one engine
   active due to the previously measured low available RAM. For a broader worker
   benchmark later, run `scripts.benchmark --throughput-only --max-envs 2`.
3. Genuine MLP PPO training and checkpoint resume, followed by genuine GRU PPO
   training. Each invocation has a bounded time budget and writes optimizer/model
   state only after completed real rollout updates. The starter run aims for
   8192 lifetime decisions per candidate; budget limits can produce unequal counts.
4. Selection on validation maps only, export to `submission/weights/best_model.pt`,
   then a separate test-map report using the checkpoint's exact decision mode.
5. The **entire** test suite with no deselection, performance measurement,
   Git LFS staging, source audit and the unmodified official validator.

This is an early trained baseline workflow, not a competitive training budget.
It never declares COMPETITION READY. Every log is an actual executed stage;
unreached stages remain unexecuted. The full test suite can expose additional
defects once the engine is available; preserve the traceback for correction.

## 5. Continue learning after the first valid trained baseline

Inside Ubuntu, set the interpreter and choose a new run directory:

```bash
PYTHON="$(cd ../.. && pwd)/work/linux/venv/bin/python"
export PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$PYTHON" -m training.train --model gru --stage auto --device cuda --envs 1 \
  --steps 1000000 --seconds 3600 --rollout 128 --sequence 32 --epochs 3 --out runs/competitive-gru
```

Use `--device cpu` if needed. Resume with a lifetime target larger than the saved
step count, and keep the same model architecture and entity-mode setting:

```bash
"$PYTHON" -m training.train --model gru --stage auto --device cuda --envs 1 \
  --steps 2000000 --seconds 3600 --resume runs/competitive-gru/latest.pt --out runs/competitive-gru
```

For deliberate continued progression practice after the staged introduction,
use `--stage mixed` with a larger lifetime target and the same checkpoint. Keep
held-out test maps out of training. Choose improvements using validation metrics,
not reward alone, and report differing training budgets when comparing candidates.

## 6. Evidence needed for readiness

Keep the timestamped folder, real `learning.jsonl` logs, checkpoint counters and
hashes, validation and held-out JSONs, full JUnit results, CPU/CUDA resource
measurements and official validator output. Share those outputs back to this chat
so defects and learning quality can be inspected. Files produced on the mounted
drive are visible in this same existing checkout.

Local validator success alone does not establish competitive combat, navigation,
keycard reasoning or completion rates. Official Linux base-image execution,
offline/read-only sandbox checks, the RTX3090 resource budget and event-valid
public-repository/LFS delivery still require verification. Do not publish the
project or claim hidden leaderboard scores without the corresponding evidence.
