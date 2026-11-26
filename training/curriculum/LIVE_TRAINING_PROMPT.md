# Prompt for an active DOOM training run

Paste the block below into the coding assistant on the laptop running training.

```text
I'm already training the Track 1 DOOM agent on this laptop. Help me measure the current learning progress and decide whether to use the staged curriculum. The repository is https://github.com/darshan1117/doomDevForge and the curriculum is on branch track1-skill-curriculum.

1. Let the active run continue. Find its command or config, Python interpreter, map root, output directory, and most recent completed checkpoint. Inspect learning.jsonl and summary.json if present. Report the device, decision count, completed updates, decisions per second, and recent exits, kills, key pickups, and unlocks that are actually logged. If there is no usable checkpoint yet, wait for the next saved update before evaluating.

2. Work from a separate writable checkout of the curriculum branch and use the proven Python 3.13, ViZDoom, and PyTorch environment. Read training/curriculum/README.md and docs/CONTRACT.md. Prefer the latest completed step-*.pt snapshot. If only latest.pt exists, copy the completed file into a new runs/ directory for evaluation. Use absolute paths for snapshots from the active run. Allow the main job its compute budget and run the evaluation on CPU when there is capacity.

3. Evaluate the active snapshot and the released best_model.pt side by side on the curriculum validation maps. Use git lfs pull in the evaluation checkout if the release file is still a pointer. Replace the snapshot placeholder and use a fresh output path:

   python -m scripts.compare_phase2 --map-root training/curriculum --split validation --checkpoint submission/weights/best_model.pt --checkpoint "<absolute-path-to-completed-snapshot.pt>" --seeds 0 --policy-seed 0 --ticks 2100 --out runs/live-check-validation.json

   Report exits as counts and rates per stage, along with survival, kills, damage, explored cells, keys and locked doors opened. On keycard maps, count how many episodes picked up a key, how many of those unlocked a door, and how many exited. One seed is an initial check. Repeat promising comparisons with engine seeds 0, 1, and 2 on the same maps. Keep the official four game ticks per decision.

4. Use the result to choose one follow-up experiment. If the agent avoids enemies and takes heavy damage, use a short combat drill. If it picks up keys and rarely unlocks doors, focus on the keys stage, including return trips to a previously seen lock. These isolated drills follow the ViZDoom papers. Use the PPO walkthrough's periodic evaluation idea by comparing existing numbered snapshots on the same validation panel.

5. After the current run finishes, compare the new curriculum against the original maps in separate output directories. For a fair comparison use the same architecture, seed, reward, PPO settings, and added decision budget in both runs. For a continuation, start both from the same full training checkpoint with optimizer and RNG state. Set each --steps target to the checkpoint's saved decision count plus the added budget, and use an explicit stage for the skill you are testing. From a fresh start, use the README's --stage auto recipe. Save periodic snapshots with --checkpoint-every and keep the Phase 2 release artifact as the benchmark.

6. Select a candidate using validation, then run the held-out test split once the choice is frozen. Report the exact checkpoint paths, actual decisions and updates, split, seeds, validation and held-out report paths, and which behavior changed. Treat the tier-weighted proxy as a local metric. If startup is write-protected, rerun from the writable checkout and identify the failing path or exact engine error. Keep already completed work and report any evaluation or training still pending.
```

The training guide has the commands for a first PPO update, a from-scratch comparison, and test evaluation. [Curriculum guide](README.md)
