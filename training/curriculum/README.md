# DOOM training curriculum

This pack plugs into the existing spatial PPO trainer. Start with short levels where the agent can learn one skill, then practice applying those skills together. The same policy reads the official observation and returns the official nine-button action throughout training.

| Stage | What the agent practices |
| --- | --- |
| `movement` | Moving and turning in short layouts and finding the exit. |
| `combat` | Learning to engage an enemy, take less damage, and survive long enough to exit. |
| `exploration` | Exploring branches and using ordinary doors. |
| `keys` | Picking up a key, returning from a detour, operating a locked door, and reaching the exit. |
| `gauntlet` | Combining combat with an ordered two-lock route. |

The pack has **60 training maps, 10 validation maps, and 10 test maps** across 25 authored layouts. Use train maps for PPO rollouts, validation maps for checkpoint selection, and test maps for the final comparison. These are public local holdouts. The organizers supply their hidden maps during judging. The trainer verifies the map files against the manifest before collecting rollouts. [Map manifest](manifest.json) · [Layout definitions](generate.py)

To regenerate the pack into a fresh local directory, run `python -m training.curriculum.generate --out runs/curriculum-repro` from the repo root.

## What informed the pack

In *Training an Agent for FPS Doom Game using Visual Reinforcement Learning and VizDoom*, Adil et al. trained a model to strafe and fire at a single stationary monster in a rectangular room. They used a pixel-based DQN and reported a mean reward of about 74 across 100 final episodes of that basic task. The setup is on printed p. 37, with results on pp. 38–40. Give the agent repeated practice on one measurable skill before combining skills. The paper’s score applies to its own scenario and reward scale. This pack keeps the repository’s spatial observations, PPO rollouts, and official four-tick decision interval. [Adil et al. (2017)](https://thesai.org/Downloads/Volume8No12/Paper_5-Training_an_Agent_for_FPS_Doom_Game.pdf) · [Agent contract](../../docs/CONTRACT.md)

The current Phase 2 release is the MLP at 200,096 decisions and 878 PPO updates. Its recorded 100-episode local comparison included 23 of 25 combat exits with zero combat kills and 20 of 25 exploration exits. On the keys tier it recorded 20 pickups and zero unlocks. It completed zero keys or gauntlet episodes. Run the release on this pack to establish a baseline for these layouts. [Phase 2 results](../../docs/PHASE2_RESULTS.md)

Kempka et al. tested isolated combat and a separate maze survival task. Their combat policies generally scored lower when evaluated with a different action repeat from training. Use the official four-tick interval in both. The maze used training-only reward shaping for pickups, and replay revealed repeated left-right turns that the average score obscured. Watch a replay when keys or survival improve without more exits. [ViZDoom platform paper, pp. 5–7](https://arxiv.org/abs/1605.02097v2)

The PyConES walkthrough uses headless training, a separate evaluation environment, and periodic checkpoint selection. The existing trainer's `--checkpoint-every` option and the fixed validation split give us that workflow. The walkthrough also shows an agent struggling after running low on ammo and another favoring one side. Inspect ammo, weapon selection, and left/right outcomes on matched maps if those behaviors show up in our replays. [PyConES PPO walkthrough](https://apsl.tech/en/blog/learn-reinforcement-learning-by-playing-doom/)

## Run a first update

Open a terminal at the repository root and use the project's Python 3.13 environment with PyTorch, ViZDoom 1.3.3, and the editable `env` package installed. The first command checks that each map runs in ViZDoom and uses a diagnostic driver to verify representative routes and doors. The second starts a new model on CPU and requests one short PPO update.

```text
python -m training.curriculum.verify --out runs/curriculum-verification.json
python -m training.train --map-root training/curriculum --model mlp --device cpu --stage movement --envs 1 --steps 64 --seconds 120 --ticks 420 --rollout 64 --sequence 16 --epochs 1 --seed 7 --out runs/curriculum-smoke-s7
```

Look in `runs/curriculum-smoke-s7` for `latest.pt`, `learning.jsonl`, and `summary.json`. Check that the summary records 64 decisions and at least one update, and that the learning log contains a finite loss. This confirms that engine interactions reached the optimizer. Use a fresh output name on each restart so the logs describe one experiment.

Use a writable checkout for these commands. ViZDoom can write `_vizdoom.ini` in the working directory, and the trainer writes checkpoints under `runs/`. If startup reports `Failed to create message queues`, check engine process and IPC permissions in your terminal before running a long job. [Ubuntu setup and diagnostics](../../docs/VERIFIED_ENVIRONMENT_RUN.md)

## Train and compare

Run a bounded trial from scratch. `--stage auto` starts with movement and adds later stages while keeping earlier ones in the sampling pool. The example uses the fraction of `--steps` completed to advance the curriculum. The time cap is checked between updates, so use `summary.json` and the logged stages to see how far the run got.

```text
python -m training.train --map-root training/curriculum --model mlp --device cpu --stage auto --curriculum-clock steps --envs 1 --steps 32768 --seconds 900 --ticks 1400 --rollout 64 --sequence 16 --epochs 2 --seed 7 --checkpoint-every 8192 --out runs/curriculum-pack-s7
```

For a matched training control, use the original maps and the same training settings.

```text
python -m training.train --map-root training/maps --model mlp --device cpu --stage auto --curriculum-clock steps --envs 1 --steps 32768 --seconds 900 --ticks 1400 --rollout 64 --sequence 16 --epochs 2 --seed 7 --checkpoint-every 8192 --out runs/curriculum-control-s7
```

Check each run’s actual decisions and updates before comparing. If time stopped one arm early, compare snapshots with matching decision counts or report the budget difference. Repeat with a second training seed when time allows.

Compare both trained candidates and the released checkpoint on the same curriculum validation maps. Use `git lfs pull` from the repo root if the released weight is still a pointer file. Keep `submission/weights/best_model.pt` as the frozen release and write results to a new file under `runs/`.

```text
python -m scripts.compare_phase2 --map-root training/curriculum --split validation --checkpoint submission/weights/best_model.pt --checkpoint runs/curriculum-control-s7/latest.pt --checkpoint runs/curriculum-pack-s7/latest.pt --seeds 0 1 2 --policy-seed 0 --ticks 2100 --out runs/curriculum-pack-s7/validation.json
```

Look at `tiers` for each checkpoint. Record exit counts and rates, survival, damage, and kills. For keys and gauntlet, check whether key pickups lead to unlocks and exits. The report includes the per-episode outcomes and a project-defined weighted proxy. A promising checkpoint should improve the targeted behavior and preserve the release’s stronger tiers. Report trade-offs by tier when the results are mixed.

Give `scripts.compare_phase2` more `--checkpoint` arguments to compare periodic snapshots such as `runs/curriculum-pack-s7/step-8192.pt` on the same validation episodes. Use the snapshot files the trainer actually wrote. If an earlier one plays better, freeze that snapshot for the final test.

If the model survives by running past opponents, try a short matched `--stage combat` experiment from scratch. Use the new pack in one arm and `training/maps` in the other. Compare kills, damage, and exit counts before adding harder maps. If key pickups improve and unlocks stay flat, run the same kind of comparison with `--stage keys` and review episodes where the model has the key near a closed lock. A repeated `USE` hold counts as one press. The agent needs to release and press again to make another attempt. [Door controls](../../docs/CONTRACT.md)

Choose a checkpoint on validation first. If `runs/curriculum-pack-s7/latest.pt` is the selected candidate, the following command compares it with the frozen release on the held-out test layouts. Keep this comparison for reporting once selection is finished.

```text
python -m scripts.compare_phase2 --map-root training/curriculum --split test --checkpoint submission/weights/best_model.pt --checkpoint runs/curriculum-pack-s7/latest.pt --seeds 0 1 2 --policy-seed 0 --ticks 2100 --out runs/curriculum-pack-s7/heldout.json
```

## Continue a checkpoint

Resume only from a full training checkpoint such as the trial’s `latest.pt`. It carries the model, optimizer, step counters, and random number generator state. Keep the model settings the same and give `--steps` a new lifetime target above the saved count. The released LFS artifact is for inference and comparison.

```text
python -m training.train --map-root training/curriculum --model mlp --device cpu --stage auto --curriculum-clock steps --envs 1 --steps 65536 --seconds 900 --ticks 1400 --rollout 64 --sequence 16 --epochs 2 --seed 7 --checkpoint-every 8192 --resume runs/curriculum-pack-s7/latest.pt --out runs/curriculum-pack-s7-continued
```

[Copyable training prompt](TRAINING_PROMPT.md)
