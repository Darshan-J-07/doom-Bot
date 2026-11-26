# Training prompt

If training is running already, start with the [active-run prompt](LIVE_TRAINING_PROMPT.md).

Copy the prompt below into a coding assistant with access to this repository. The paths assume its working directory is the repository root.

```text
Use the existing doomDevForge repository to run a small, reproducible Track 1 PPO experiment. Read training/curriculum/README.md, training/curriculum/manifest.json, training/train.py, scripts/compare_phase2.py, docs/CONTRACT.md, and docs/PHASE2_RESULTS.md first.

The goal is to find out whether this staged curriculum improves combat survival and key-to-exit progression. Preserve submission/weights/best_model.pt as the frozen Phase 2 release. Keep the official observation, decision interval, and DoomAgent interface. Save new checkpoints, logs, and evaluation files under fresh runs/curriculum-* paths.

The inspiration is Adil et al. (2017), Training an Agent for FPS Doom Game using Visual Reinforcement Learning and VizDoom, DOI 10.14569/IJACSA.2017.081205. Printed p. 37 describes the single-room strafe-and-shoot drill, and pp. 38–40 give same-task test results. Use that isolated drill and repeated measurement as a model for this curriculum. Keep this repository’s spatial PPO, reward, and action interface for the experiment.

Use Kempka et al., arXiv:1605.02097v2, for isolated scenario design, matching training and evaluation timing, and replay-based failure analysis. Apply the PyConES walkthrough’s periodic validation by comparing actual step-*.pt files on the same frozen validation episodes. Keep the official four-tick interval. If you see circling, one-sided behavior, or ammo depletion, record the behavior and change one training variable at a time.

1. Use the working Python 3.13, ViZDoom 1.3.3, and PyTorch environment. Run `python -m training.curriculum.verify --out runs/curriculum-verification.json` and check the manifest splits and engine smoke output. Fix in-scope setup failures as needed. Run the README’s 64-decision movement command. Check that the run has actual engine decisions, an optimizer update, a finite loss, summary.json, and latest.pt before starting full trials.

2. Train from scratch on training/curriculum using the README’s CPU command. Start with the feed-forward spatial MLP, one engine, stage auto, a seed of 7, 32,768 decisions and a 900-second cap. Run the same training on training/maps as the control. Change only the map root and output path. Keep separate checkpoints and report actual decisions, completed updates, elapsed time, decisions per second, and stop reason. If the time limit gives the arms unequal decision counts, use matching saved snapshots when available and state the actual budgets. If continuing an experiment, use the matching full training checkpoint and a larger lifetime step target. Use the release weight for comparison.

3. Use `python -m scripts.compare_phase2 --map-root training/curriculum --split validation` to evaluate both candidates and the frozen release. Supply each path with `--checkpoint`, use `--seeds 0 1 2 --policy-seed 0 --ticks 2100`, and give the JSON a fresh path under runs/. Report exit, survival, kills, damage, key pickups, and unlocks by stage from the same episodes. Explain whether behavior improved from key pickup to door unlock to exit. Treat the tier-weighted proxy as a local score. If there is time, repeat the matched training pair with another training seed before drawing conclusions.

4. Choose the candidate using validation. Favor improvement on the targeted failure modes alongside retained exit performance in easier stages. Freeze the choice and then run the README’s test comparison with the selected checkpoint path. Keep development on the train and validation splits. Save the test report separately and present any remaining failures and the sample sizes.

5. Report verified results with paths to the logs, checkpoint, validation JSON, and held-out JSON. State which experiments completed and what their measured results support. For any engine, dependency, or compute blocker, include the failing command and error. Keep the frozen release intact and leave promotion and publication for a separate decision.
```
