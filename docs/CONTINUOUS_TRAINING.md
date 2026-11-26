# Three-hour active learning campaign

The Ubuntu user service runs one PPO learner at a time and a separate development evaluator on spare CPU cores. The target is 10,800 seconds credited only after a rollout and optimizer update are saved in an atomic checkpoint. Initial setup, reporting, cooldown, viewer pacing, frozen-model evaluation and validation have separate accounting. Supervised fitting adds its measured optimization duration when its checkpoint is saved. The campaign's four-hour UTC target is for planning. Individual workers have fresh monotonic timeouts.

The recorded previous run ended after the paired promotion gates failed. The message about an expired UTC guard was printed by a separate test with an artificially expired clock. The current campaign has a fresh clock record, step counters and active-time ledger. Restarting the user service restores saved meters from the same campaign. Grouped development gates govern candidate selection. A failed comparison advances the experiment plan, and the published submission remains at its verified hash. [Campaign state](../runs/continuity-campaign-20261127/status.json) · [Deadline audit](../runs/continuity-campaign-20261127/provenance/deadline-audit.json)

The workflow expects the local archived training checkpoints and verified WAD manifests already available in the research checkout. Weight files and machine-specific run outputs remain local. The current branch can replay this work with those inputs and the installed Python 3.13 environment.

## Monitor the learner

```bash
cd /home/darshan/projects/doom-nexus
py=~/doom-test/.venv/bin/python
watch -n 3 "$py -m scripts.continuity_campaign status"
```

Use a separate terminal for the event log or process status.

```bash
tail -f runs/continuity-campaign-20261127/progress.log
systemctl --user status doom-nexus-continuity.service
```

The dashboard reports completed PPO decisions, optimizer updates, checkpoint-backed active seconds, remaining active seconds and the last paired evaluation. Evaluation seconds can overlap learning in real time.

## Watch the latest checkpoint

```bash
SDL_VIDEODRIVER=x11 ~/doom-test/.venv/bin/python -m scripts.view_adaptive_campaign \
  --run runs/continuity-campaign-20261127 --panel post-lock --speed 1
```

The native viewer reloads the newest completed checkpoint between episodes. It labels these episodes frozen-checkpoint evaluation. The viewer itself can be closed with Ctrl-C. Its gameplay is separate from the headless learner and adds zero training time. Use `--panel familiar` to watch one of the four benchmark tiers. `--speed 0.5`, `--speed 1` and `--speed 4` control the playback rate.

## Stop and resume

Run this command to request a safe pause. Wait until the status command reports `paused` before starting the service again.

```bash
~/doom-test/.venv/bin/python -m scripts.continuity_campaign stop
~/doom-test/.venv/bin/python -m scripts.continuity_campaign status
```

Resume headless from the campaign's last recoverable checkpoint.

```bash
~/doom-test/.venv/bin/python -m scripts.continuity_campaign resume
```

To see an actual PPO rollout, pause the headless service first and resume with the display flag. One environment renders and the other environments continue headless. This view runs uncapped to preserve training throughput.

```bash
DISPLAY=:0 SDL_VIDEODRIVER=x11 ~/doom-test/.venv/bin/python -m scripts.continuity_campaign resume --visualize
```

The service can continue independently of the launching terminal while Ubuntu and its user service manager remain running. Resume after a deliberate OS shutdown or service stop with the commands above.

## Evaluation receipts and final reports

Paired development episodes use the exact numbered exports, the pinned switch and familiar cohorts, and a separate post-lock diagnostic. The late gate requires two independently seeded passing policies before any reserved final WAD is opened. The selected candidate goes through exact-artifact parity and the unchanged validator. The service writes HTML, JSON and CSV to `reports/continuity-campaign-20261127` after the active budget completes. The official validator and published model remain in their original locations.
