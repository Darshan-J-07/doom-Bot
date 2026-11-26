# Phase 2 experimental protocol

Use the original 30 train maps for PPO. The original 10 validation maps guide experiment and checkpoint selection. The historical test maps are diagnostic because Phase 1 already evaluated them and Phase 2 replayed their failures.

Reserve five new map seeds per Tier 1–4 category, starting at 410000 and advancing 100 per tier. Use engine seeds 41000–41004 and a fixed policy reset seed of 0, matching submission inference, for 100 new episodes per frozen checkpoint. Generate this split only after candidate selection. Verify its 20 WAD hashes against all 50 existing maps. The maps use the existing procedural generator and are unfamiliar instances of the same scenario families. The gauntlet still has one lock.

Use equal budgets, identical sampling rules, common evaluation seeds, and identical PPO settings for architecture controls. Policy-dependent episode lengths can change the realized count of maps. Set the recurrent state to zero before every decision in a validation-only intervention to estimate whether the saved policy uses memory. Sensitivity to intervention is not evidence of explicit mapping or robustness across training seeds.

Score each difficulty tier with the existing local proxy. Rank the mean tier scores with weights 1, 2, 3 and 4. Report survival, kills, damage, discoveries, doors, keys, unlocks and exit fraction separately. Movement is a warm-up control and contributes no points to the weighted proxy. The organizer formula and hidden maps are private.

## Stop and resume

Training writes an atomic `latest.pt` after every completed update. `--checkpoint-every 10000` preserves the first completed update at or beyond each interval. A stop file or low-memory threshold takes effect at update boundaries. Stop with:

```bash
touch runs/phase2/STOP
```

After the run exits, remove the stop file and use the same trainer settings, `--resume <run>/latest.pt`, a new output directory and a larger cumulative `--steps` target. Checkpoint state includes model weights, optimizer and policy/map randomness. Engine episodes restart on resume. The learning-rate schedule uses the new cumulative target.

Four serial CPU environments, rollout 64, sequence 16, eight sequences per minibatch and two PPO epochs measured 339 decisions/sec in the short sweep. At that rate, warm-starting from 8,192 to 100,000 decisions takes about 4.5 minutes before cooldown and evaluation. From 100,000, the next 150,000 decisions would take about 7.4 minutes at the same rate. Each subsequent 250,000 decisions would take about 12.3 minutes, and 500,000 would take about 24.6 minutes. Use completed-run throughput to update the estimates before launching a new budget.

## Freeze and release

Select a single challenger using the tier-weighted validation proxy. Materialize the final maps after recording the challenger hash. Run the preserved baseline and challenger once on those maps, with no tuning on the resulting episodes. Release the challenger only if it improves the final tier-weighted proxy, preserves at least the baseline's combined weighted Tier 3 and Tier 4 score, and passes the official local CPU and CUDA checks. Independently report the Docker check and keep the baseline as the release if the challenger misses the gameplay gate. Keep all trained snapshots regardless of release outcome.

## Recurrent control

The original MLP and GRU runs mixed architecture, curricula and decision counts. At 8,192 matched decisions and 32 updates each, the fresh legacy GRU scored 29.62 versus 13.08 for the fresh legacy MLP. On the same validation maps and seeds the original movement-trained MLP scored 24.87. The GRU reached 0.75 unlocks per Tier 3 episode. Zeroing its hidden state on each decision yielded 13.66 and zero Tier 3 unlocks. This single-seed control shows the recurrent state affects useful behavior. Replicating across training seeds remains necessary.

At rollout start the hidden state is zero. The policy masks hidden state and previous action at each episode boundary. PPO detaches the rollout hidden state at each sequence start, differentiates through at most 16 decisions in this experiment, and ignores padded tail steps. The ragged-batch regression compares output and value against independent unpadded execution across an episode restart. Terminations suppress bootstrap, timeouts bootstrap from the last available wrapper observation, and neither boundary carries GAE into another episode. The engine lacks a fresh observation at the timeout itself. Mid-episode hidden state can straddle a PPO update, and restoring a checkpoint starts fresh engine episodes.

Both models used the same base learning rate of 0.0003, linear decay with a 10% floor, normalized policy advantages, entropy coefficient 0.01 and the same sampled categorical action distribution in collection and evaluation. The validation controls vary the policy RNG with the episode seed, while final evaluation fixes it at 0 like the submitted agent. Median entropy was 5.367 MLP and 5.372 GRU. Median pre-clip gradient norm was 0.807 MLP and 1.010 GRU, with clipping at 0.5. The GRU had higher median value loss, 1.507 versus 0.057. Realized training episode lengths differed despite the common map sampling distribution.

The progress reward uses first-visit and first-interaction credit, bounded health-at-exit credit and a persistent-stall cost. The fresh MLP progress control scored 10.27 versus 13.08 for the legacy control. The 100,000-decision run uses legacy shaping. The experiment does not establish whether a single shaping coefficient would help.

## Curriculum coverage

The 50 verified WADs include narrow corridors, single-enemy rooms, branched exploration layouts, keycard detours and three-enemy one-lock gauntlets. The static WAD inventory is in `benchmarks/phase2/curriculum-audit.json`. The original movement maps offer straight corridors and no dedicated slalom. The original combat maps use the same 3×3 room grid with varying spawns. The gauntlet traces establish a training gap after key pickup and poor survival under sustained combat.

The opt-in expansion preserves the original 30 maps and adds six train-only gauntlets with two ordered locks, red and blue keys on separate detours, three enemies and fewer pickups. All six loaded and completed a short real-engine smoke episode. To prepare a fresh set, run `python -m scripts.prepare_expanded_curriculum --out runs/phase2/expanded-curriculum`. Training can consume it with `--map-root runs/phase2/expanded-curriculum`. Release comparisons need a separate controlled trial before adopting it. The 100,000- and 250,000-decision runs use the original split.
