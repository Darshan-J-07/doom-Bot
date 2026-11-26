# Adaptive campaign research and controlled follow-ups

## Starting evidence

The archived switch-only imitation effort evaluated two structured seeds and an RGB arm. Each pure behavioral-cloning checkpoint completed zero of 36 conditional switch episodes on its development panel. The training-only corrective variant mixed teacher and policy states on door-free switch drills. It completed six switches after 80 supervised updates and zero after 300. Those results describe a different panel from the campaign's current paired diagnostics. [Archived screen summary](../runs/limit-campaign-20261127/screen-summary.txt)

The earlier residual GRU initialized to identical MLP outputs. At the final matched milestone it completed six hard validation episodes, compared with nine for the MLP. It also used 244.2 seconds of training versus 213.3 seconds. Better memory remains a hypothesis for longer progression routes. [Archived memory experiment](SKILL_CAMPAIGN.md)

Khan et al. evaluated visual Q-learning in a rectangular chamber using left, right and shoot actions. Their result supports isolated practice objectives and intermediate testing. It gives no evidence about keycard dependencies, arbitrary-layout navigation or using a larger frame skip under this event's fixed four-tic contract. [Khan et al., 2017](https://thesai.org/Publications/ViewPaper?Code=IJACSA&Issue=12&SerialNo=5&Volume=8)

## Bounded imitation follow-up

The Track 1 problem statement requires direct gameplay learning and confines the final agent to trained neural decisions from the official observation. The demonstration policies interact with ViZDoom during collection. Their authored geometry and privileged state stay inside the collection harness. Matched PPO follow-ups use fresh on-policy trajectories. [Track 1 extraction](../runs/adaptive-campaign-20261127/provenance/official-tracks-extracted.txt)

The experiment tests whether recovery demonstrations can help the starting MLP while limiting change to its movement, turn and interaction logits. It keeps the feature extractor, critic, attack output and weapon selection output fixed during supervision. It samples behavior-cloning targets from successful teacher decisions and adds a separate frozen-policy KL term on states visited by the actual learner during collection. This is one round of learner roll-ins and teacher rescues. DAgger motivates collecting expert decisions on learner-visited states across iterations. [Ross, Gordon and Bagnell, 2011](https://proceedings.mlr.press/v15/ross11a.html)

The mentor dataset uses 64 approved recovery training maps and 24 newly authored training maps across closed doors, key detours and guarded key detours. There are 88 successful teacher-only episodes and 85 successful learner-prefix episodes. Five learner prefixes reached the exit without takeover. The 19,363 transitions include 17,301 teacher decisions and 2,062 actual learner decisions. The new maps include 18 engine-verified key-pickup, closed-lock opening, traversal and switch exits. The guarded routes demonstrate survival through the objective, with combat actions kept outside the imitation objective. Collection took 56.9 seconds. [Map identities](../runs/adaptive-campaign-20261127/imitation/data/training-manifest.json) · [Route receipts and failed attempts](../runs/adaptive-campaign-20261127/imitation/data/collection.json)

The supervised probes use 20 and 60 updates, batch size 128 per source, Adam at 0.00003 and seeds 59 and 83. Per-route weighting keeps long paths from overwhelming short alignment episodes. Each checkpoint records supervised work separately from the original 314,496 PPO decisions. The matched PPO comparison fixes the supervised endpoint at 60 updates. The no-BC and BC checkpoints both start from fresh Adam without old moments. The two arms use the same mixed rollout allocation, seed, learning rate and 12,000 added decisions. Recovery tests use the paired development panel and the separate familiar retention benchmark. [Pre-registration](../runs/adaptive-campaign-20261127/imitation/protocol.json)

The following follow-up commands assume a prepared campaign directory and a current campaign clock. Use a fresh campaign directory after the original deadline.

```bash
cd /home/darshan/projects/doom-nexus
py=~/doom-test/.venv/bin/python
study=runs/adaptive-campaign-20261127
"$py" -m scripts.adaptive_imitation collect --out "$study/imitation/data-reproduction" --seconds 400
"$py" -m scripts.adaptive_imitation fit --dataset "$study/imitation/data-reproduction/trajectories.pt" \
  --out "$study/imitation/seed-59-reproduction" --seed 59 --updates 60 --seconds 120
"$py" -m scripts.adaptive_followup ppo --out "$study" --source "$study/imitation/seed-59-reproduction/bc-060.pt" \
  --seed 59 --label bc-59-reproduction --decisions 12000 --seconds 240
```

For the matched no-BC command, use `$study/imitation/control-fresh-adam.pt` as the source and a fresh output label. Both arms restore the original model and the specified fresh optimizer before collection, with only the actor warm-start differences in the BC arm. Substitute seed 83 for the replication. Sequence training can reuse the same control because both memory initializers use fresh Adam.

The decisions below come from exported DoomAgent artifacts. The switch and walkover column groups use 32 independent layouts at three engine seeds each. The familiar suite uses 24 layouts at five engine seeds each.

| Before PPO fine-tuning | Switch exits | Entered range | Alignment | Walkover | Familiar | Gauntlet deaths |
|---|---:|---:|---:|---:|---:|---:|
| Frozen starting MLP | 18/54 | 21/54 | 12/18 | 9/24 | 30/120 | 20/30 |
| BC seed 59, 60 updates | 12/54 | 15/54 | 12/18 | 18/24 | 9/120 | 23/30 |
| BC seed 83, 60 updates | 12/54 | 15/54 | 12/18 | 18/24 | 9/120 | 22/30 |

Supervision took approximately 1.6 seconds per seed and left the PPO counters unchanged. Expert negative log likelihood fell from 1.352 to 1.238 in seed 59 and from 1.331 to 1.224 in seed 83. The paired gates failed in both seeds. Each policy completed nine more walkover episodes and 21 fewer familiar episodes than the frozen MLP. [Measured BC gate](../runs/adaptive-campaign-20261127/evaluation/gate-bc-only.json)

| After 12,000 fresh PPO decisions | Switch exits | Entered range | Familiar | T3 locks opened | T4 locks opened | Gauntlet deaths |
|---|---:|---:|---:|---:|---:|---:|
| Fresh-Adam MLP seed 59 | 9/54 | 21/54 | 13/120 | 0 | 5 | 24/30 |
| BC then PPO seed 59 | 18/54 | 21/54 | 20/120 | 20 | 15 | 22/30 |
| Fresh-Adam MLP seed 83 | 6/54 | 9/54 | 5/120 | 0 | 5 | 20/30 |
| BC then PPO seed 83 | 12/54 | 24/54 | 10/120 | 0 | 11 | 21/30 |

Behavioral cloning improved switch completion relative to the matched fresh-Adam continuation in both seeds. The paired differences were +16.7 and +11.1 percentage points. Family-resampled intervals were [0, +33.3] and [−16.7, +50.0] percentage points across three families. No arm finished a familiar keycard level or gauntlet. The seed-59 clone gained 20 keycard-tier lock openings relative to its control. This gain did not replicate at seed 83, although both seeds opened more gauntlet locks. Gauntlet ammo spending also rose in both seeds relative to the controls. Both imitation-plus-PPO policies missed the frozen baseline's 30 familiar exits and failed the advancement gates. [Paired BC vs PPO results](../runs/adaptive-campaign-20261127/evaluation/paired-followup-bc.json) · [Late gate](../runs/adaptive-campaign-20261127/evaluation/gate-bc-ppo.json)

On `early-lock-001` at engine seed zero, BC-PPO seed 59 took the key at tic 372 and opened the lock at 1,244. It recorded no door traversal and hit the 2,100-tic limit at full health. The final 856 tics recorded no progress. `MOVE_LEFT` appeared in 452 of 525 action vectors and `USE` appeared in 297. This episode isolates a failure to follow through after unlocking. [Trajectory](../runs/adaptive-campaign-20261127/evaluation/familiar/bc-ppo-59-12000/map-011-seed-0.jsonl.gz)

On the 138 recorded teacher observations immediately before a successful switch press, mean neural `USE` probability fell from 0.609 to 0.560 in both supervised seeds. Teacher press samples had 4.8% of the training sampling weight. The mean supervised loss had masked a loss on the rare terminal interaction. A separate balanced loss on usable and similar non-usable training poses could test that mechanism. These states come from demonstration training, and the paired gameplay metrics remain the transfer test. [Recorded training-state logits](../runs/adaptive-campaign-20261127/imitation/policy-on-teacher-states.json)

The USE follow-up changes only the positive label weight in the USE cross-entropy from one to four. It divides the weighted loss by the sum of chosen label weights. The two fits reuse the demonstration set, optimizer settings, seeds and deterministic sample draws. Positive USE labels account for 365/7,680 and 371/7,680 teacher draws. Their mean effective weight within the USE loss rises to 16.43% and 16.65%. The recorded sample-draw digests match independent replay. Mean `USE` probability on the same 138 teacher poses rises to 0.566 in both seeds. [Pre-registration](../runs/adaptive-campaign-20261127/imitation/use-weighting-protocol.json) · [Class weight receipts](../runs/adaptive-campaign-20261127/imitation/use-weighting-diagnostics.json)

The weighted variants kept the same per-episode exit outcomes as regular BC across both seeds and both panels. Their action sequences changed in 60/96 and 57/96 switch-panel episodes and 77/120 and 71/120 familiar episodes. Seed 83 recorded one additional gauntlet death. Both seeds failed the development gate. The controlled follow-up ended at the supervised checkpoint. To reproduce it, add `--use-loss-weight 4` to the imitation fit command with a fresh output directory. [Weighted BC contrast](../runs/adaptive-campaign-20261127/evaluation/paired-followup-bc-use4.json) · [Episode trace comparison](../runs/adaptive-campaign-20261127/imitation/use-weighting-gameplay-traces.json)

## PPO and observation memory

The custom PPO samples six categorical action groups and sums their log probabilities. The update uses the joint clipped policy ratio, a clipped critic objective, normalized GAE and gradient clipping. The finite-horizon setting zeros the bootstrap at the map cap. GAE cuts the recursive advantage tail on both terminated and truncated transitions. The non-finite-horizon setting can bootstrap on truncation and has a caveat because the wrapper returns its last observation at terminal steps. [PPO paper](https://arxiv.org/abs/1707.06347) · [GAE paper](https://arxiv.org/abs/1506.02438) · [Local implementation](../training/ppo.py)

The recurrent path stores the hidden vector before the action and replays contiguous slices. Episode-start masks reset memory and previous action. Padding follows the valid prefix and contributes no gradient. Optional burn-in recomputes context using the current model, with gradients applied only to the training sequence. The first PPO update checks pre-update replay log probability parity. Context at the first sampled slice can still depend on stored hidden states from the preceding rollout. [Sequence implementation](../training/ppo.py) · [Trainer](../training/train.py)

The existing occupancy model uses a differentiable 9×9 ego-centric evidence grid. Official ray distances and hit types provide free-space and obstacle evidence. Permitted velocity channels and previous turn buttons initialize an affine memory warp. The readout is learned. The grid contains no global position or WAD data and does not perform route search. The 48-unit grid spacing yields a local neighborhood around the player. The current grid does not explicitly remember door color or encode long-term map topology. [Model and occupancy implementation](../submission/memory.py)

The matched warm-start probe initializes both residual GRU and occupancy with zero residual readout. The MLP parameters and output are initially identical. Each architecture trains with fresh Adam on CPU. A parity check replays 720 actual legal steps across 18 windows, checks exact logits and critic values, checks episode resets, and confirms that memory states change across actions. This tests initialization and state propagation. Generalization and progress require paired gameplay evaluation after training.

Both architectures trained 12,000 decisions each at seeds 59 and 83 using 32-decision sequences and 12 burn-in decisions. After training, clearing memory at every replay step changes 4 and 0 of 720 greedy actions for the GRU and 17 and 46 for occupancy. Each memory's parameters and carried logits had changed. All 22 sampled episode boundaries per model still produced exact reset parity. This test measures the policy's sensitivity to state on recorded training trajectories. [Trained-state ablation](../runs/adaptive-campaign-20261127/memory/trained_memory_usage.json)

| Architecture after 12,000 decisions | Seed | Switch exits | Entered range | Familiar | Gauntlet deaths | Decisions/sec |
|---|---:|---:|---:|---:|---:|---:|
| Fresh-Adam MLP | 59 | 9/54 | 21/54 | 13/120 | 24/30 | 344.9 |
| Residual GRU | 59 | 9/54 | 12/54 | 21/120 | 20/30 | 254.8 |
| Occupancy | 59 | 12/54 | 21/54 | 10/120 | 17/30 | 212.1 |
| Fresh-Adam MLP | 83 | 6/54 | 9/54 | 5/120 | 20/30 | 298.6 |
| Residual GRU | 83 | 12/54 | 18/54 | 10/120 | 22/30 | 234.2 |
| Occupancy | 83 | 12/54 | 21/54 | 17/120 | 20/30 | 244.2 |

The residual GRU improved familiar exits relative to its equally trained MLP controls. Its range changes were −9 and +9 episodes. Both occupancy seeds completed more switches than their matched MLPs while remaining below the frozen warm policy's 18/54. Occupancy seed 59 reduced gauntlet deaths by three relative to the frozen start. Its familiar completion dropped by 20 episodes. No architecture passed the development gate. The four recurrent learners used 48,000 decisions and 204.1 seconds of learner time. The two MLP controls are shared with the imitation comparison. Throughput was measured during the campaign while evaluation load varied. [GRU paired contrast](../runs/adaptive-campaign-20261127/evaluation/paired-followup-residual-gru.json) · [Occupancy paired contrast](../runs/adaptive-campaign-20261127/evaluation/paired-followup-occupancy.json)

Initialize the memory candidates with the exact demonstration set and selected architecture, then use the same PPO follow-up command with the resulting checkpoint as its source.

```bash
"$py" -m scripts.adaptive_followup initialize --model occupancy --seed 59 \
  --dataset "$study/imitation/data/trajectories.pt" --out "$study/memory/occupancy-59-reproduction.pt"
"$py" -m scripts.adaptive_followup ppo --out "$study" --source "$study/memory/occupancy-59-reproduction.pt" \
  --seed 59 --label occupancy-59-reproduction --decisions 12000 --seconds 250
```

## Exploration and reward choices

The fixed course reward already caps credits from first-time coverage, key pickup, door lift and switch approach. These rewards use training-only instrumentation. The new mixed comparison keeps all reward coefficients fixed. Historical performance with additional trigger-discipline penalties regressed, so those penalties stay out of the controlled arms. [Course reward](../training/mechanics.py) · [Baseline reward audit](EXIT_RECOVERY_STUDY.md)

RND learns prediction error for the features of a frozen random network. An implementation here would require a separately optimized predictor, running normalization, bounded intrinsic weighting and an isolated paired ablation. An episodic count bonus has far less implementation overhead and existing capped first-visit rewards provide the current reference. Any novelty test should track exits, stalls, held-out approach and lock progression alongside reward. [Burda et al., 2018](https://arxiv.org/abs/1810.12894)
