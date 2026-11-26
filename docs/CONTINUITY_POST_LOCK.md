# Conditional post-lock practice

The post-lock phase reserves two of eight training lanes for exit-switch practice with a preopened, currently unlocked Doom doorway. The other six retain near-switch alignment, walkover exits, keys, combat, exploration and gauntlets. All use the same archived MLP, optimizer configuration, action grouping, eight serial CPU environments and four-tic decision interval. The learner records placement data for the curriculum, reward and diagnostics. The exported agent still receives the official observations alone.

Generated drills start either on the approach side or already across the doorway. The latter never assume a crossing event. They isolate navigation after an unlock, with no required key pickup or lock-opening objective. The authored families vary corridor shape, turns, dead ends, side rooms, widths, heading, distance and distractor switch placement. The standard engine verified every new exit. Approach-side verification required a measured doorway crossing. [Production WAD manifest](../runs/continuity-campaign-20261127/post-lock-maps/manifest.json) · [Engine witnesses](../runs/continuity-campaign-20261127/post-lock-maps/qa.json)

The staged schedule starts its two post-lock lanes in landing layouts, adds corners after 15,360 decisions, opens the new route families after 39,936, and includes the older conditional routes again after 81,920. This schedule restarts when a new learner block starts. A block can also select `--focus balanced`, `--focus landing` or `--focus route`. Lane proportions stay fixed.

```bash
cd /home/darshan/projects/doom-nexus
py=~/doom-test/.venv/bin/python
run_dir=runs/continuity-campaign-20261127
warm=runs/limit-campaign-20261127/training/approach-41/step-314496.pt

# Create another verified copy in a new destination.
"$py" -m scripts.continuity_post_lock prepare --out "$run_dir/post-lock-maps-repeat" \
  --seed 981371 --per-family 12

# Reproduce a standalone metered block after the main learner releases the machine.
"$py" -m scripts.continuity_post_lock learner --out "$run_dir/repeat-block" \
  --resume "$warm" --seed 59 --map-root "$run_dir/post-lock-maps" \
  --active-seconds 450 --seconds 750 --decisions 500000 --meter-id post-lock-repeat

# Resume an interrupted repeat block within the same active limit.
"$py" -m scripts.continuity_post_lock learner --out "$run_dir/repeat-block-resume" \
  --resume "$run_dir/repeat-block/latest.pt" --seed 59 --map-root "$run_dir/post-lock-maps" \
  --active-seconds 450 --seconds 750 --decisions 500000 --meter-id post-lock-repeat
```

Use `--visualize --speed 1` on a standalone learner to see one real rollout. Playback waiting is excluded from the active learning meter. The main controller manages the authorized continuous training budget.

While the main learner works, this viewer opens a native 640×480 window and pins a complete checkpoint per evaluation episode.

```bash
SDL_VIDEODRIVER=x11 "$py" -m scripts.view_adaptive_campaign \
  --run "$run_dir" --panel post-lock --speed 1
```

The matched diagnostic panel contains 12 layouts from two separate development families and runs three paired engine seeds per layout. The checkpoint's actual engine exit, invalid actions and errors are authoritative. The range, alignment and fresh-use measurements sample the state before each four-tic decision. An engine-confirmed exit can occur before all precursor samples are captured. The conditional panel remains separate from the familiar full keycard and gauntlet cases.

```bash
"$py" -m scripts.continuity_post_lock_eval \
  --repo runs/limit-campaign-20261127/exports/approach-41-12000 \
  --map-root "$run_dir/post-lock-maps" \
  --out "$run_dir/post-lock-diagnostics/approach-repeat" --seeds 0 1 2
```

The frozen approach export completed 12/36 episodes on this panel, spread over four of 12 independent layouts. These included nine switch episodes and three walkover episodes. It crossed the preopened door in all 12 approach-side episodes and completed zero of their six switch episodes. [Frozen baseline and stage traces](../runs/continuity-campaign-20261127/post-lock-diagnostics/approach-41-12000/post-lock-funnel.json)
