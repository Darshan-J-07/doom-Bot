# Reproducible demonstration and technical defense

Use the commands in README to reproduce held-out evaluation. Add `--record-dir
recordings` to save real ViZDoom LMP replays. `scripts/demo.py` saves a raw RGB
frame array and episode metrics on one held-out map after training succeeds.
No trained replay or viewable GIF was produced in this blocked build. Display the metrics and the selected
checkpoint configuration beside the replay. Never promise a successful exit,
keycard solution or combat win unless that recorded episode shows it.

Suggested presentation sequence: show the trained checkpoint provenance; open a
held-out replay; report survival, kills, distinct cells, doors, keys and exits;
explain the spatial encoder and memory; compare the measured baseline; close with
the observed failure cases. Use only actual recorded gameplay. Replay generation
does not certify the hidden evaluation distribution.

| Likely question | Defensible answer |
|---|---|
| Why spatial telemetry? | The wrapper already computes local geometry and entities. This keeps the network small on laptop hardware. No measured RGB ablation is claimed. |
| How does memory work? | If the selected model is recurrent, a 128-unit GRU updates a learned hidden vector each step. It is a compressed history, not a guaranteed map. |
| Partial observability? | Only egocentric sensors and previous action reach the policy. Recurrence can retain past evidence; long dependencies remain difficult. |
| Why PPO? | It permits on-policy learning of a compact policy, with clipped changes and explicit recurrent sequences. It is implemented here rather than imported as a default agent. |
| What is original? | The sensor-bearing circular ray encoder, masked entity fusion, action factorization and integrated recurrent training pipeline are custom code in this project. |
| Generalization? | Training, validation and test use separate map seeds/hashes with rotations and placement variation. Only measured held-out outcomes support a capability claim. |
| Reward hacking? | No stationary survival bonus; novelty and door openings pay once per episode. Exit reward dominates individual shaping events. Wandering incentives still need monitoring. |
| Variable enemies? | Shared entity encoder and masked mean/maximum handle 0-32 visible slots without NaNs. This is tested, not proof of combat skill. |
| New layouts or loops? | Actions remain learned. There is no route planner or heuristic loop escape; inadequate training can still produce deadlocks. |
| Cross-map leakage? | Reset clears all recurrent state, previous actions and local RNG. Independent-instance and reset tests cover this. |
| Hardware constraints? | Cite measured `benchmarks/performance.json` and the official validator log. Windows CPU measurements do not certify RTX3090 CUDA or the Linux sandbox. |
| Weaknesses? | Quote the actual held-out metrics and observed replay failures. Small generated maps and brief training are major limits. |

Before submission: create a public team repository from the official template
during the event, transfer and commit these changes and LFS weights, use the
organizer base image and exact dependency lock, rerun all applicable tests and the
unmodified validator, then verify the pushed commit and downloadable LFS object.
No publication has been performed by this local build.
