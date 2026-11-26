# Phase 2 baseline failure analysis

The committed MLP was replayed on the previously exposed diagnostic split with its saved stochastic action mode, seeds 0 and 1, and the 2,100-tic cap. The replays reproduced the original exit result and engine tic count for all 20 episodes. Decision traces are in `runs/phase2/diagnostics/`. Positions and event counters are diagnostic-only. No privileged fields were passed to the policy.

| Scenario | Tier | Exits | Timeouts | Deaths | Keys | Doors | Unlocks |
|---|---:|---:|---:|---:|---:|---:|---:|
| Movement control | 0 | 4/4 | 0 | 0 | 0 | 0 | 0 |
| Combat | 1 | 4/4 | 0 | 0 | 0 | 0 | 0 |
| Exploration | 2 | 2/4 | 2 | 0 | 0 | 2 | 0 |
| Keys | 3 | 0/4 | 4 | 0 | 3 | 0 | 0 |
| Gauntlet | 4 | 0/4 | 0 | 4 | 0 | 0 | 0 |

| Failed map and seed | Observed sequence |
|---|---|
| Exploration 01 / 0 | Opened one door, revisited cells 50 times and went 433 decisions without a new cell before timing out. |
| Exploration 01 / 1 | Opened one door, revisited cells 45 times and went 204 decisions without a new cell before timing out. |
| Keys 00 / 0 | The key was visible on 69 decisions and never collected. Three rising-edge USE commands occurred with a nearby locked-door ray and no key in inventory. |
| Keys 00 / 1 | Collected the key and timed out. There were three front-ray locked-door opportunities across the episode and zero rising-edge USE commands during those opportunities. |
| Keys 01 / 0 | Collected the key and timed out with no front-ray locked-door opportunities. There were 49 cell revisits. |
| Keys 01 / 1 | Collected the key and timed out with no front-ray locked-door opportunities. There were 54 cell revisits. |
| Gauntlet 00 / 0 | Died at 102 damage with zero kills. Attacked on 7 of 18 decisions with a visible enemy. |
| Gauntlet 00 / 1 | Died at 114 damage with zero kills and zero decisions with a visible enemy. |
| Gauntlet 01 / 0 | Died at 102 damage after one kill. Had four nearby front-ray locked-door USE rising edges without collecting the key. |
| Gauntlet 01 / 1 | Died at 105 damage after one kill. Attacked on 28 of 49 decisions with a visible enemy. |

Wall-adjacent forward commands with under two units of motion occurred only four times across the failures. Low-motion turns and repeated cell transitions were more common. All ten runs selected an empty weapon at some point, but every frame had ammunition in at least one of the four tracked reserves. The data support inefficient weapon selection and severe gauntlet damage. They do not establish the cause of each hit.

The door-opportunity diagnostic counts door rays within 64 units across the front three rays and distinguishes a new USE press from holding USE. This is a sensor proxy for reachability, not an engine use-line check. A return through previously visited cells can be productive backtracking. Short reversals, long gaps without discovering a cell and failure to approach the lock after obtaining the key provide stronger evidence together. Observations do not expose a generic exit indicator, so locating an exit cannot be distinguished from passing it without activating it with these traces alone.

The original generator has corridor movement, single-opponent combat, random branched exploration, a key detour, and a three-enemy gauntlet. Its gauntlet has only one lock. It underrepresents multi-lock ordering, genuinely scarce ammunition and varied combat layouts. Training uses the verified train maps. The historical test split is now diagnostic-only, and the new final test protocol is isolated from experiment selection.
