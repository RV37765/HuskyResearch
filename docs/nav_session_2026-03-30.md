# Navigation Session — March 30, 2026

## What We Were Trying to Do
Run the Husky autonomously through the hallway using a saved map (`hallway_strip`). The goal was to send a destination coordinate and have the robot drive there without human input — essentially proving the full nav stack works on the real robot.

---

## What Worked
- **Nav stack launched successfully** — map_server, AMCL, and move_base all started cleanly
- **AMCL localized** — particles converged (covariance ~0.001), robot got a position estimate within the map
- **Move_base planned and executed** — it generated a path, drove the robot, and declared `[move_base] Goal reached!`
- **All major bugs from prior sessions are fixed** — scan_relay running, scan_fixed healthy at 9.9 Hz, particles reduced to 100/500 (no CPU overload), TF publishing continuously (update_min = 0)

## What Didn't Work
- **Physical navigation was wrong** — the robot drove into walls repeatedly instead of through open space
- **"Goal reached" was false** — move_base believed it reached the goal based on its internal position estimate, but that estimate was off from reality

---

## Why It Failed — Root Cause

**AMCL localized to the wrong place.**

The robot's estimated position in the map didn't match where it actually was in the hallway. Move_base planned a path through what looked like open space on the map, but those areas were actually walls in the real world. The local costmap (which uses live scan data) kept seeing real walls and triggering recovery, but the global plan kept routing through the same wrong area because it was based on a bad position estimate.

Two compounding factors:
1. **Featureless hallway** — long uniform walls look nearly identical from every position along them. AMCL can converge confidently to a position that's off by several meters because the scan pattern matches well even when misaligned.
2. **SLAM drift** — the hallway_strip map was built in two passes down the hall. Without strong loop closure features, the map has accumulated drift, so even a correctly localized robot would find obstacle positions slightly wrong.

There was also a **42-second scan gap** after calling `global_localization` (particles spreading across the full map briefly spiked CPU), which caused AMCL to converge after a blind window — possibly to a shifted position.

---

## Key Terminal Signals
| Message | Meaning |
|---|---|
| `Costmap2DROS transform timeout` | AMCL stopped publishing TF (scan gap) |
| `Extrapolation Error... 30s into the past` | Costmap still requesting stale timestamp |
| `No laser scan received for 42s` | scan_fixed was not publishing during particle spread |
| `Got new plan` | Move_base replanned (robot hit something) |
| `Goal reached` | Move_base thinks position was reached — based on bad localization |

---

## Where Things Stand
- The navigation stack is **fully functional end-to-end** (AMCL → move_base → DWA planner → motors)
- Autonomous goal-directed motion, path planning, and recovery behaviors all executed
- The limitation is **localization accuracy**, not code
- The hallway is a difficult environment for map-based localization

---

## What's Needed Next
See [Moving Forward](#moving-forward-how-to-diagnose-and-fix-this) below.

---

## Moving Forward — How to Diagnose and Fix This

### Step 1 — Confirm the localization error with a known landmark
Before changing anything, verify that AMCL is actually wrong:
- Place the robot at a physically known spot (e.g., one end of the hallway, against the wall)
- Run nav, global_localize, drive, check `rostopic echo /amcl_pose -n 1`
- Convert those coordinates to pixel coordinates on the hallway_strip map:
  ```
  pixel_x = (world_x - origin_x) / resolution = (world_x + 97.695) / 0.05
  pixel_y = height - (world_y - origin_y) / resolution = 645 - (world_y + 39.422) / 0.05
  ```
- Open `hallway_strip.pgm` in an image viewer, go to that pixel — does it match the robot's actual location?
- If it's off by more than ~0.5m, AMCL is mislocalizing

### Step 2 — Try a better environment (crop field)
The hallway is the hardest case for AMCL. Crop rows create a repeating geometric pattern that gives AMCL clear features to anchor to. The field may localize significantly better than a blank hallway.

### Step 3 — Improve the SLAM map
If continuing in the hallway:
- Map with 3–4 full passes instead of 2
- Pause at both ends of the hallway (distinct endpoints = loop closure anchors)
- Save map immediately after SLAM — don't let the robot sit idle (drift accumulates)
- Compare the new map's pgm to hallway_strip visually — walls should be single-pixel-thin lines, not thick blurry bands

### Step 4 — Reduce the scan gap on global_localization
The 42s scan gap after `global_localization` is a problem. Fix: after calling the service, wait 5 seconds before driving. This gives AMCL time to spread particles without starving the scan pipeline.

### Step 5 — Consider artificial landmarks
If the hallway remains the test environment long-term: place a cardboard box or chair at a known location. AMCL will lock onto it as a unique feature and localization reliability improves dramatically.
