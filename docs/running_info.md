# Running Info

Quick reference for running each workflow. Have two Ubuntu (WSL) terminals open: **Terminal 1** (primary, runs the launch file) and **Terminal 2** (supporting, for secondary commands like saving maps).

---

## 1. SLAM — Crop Row Simulation

Builds a map by driving the robot manually through the simulated field.

**Terminal 1:**
```bash
cd ~/catkin_ws && source devel/setup.bash
roslaunch husky_custom_sim husky_crop_slam.launch
```

Wait ~2 minutes for Gazebo to load and the controllers to come online — you'll see `Controller Spawner: Loaded controllers: husky_joint_publisher, husky_velocity_controller`. Gazebo and RViz open automatically.

Drive using the xterm teleop window (click it first to give it keyboard focus):
- `i` = forward, `,` = backward, `j`/`l` = turn, `k` = stop
- Cover all 5 aisles, including the outer ones — the boundary walls at the field edges must appear in the map for localization to work later

**Save the map (Terminal 2, while SLAM is still running in Terminal 1):**
```bash
source ~/catkin_ws/devel/setup.bash
rosrun map_server map_saver -f $(rospack find husky_custom_sim)/maps/crop_map
```

You should see `[INFO] Map saved`. The map saver must run while SLAM is active — after you stop it, the map topic goes offline.

**Naming convention:**
- `crop_map` — active map, always loaded by the navigation launch
- `crop_map_YYYYMMDD` — date-stamped backup (e.g. `crop_map_20260312`)
- `crop_map_label` — descriptive variant (e.g. `crop_map_6row_v2`)
- `real_crop_map` — physical field map (never overwrite with a sim map)

**Copy to Windows repo (Terminal 2):**
```bash
cp ~/catkin_ws/src/src/husky_custom_sim/maps/crop_map.* \
   /mnt/c/Users/ryanv/OneDrive/Desktop/VIPR/HuskyResearch/HuskyResearch/src/husky_custom_sim/maps/
```

Stop SLAM with `Ctrl+C` in Terminal 1 when done.

---

## 2. Navigation — Crop Row Simulation

Loads the saved map, localizes the robot on it, and drives to goals you set in RViz.
Requires `crop_map.pgm` + `crop_map.yaml` in `maps/` from a current SLAM run.

**Terminal 1:**
```bash
cd ~/catkin_ws && source devel/setup.bash
roslaunch husky_custom_sim husky_crop_nav.launch
```

Wait ~2 minutes for controllers. Gazebo and RViz open automatically.

In RViz you'll see a cloud of green arrows (AMCL particles) near the robot's start position. Wait for them to tighten into a cluster — this means the robot knows where it is. Then use **2D Nav Goal** in the RViz toolbar, click inside an aisle on the map, and watch the robot plan and drive. Start with the center aisle (Y≈0) to verify before testing others.

Watch Terminal 1 for: `Goal reached!`

**If particles don't converge:** Use **2D Pose Estimate** in RViz — click the robot's actual map position and drag to set heading. This gives AMCL a manual starting hint.

**If goal fails or robot spins in place:** Click closer to the center of the aisle (avoid the edges near plant rows). If it still fails, check that the map looks correct in `src/husky_custom_sim/maps/crop_map.pgm`.

Stop with `Ctrl+C` in Terminal 1.

---

## 3. Real Robot — SLAM

*(To be added after first real-hardware SLAM session)*

---

## 4. Real Robot — Navigation

*(To be added after real map is generated and navigation is verified)*
