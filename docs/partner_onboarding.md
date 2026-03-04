# Partner Onboarding: VIPR Cotton Field Navigation

## What are we actually building?

A robot that can:
1. **Map** a cotton field by driving through it once
2. **Localize itself** on that map every time it runs after that
3. **Navigate autonomously** to goals within the field
4. **Follow rows** — staying centered between plant lines without GPS or cameras

The long-term goal is a robot that can enter a cotton row, drive to a weed, stop at it, and allow a tool to remove it — all without a human driving it.

---

## The hardware

**Clearpath Husky A200** — a four-wheeled differential-drive ground robot. About the size of a large cooler. Moves by giving different speeds to the left and right sides (like a tank). Has an onboard computer (NVIDIA Jetson) and connects to a WiFi network.

**Velodyne VLP-16 LiDAR** — a spinning laser range finder mounted on top of the robot. Fires 16 laser beams in all directions simultaneously, 10 times per second. Each beam measures distance to whatever it hits. The result is a 3D cloud of distance measurements covering a full 360° around the robot. We use this to see the world.

**What the robot cannot see:** Color, texture, or images. It only knows distances. Every plant is just a cluster of points. Every wall is a flat surface of points. The robot navigates entirely from geometry.

---

## Part 1: SLAM — Building the Map

### The chicken-and-egg problem

To navigate, a robot needs a map. To build a map, it needs to know where it is. But it can't know where it is without a map. This circular dependency is called the **SLAM problem**: Simultaneous Localization and Mapping.

### How SLAM solves it

Think of someone blindfolded in a room, holding a tape measure:
- They start at the door. That's position zero.
- They take a step, measure distance to the nearest wall, spin around and measure again. They now have a small snapshot of their immediate surroundings.
- They take another step. Their feet tell them approximately how far they've gone (odometry). They take another set of measurements.
- Each new snapshot is compared to the last: does this look like I'm in the same room, just shifted a little? If yes, stitch it together.
- Eventually, if they walk the whole room, they have a complete floor plan — even though they were blindfolded the whole time.

The robot does this with the VLP-16 instead of a tape measure, and wheel encoders instead of footsteps.

### What a SLAM map looks like

The output is a 2D occupancy grid — a grayscale image where:
- **Black pixels** = obstacle (a plant stem, a wall, a fence post)
- **White pixels** = free space (an aisle, open area)
- **Grey pixels** = unknown (the robot never drove there / LiDAR couldn't see it)

At 5cm per pixel resolution, a 13m cotton row field produces a map roughly 300×300 pixels. Each black dot in a row corresponds to a cotton plant's stem as seen by the laser.

### How we use the VLP-16 for SLAM

The VLP-16 produces a 3D point cloud. SLAM works in 2D. We solve this by slicing the 3D cloud at a specific height range — like cutting a horizontal plane through the field at shin-to-waist height. This slice captures the plant stems cleanly without picking up ground noise or overhead clutter.

**The pipeline:**
```
VLP-16 fires lasers in all directions (360° × 16 rings)
           ↓
3D point cloud: ~300,000 points per second
           ↓
Height filter: keep only points between 5cm and 70cm above robot base
           ↓
2D laser scan: one clean horizontal slice
           ↓
SLAM Toolbox: builds and updates the occupancy grid map
           ↓
Map saved to disk as a .pgm image + .yaml metadata file
```

### Running SLAM (what you will actually do)

1. SSH into the robot's Jetson computer
2. Launch the SLAM software
3. Drive the robot manually through all the cotton row aisles using a keyboard
4. When you've covered the full field, save the map with one command
5. That map file is now your robot's knowledge of this environment — it never needs to re-explore

---

## Part 2: Navigation — Using the Saved Map

### The key shift

After SLAM, the map is done. It's a file on disk. The robot no longer needs to explore — it just needs to figure out **where it currently is** on the map it already made.

This is a very different problem. SLAM was about discovery. Navigation is about tracking.

### AMCL: Figuring out "where am I?"

**AMCL** (Adaptive Monte Carlo Localization) is the software that solves this. It uses a technique called a **particle filter**:

Imagine you wake up in the cotton field with no idea which aisle you're in. You look left and right — you see plant rows about 1 meter away. You know there are 6 rows. That narrows it to 5 possible aisles. You then look forward and see a wall 11 meters ahead. That narrows it to one specific position on the map. Now you know where you are.

AMCL does this computationally:
1. Start with thousands of "particle" guesses spread across the map — each one is a hypothesis about where the robot might be
2. The robot gets a laser scan — compare each particle's predicted scan (from the map) to the actual scan
3. Particles that predict accurately survive; particles that predict poorly die
4. After a second or two, all surviving particles cluster at the robot's true location

**The challenge in a symmetric field:** If all aisles look identical to the laser (same distance left and right, same distance front and back), AMCL can't tell which aisle it's in. This is why we added boundary walls to the simulation — the outer field boundaries are asymmetric and give AMCL unique reference points. On the real field, fence posts, equipment sheds, or tree lines play this role naturally.

### move_base: Getting from A to B

Once AMCL knows where the robot is, `move_base` handles driving to a goal:

1. You (or a script) send it a goal: "go to position (X, Y) on the map"
2. It plans a path from current position to goal using the saved map (global path)
3. It sends velocity commands to the wheels to follow that path
4. While driving, the live laser scan detects any obstacles not on the saved map (local avoidance)
5. When it arrives, it reports "Goal reached"

The robot drives in real space while navigating on a map — like following GPS directions on a road map.

### Running Navigation (what you will actually do)

1. SSH into the Jetson, launch the navigation stack with the saved map
2. Open RViz on your laptop — you'll see the robot on the map with a particle cloud
3. Wait for AMCL to converge (particle cloud tightens to a single cluster)
4. Click "2D Nav Goal" in RViz, click a point inside an aisle
5. Watch the robot plan and drive there

---

## Part 3: Row Following — The Research Goal

Navigation gets the robot *into* the field and *into* an aisle entrance. Row following takes over once the robot is in the aisle.

**What row following does:** Keeps the robot centered between two plant lines as it drives from one end of the row to the other. No goal points needed — the robot steers by reading plant distances.

**How it works (the algorithm we're building):**
```
Live laser scan arrives (360° snapshot)
           ↓
Extract left cluster: points at ~90° angle, 0.5–2.0m away → left plant distance
Extract right cluster: points at ~270° angle, 0.5–2.0m away → right plant distance
           ↓
Lateral offset = (left_distance - right_distance) / 2
If offset > 0: robot is too far right → steer left
If offset < 0: robot is too far left → steer right
           ↓
Publish velocity command: forward speed constant, angular rate proportional to offset
```

This is called a **reactive controller** — it doesn't plan ahead, it just reacts to what it currently sees. Simple, fast, and robust to sensor noise.

**The full autonomous flow:**
```
1. AMCL: robot knows it's at the aisle entrance (X=-2, Y=0)
2. move_base: drives robot to aisle start position (X=0, Y=0)
3. Row follower activates: takes over cmd_vel control
4. Robot drives centered between row 3 and row 4 to the far end
5. Robot stops, reports success
6. Can repeat for adjacent rows
```

---

## The Software Stack (What All of This Is)

**ROS (Robot Operating System)** is not an operating system — it's a middleware framework. Think of it as a message bus: software components ("nodes") publish and subscribe to named channels ("topics"). The LiDAR driver publishes to `/velodyne_points`. SLAM subscribes to that and publishes to `/map`. AMCL subscribes to `/map` and `/scan`. Everything talks to everything through topics.

**Key ROS concepts you'll encounter:**

| Term | What it is |
|---|---|
| Node | One running program (e.g., slam_toolbox, amcl, move_base) |
| Topic | A named data channel (e.g., `/scan`, `/map`, `/cmd_vel`) |
| Launch file | An XML file that starts multiple nodes at once |
| TF | The "transform" system — tracks where every sensor/part is relative to the robot |
| RViz | The visualization tool — renders the robot, map, scans, paths in 3D |
| rosbag | Records all topic data to a file for replay later |

---

## The File Structure

Everything lives in the ROS package `husky_custom_sim` (inside the catkin workspace):

```
husky_custom_sim/
├── launch/
│   ├── husky_crop_slam.launch    ← Step 1: map the field
│   └── husky_crop_nav.launch     ← Step 2: navigate on the saved map
├── config/
│   ├── slam_toolbox_params.yaml  ← SLAM tuning (resolution, scan range)
│   ├── global_costmap.yaml       ← Global planner's obstacle layer
│   ├── local_costmap.yaml        ← Local planner's obstacle layer
│   └── move_base_params.yaml     ← Speed, tolerances, patience timers
├── maps/
│   ├── crop_map.pgm              ← The actual map image (black/white)
│   └── crop_map.yaml             ← Map metadata (resolution, origin)
└── crop_row.world                ← Gazebo simulation world (simulation only)
```

The `launch/` files are what you run. They start all required nodes in the right order.

---

## What You Need to Contribute

To work on the physical robot side, you need:

1. **A laptop running Ubuntu 20.04** (dual-boot or native — VM works but RViz is slow in a VM)
2. **ROS Noetic installed** on your laptop
3. **Network access to the Husky** — either WiFi (if the robot has a WiFi adapter) or Ethernet
4. **SSH access** to the Husky's Jetson computer (Ryan has the credentials)
5. **The GitHub repo cloned** into a catkin workspace on both your laptop and the Jetson

From there, the workflow is:
- You SSH into the Jetson and run SLAM/navigation nodes there (computation on-robot)
- You run RViz on your laptop for visualization
- You use keyboard teleop from your laptop during SLAM

The most useful thing you can do right now: **get Ubuntu 20.04 + ROS Noetic running on your laptop** and clone the repo. Everything else follows from there.

---

## Key Things That Are Different on the Real Robot vs Simulation

| Simulation | Real Robot |
|---|---|
| Gazebo generates fake sensor data | VLP-16 publishes real laser data |
| Flat ground, no noise | Uneven terrain, reflections, sunlight interference |
| Perfect wheel odometry | Wheel slip, encoder noise |
| We start Gazebo ourselves | Jetson boots and starts the robot automatically |
| We publish the sensor TF manually | URDF on Jetson publishes all TFs automatically |
| Bypassed safety/velocity manager | Full safety stack active — must use correct cmd_vel topic |
| ROS runs on one machine | ROS master on Jetson, display on your laptop (network required) |

**None of this means the code is wrong** — it means some parameters (height filter, scan noise tolerance, AMCL particle counts) will need adjustment once we run on real hardware. This is normal in robotics. The architecture is correct.

---

## One-Sentence Summary Per Component

- **VLP-16** — spinning laser that measures distance to everything in 360°
- **pointcloud_to_laserscan** — slices the 3D laser data to a 2D ring at plant-stem height
- **slam_toolbox** — builds a map of the field as the robot drives through it
- **map_server** — loads that saved map so other nodes can use it
- **AMCL** — figures out where the robot is on the saved map using laser scan matching
- **move_base** — drives the robot from its current location to a goal point
- **Row follower** (to be built) — reads left/right plant distances and steers between rows
- **RViz** — visualizes everything: robot position, map, laser scan, planned path
