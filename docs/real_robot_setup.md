# Real Robot Setup Guide
**Husky A200 + VLP-16 + NVIDIA Jetson — Physical Hardware**

---

## BEFORE YOU START: Ask Daniel + Run Discovery

### Ask Daniel first (blockers — can't proceed without these)

1. **What is the SSH address?** How do you connect to the robot? (`ssh <user>@<ip>` — what are both?)
2. **Has `catkin_make` been run** on the robot after the repo was cloned, or is it a fresh clone?
3. **Is the workspace sourced in `~/.bashrc`?** Does a new SSH session automatically have ROS commands available?
4. **Is slam_toolbox installed on the robot?** (He had a missing package error on his machine — confirm it's on the robot itself)

---

### Discovery block — run these immediately after SSH, before anything else

Paste all output back to Claude. Each command takes <5 seconds.

```bash
# Who is this machine and what ROS version?
cat /etc/os-release | grep -E "^VERSION="
rosversion -d

# What is already auto-running? (Husky bringup starts nodes on boot)
rostopic list | grep -E "odom|velodyne|joy|cmd_vel|scan"

# Is the VLP-16 publishing? (should be ~10 Hz)
timeout 5 rostopic hz /velodyne_points

# Is the PS4 controller visible as a device?
ls /dev/input/js*

# Are our required packages installed?
rospack find slam_toolbox && echo "OK" || echo "MISSING"
rospack find pointcloud_to_laserscan && echo "OK" || echo "MISSING"
rospack find teleop_twist_joy && echo "OK" || echo "MISSING"

# Are our launch files present on the robot?
ls ~/catkin_ws/src/src/husky_custom_sim/launch/

# Is the TF tree up? base_link → velodyne must exist before launch
rosrun tf tf_echo base_link velodyne
```

---

### What to report back (paste these exact outputs)

| What | Expected | Action if wrong |
|---|---|---|
| `rosversion -d` | `noetic` | Flag — may need launch file changes |
| `rostopic hz /velodyne_points` | `~10 Hz` | VLP-16 not running — see Step 3 |
| `ls /dev/input/js*` | `/dev/input/js0` | PS4 not connected — re-pair before launch |
| `rospack find slam_toolbox` | path printed | Install: `sudo apt install ros-noetic-slam-toolbox` |
| `rospack find teleop_twist_joy` | path printed | Install: `sudo apt install ros-noetic-teleop-twist-joy` |
| `tf_echo base_link velodyne` | transform printed | TF not up yet — wait and retry after bringup confirms |
| `ls .../launch/` | `husky_real_slam.launch` in list | `catkin_make` not run — run it first |

**Paste all of that back before launching anything.** This takes 2 minutes and prevents wasted time troubleshooting mid-session.

---

## STEP 0: Verify What's on the Robot (Do This First)

Different Jetson models ship with different Ubuntu versions. This determines which ROS version is installed.

SSH into the Jetson and run:
```bash
cat /etc/os-release        # Ubuntu version
rosversion -d              # ROS version currently installed
uname -m                   # Should say: aarch64 (ARM64)
```

Expected outcomes:
| Jetson Model | Ubuntu | ROS | Status |
|---|---|---|---|
| Jetson Orin / Xavier NX | 20.04 | Noetic | ✓ Compatible with our code |
| Jetson Xavier AGX | 18.04 or 20.04 | Melodic or Noetic | Check carefully |
| Jetson TX2 | 18.04 | **Melodic** | Launch files need minor changes |

**If ROS Melodic:** The launch file structure is the same, but some package names differ. Flag this and we'll adjust before continuing.

Also check if the robot auto-starts any ROS nodes on boot:
```bash
systemctl list-units | grep ros    # or:
ls /etc/ros/                       # Clearpath often puts startup scripts here
```
If you see active ROS services, the Jetson is already running `roscore` and the base robot drivers. **Do not kill these** — they handle the motor controllers.

---

## STEP 1: Network Setup

The Husky's Jetson is the **ROS Master** — it runs `roscore`. Your laptop is a client that connects to it over the network.

### Find the robot's IP address

```bash
# On the Jetson (via SSH):
hostname -I
# Or check the network interface:
ip addr show | grep inet
```

Note this IP — you'll use it in every session. Clearpath robots often have a static IP like `192.168.131.1` on their internal Ethernet port.

### On your laptop: Configure ROS networking

Add these lines to your laptop's `~/.bashrc`:
```bash
# Replace with the Jetson's actual IP address
export ROS_MASTER_URI=http://192.168.131.1:11311
export ROS_IP=<YOUR_LAPTOP_IP>   # run: hostname -I to find this
```

Then reload:
```bash
source ~/.bashrc
```

**Test it works:**
```bash
# On your laptop (after configuring above):
rostopic list    # Should show topics from the robot, not "connection refused"
```

If `rostopic list` hangs, the network routing is wrong. The most common fix:
- Make sure both machines are on the same network subnet
- On Ethernet direct connection: set laptop to static IP `192.168.131.100`, netmask `255.255.255.0`

### SSH into the Jetson

```bash
ssh <username>@<jetson_ip>
# Typical Clearpath default: ssh administrator@192.168.131.1
# Or: ssh ubuntu@192.168.131.1
```

You'll spend most of your time in this SSH terminal running ROS nodes on the Jetson. RViz runs locally on your laptop.

---

## STEP 2: Clone the Repo on the Jetson

```bash
# On the Jetson (over SSH):
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/RV37765/HuskyResearch.git src
# The repo has its own src/ folder, so path becomes:
# ~/catkin_ws/src/src/husky_custom_sim/...

cd ~/catkin_ws
catkin_make
source devel/setup.bash

# Add to .bashrc so it loads every SSH session:
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
```

**On your laptop** (for RViz only — you don't need the full repo, just ROS):
```bash
# Laptop just needs ROS Noetic installed and ROS networking configured
# No catkin_make needed if you're only running RViz
```

---

## STEP 3: Verify the VLP-16 is Publishing

The Husky's base launch likely already starts the VLP-16 driver on boot. Check:

```bash
# On Jetson or from laptop (with ROS networking configured):
rostopic list | grep velodyne
# Should see: /velodyne_points

rostopic hz /velodyne_points
# Should show: ~10 Hz (10 scans per second)
```

If `/velodyne_points` is not there:
```bash
# Check if velodyne_driver node is running:
rosnode list | grep velodyne

# If not running, start it manually:
roslaunch velodyne_pointcloud VLP16_points.launch
```

### Check the TF tree

This is critical. On the real robot, the URDF publishes all TFs automatically. We do NOT want our manual static_transform_publisher running and conflicting.

```bash
rosrun tf tf_echo base_link velodyne
# Should return a transform without error
# If it errors with "No transform found": TF not publishing yet — check URDF loaded
```

If `base_link → velodyne` is already being published by the URDF, our launch file's `static_transform_publisher` for that frame will conflict. See Step 5 — real hardware launch files handle this.

---

## STEP 4: Set Environment Variables on the Jetson

The Clearpath Husky uses environment variables to configure which sensors are enabled. Check if they're already set:

```bash
# On Jetson:
env | grep HUSKY
```

If not set, add to Jetson's `~/.bashrc`:
```bash
export HUSKY_LASER_3D_ENABLED=1
export HUSKY_LASER_3D_TOPIC=velodyne_points
export HUSKY_LASER_3D_TOWER=0    # 0 = no tower mount (sensor direct to top plate)
```

**Important:** If the Husky already has a startup configuration file (often `/etc/ros/setup.bash` or similar), these may already be set there. Check before duplicating.

---

## STEP 5: Real Hardware Launch Files

The simulation launch files (`husky_crop_slam.launch`, `husky_crop_nav.launch`) cannot be used directly on the real robot because they:
- Start Gazebo (not needed — real world exists)
- Spawn a virtual Husky (not needed — real Husky is already there)
- Publish the sensor TF manually (the real robot URDF already does this)
- Bypass twist_mux (safety manager is active on real robot)

We have separate launch files for real hardware: `husky_real_slam.launch` and `husky_real_nav.launch`.

**Key differences in the real launch files:**
1. No Gazebo, no spawn_husky, no controller_respawner
2. Static TF node is removed (URDF handles it)
3. `cmd_vel` remap is adjusted for twist_mux
4. AMCL and move_base parameters tuned for real sensor noise

See [husky_real_slam.launch](../src/husky_custom_sim/launch/husky_real_slam.launch) and [husky_real_nav.launch](../src/husky_custom_sim/launch/husky_real_nav.launch).

---

## STEP 6: Running SLAM on the Real Field

**Do this before trying navigation.** The simulation crop_map is useless on real hardware — you need a map of the actual field.

### What you need
- The Husky powered on and connected
- SSH terminal into Jetson
- A second terminal on your laptop for RViz
- Enough open space to drive (ideally the actual cotton field, or any representative space first)

### Launch SLAM

```bash
# Terminal 1 — On Jetson (SSH):
roslaunch husky_custom_sim husky_real_slam.launch

# Terminal 2 — On your laptop:
rviz
# Add displays: /map (Map), /scan (LaserScan), /tf (TF)
# You should see laser returns and an empty map
```

### Drive the robot

Use joystick if available. Otherwise keyboard teleop:
```bash
# Terminal 3 — On your laptop:
rosrun teleop_twist_keyboard teleop_twist_keyboard.py
# This publishes to /cmd_vel — make sure twist_mux priority is set correctly
```

**Driving tips for real hardware:**
- Go slower than you think (0.3 m/s max during mapping) — wheel slip ruins odometry
- Drive all aisles you want navigable — unmapped areas will be "unknown" (unnavigable)
- Do 2 full passes per aisle if possible — loop closure corrects drift
- Avoid sharp turns at full speed — scan overlap breaks down
- Stop briefly at the ends of each aisle

### Save the map

When you've mapped the full field, save while SLAM is still running:
```bash
# Terminal 4 — On Jetson (SSH):
rosrun map_server map_saver -f ~/catkin_ws/src/src/husky_custom_sim/maps/real_crop_map
```

This creates `real_crop_map.pgm` and `real_crop_map.yaml`. Copy these to your laptop for inspection:
```bash
# On your laptop:
scp <username>@<robot_ip>:~/catkin_ws/src/src/husky_custom_sim/maps/real_crop_map.* \
    ~/Desktop/
```

Open `real_crop_map.pgm` in any image viewer. You should see:
- Clear black lines where plant rows are
- White aisles between rows
- Distinct features (field boundary, end of rows)

If the map looks smeared or rows don't appear distinct, repeat SLAM with slower driving.

---

## STEP 7: Running Navigation on the Real Field

**Requires:** A completed real_crop_map.pgm/yaml in the maps/ folder.

```bash
# Terminal 1 — On Jetson (SSH):
roslaunch husky_custom_sim husky_real_nav.launch

# Terminal 2 — On your laptop:
rviz
# Open the saved nav_view.rviz config, or add:
# /map (Map), /scan (LaserScan), /amcl_pose (PoseWithCovarianceStamped),
# /particlecloud (PoseArray), /move_base/GlobalPlanner/plan (Path)
```

**Wait for AMCL to converge.** You'll see a cloud of green arrows (particles) in RViz. Initially they're spread out. As the robot gets a few laser scans, they'll tighten to a cluster. This takes 5-15 seconds if the initial pose is close to the spawn position.

**If particles don't converge:**
- Use the "2D Pose Estimate" button in RViz to manually tell AMCL where the robot is
- Click on the robot's actual position on the map, drag the arrow to show heading

**Send a goal:**
- RViz toolbar → "2D Nav Goal"
- Click inside an aisle on the map
- Watch the robot plan and drive

---

## Real-World Parameter Tuning (Expect to Adjust These)

These values will likely need adjustment once you're on real hardware. The simulation values are starting points.

### Scan height filter (`husky_real_slam.launch` and `husky_real_nav.launch`)
```yaml
min_height: 0.05    # May need to raise if ground returns bleed through (try 0.10-0.15)
max_height: 0.70    # May need adjustment based on actual plant height in field
```
**How to check:** In RViz, add the `/scan` topic and look at it while the robot is stationary. You should see returns only from plant stems and walls — not from the ground or open sky.

### AMCL particles
```yaml
min_particles: 500    # Increase from sim values — more noise needs more hypotheses
max_particles: 2000
```

### move_base patience timers
```yaml
# In move_base_params.yaml:
# Real robot runs at real-time (RTF 1.0), not 0.15 like WSL2
# These can be reduced back toward default values:
planner_patience: 5.0      # was 30.0 in sim
controller_patience: 15.0  # was 60.0 in sim
```

### DWA local planner speeds
```yaml
# Real robot in a real field — slower is safer initially
max_vel_x: 0.3         # Reduce from sim value while tuning
min_vel_x: -0.1
max_vel_trans: 0.3
max_vel_theta: 0.5
```

---

## Common Real-Robot Issues and Fixes

**"No transform from base_link to velodyne"**
- Our static TF publisher is conflicting with the URDF's TF. Check `rosnode list` for duplicate TF publishers and kill our manual one. The real launch files should handle this, but verify.

**AMCL particles never converge**
- Environment is too symmetric — real field may have unique features (end of field, equipment). Drive robot toward a distinctive feature first to help AMCL lock on.
- Use "2D Pose Estimate" in RViz to give AMCL a manual hint.

**Robot drives but veers off path**
- Wheel slip — reduce `max_vel_x`. The local costmap's `sim_time` for DWA may also need adjustment.
- Check that `odom_frame_id: odom` and the odometry topic (`/odometry/filtered` on Husky) is publishing correctly.

**cmd_vel sent but robot doesn't move**
- twist_mux priority — our cmd_vel may be below the joystick priority. Check `/joy_teleop/cmd_vel` vs `/cmd_vel` remapping in the launch file.
- E-stop active — check the Husky's status LEDs. Yellow/red = e-stop engaged.

**VLP-16 not publishing**
- Power cycle the sensor (if separate power)
- Check `rosnode list` for velodyne_nodelet_manager
- Check the VLP-16's IP (default `192.168.1.201`) is reachable from Jetson: `ping 192.168.1.201`

---

## Session Workflow Checklist

Every real-robot session:
- [ ] Power on Husky, wait for boot (~30s)
- [ ] SSH into Jetson, verify `rostopic list` works
- [ ] Check `/velodyne_points` is publishing at ~10 Hz
- [ ] Check `rosrun tf tf_echo base_link velodyne` returns valid transform
- [ ] Confirm e-stop is released (blue LED on Husky = ready)
- [ ] Launch SLAM or navigation (see above)
- [ ] Open RViz on laptop for visualization
- [ ] Before shutdown: save map (if SLAM session), `rosnode kill -a` to clean up

---

## Files Specific to Real Hardware

| File | Purpose |
|---|---|
| `launch/husky_real_slam.launch` | SLAM without Gazebo/spawn — for real field mapping |
| `launch/husky_real_nav.launch` | Navigation without Gazebo — for real field driving |
| `maps/real_crop_map.pgm` | Map of the actual field (generated during real SLAM session) |
| `maps/real_crop_map.yaml` | Metadata for the real field map |

The simulation files (`husky_crop_slam.launch`, `husky_crop_nav.launch`) stay unchanged — they're still useful for testing code changes before taking the robot to the field.
