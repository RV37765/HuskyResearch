--------------------------------
What the robot is and what it does:
--------------------------------
The Clearpath Husky A200 is a four-wheeled differential-drive ground robot - it steers like a tank, varying speed between the left and right sides. Mounted on top is a Velodyne VLP-16 LiDAR (Light Detection and Ranging), which spins 360° and fires 16 laser beams simultaneously, 10 times per second. Each beam measures the distance to whatever it hits. The result is a 3D point cloud - roughly 300,000 distance measurements per second covering a full sphere around the robot. The robot navigates entirely from this geometry. It has no camera, no GPS, no color perception.

--------------------------------
SLAM - Building the map:
--------------------------------

SLAM stands for Simultaneous Localization and Mapping. The problem it solves: to build a map you need to know where you are, but to know where you are you need a map. It solves both at once.

We use a package called slam_toolbox. It can't work directly with the 3D point cloud - SLAM operates in 2D. So first we run the point cloud through pointcloud_to_laserscan, which applies a height filter: keep only points between 5cm and 70cm above the robot's base. This horizontal slice captures cotton plant stems cleanly and discards ground returns and overhead clutter. The output is a 2D laser scan - one flat ring of distance measurements.

As you drive the robot, slam_toolbox takes each incoming scan and compares it to what it's already seen. It uses wheel odometry (encoder counts on each wheel) to estimate how far the robot has moved between scans, then stitches the scans together. The output is an occupancy grid - a grayscale image where black pixels are obstacles, white is free space, and grey is unknown (never scanned). Each black dot in a crop row corresponds to a plant stem.

The key challenge in a symmetric environment like a crop field is loop closure - when the robot returns to a previously mapped area, SLAM recognizes it and corrects accumulated drift. That's why driving each aisle twice produces a better map.

The map is saved as two files: crop_map.pgm (the image) and crop_map.yaml (metadata: resolution in meters/pixel, origin coordinates).

--------------------------------
Localization - Where am I on the map?:
--------------------------------
Once the map exists, the robot doesn't need to re-explore. It just needs to figure out where it currently is on the saved map. This is handled by AMCL (Adaptive Monte Carlo Localization), which uses a particle filter.

At startup, AMCL spawns thousands of hypotheses - particles - spread across the map. Each particle is a guess: "maybe the robot is here, facing this direction." As new laser scans arrive, each particle predicts what the scan should look like from its hypothesized position, and compares that prediction to the actual scan. Particles with accurate predictions survive; inaccurate ones die off. After a few seconds, all surviving particles converge to a tight cluster at the robot's true location.

The challenge in a cotton field is symmetry - every aisle looks identical (same width, same plant spacing). AMCL can't tell which one it's in. We address this in simulation by adding boundary walls at the field edges, which give AMCL unique reference points. On the real field, fence lines, equipment sheds, or the end of the field play that role.

--------------------------------
Navigation - Getting from A to B
--------------------------------

With localization working, move_base handles autonomous driving. It has two layers:

The global planner looks at the full saved map and plans an optimal path from the robot's current position to the goal. This path avoids known obstacles (the plant rows).

The local planner (DWA - Dynamic Window Approach) executes that path in real time, sending velocity commands to the wheels. It also watches the live laser scan for obstacles not on the saved map - anything that appeared since mapping.

The output of the local planner is a cmd_vel message (linear and angular velocity). In simulation we send this directly to the wheel controllers. On the real Husky, it goes through twist_mux, a priority manager that decides whose velocity command wins - navigation, joystick teleop, or e-stop all compete for control.

You send a goal via RViz using the 2D Nav Goal tool. The terminal confirms arrival with Goal reached!.

--------------------------------
Row detection - What we're building next
--------------------------------

This is the original research contribution - it doesn't exist yet. The idea: once the robot is inside a crop row aisle, it should steer itself to stay centered between the two plant lines, no GPS or map needed.

The algorithm reads the live laser scan and extracts two clusters of points - the plants to the left (~90°) and the plants to the right (~270°). It calculates the distance to each side. If the robot is centered, both distances are equal. If the robot drifts right, the right distance shrinks and the left distance grows. The lateral offset drives a proportional steering correction. This is a reactive controller - it doesn't plan ahead, it just reacts to what it currently sees, 10 times per second.

--------------------------------
How the pieces connect
--------------------------------

VLP-16 → /velodyne_points (3D point cloud)
    → pointcloud_to_laserscan → /scan (2D laser ring)
        → slam_toolbox → /map (occupancy grid, during mapping)
        → AMCL → /amcl_pose (robot's estimated position)
        → move_base → /cmd_vel (velocity commands)
            → wheel controllers → robot moves
Everything communicates through ROS topics - named data channels that nodes publish to and subscribe from. The LiDAR driver publishes, SLAM subscribes. AMCL subscribes to both the map and the scan. move_base subscribes to the pose and publishes velocity. Nothing is hardwired - any node can be swapped or inspected independently
