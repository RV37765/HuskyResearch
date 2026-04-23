#!/usr/bin/env python3
"""
Row Follower — keeps the Husky centered between crop rows using live LiDAR data.

How it works:
  1. Reads /scan_fixed (VLP-16 laser scan, corrected by scan_relay)
  2. Extracts points on the LEFT and RIGHT sides of the robot
  3. Computes average distance to obstacles on each side
  4. If left is closer than right → steer right, and vice versa
  5. Publishes a Twist to /cmd_vel to correct heading

Run on top of husky_slam_nav.launch:
  1. roslaunch husky_custom_sim husky_slam_nav.launch
  2. rosrun husky_custom_sim row_follower.py
  Robot will drive forward, center itself between walls/rows, and
  slam_toolbox builds a map of the environment as it goes.

At row end (both walls disappear), robot slows and holds last correction
to handle corners gracefully rather than driving blind.
"""

import math
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry

# --- Tuning parameters ---
FORWARD_SPEED   = 0.20      # m/s — slightly faster for smoother mapping
ANGULAR_GAIN    = 1.2       # How aggressively to correct heading
MAX_ANGULAR     = 0.4       # Max turn rate (rad/s)

# When the robot is within this distance of an active move_base goal,
# row_follower stops publishing and yields to move_base for clean goal settling.
GOAL_YIELD_RADIUS = 1.5     # meters

# --- Scan angle windows ---
# Angles relative to robot forward (0 = straight ahead, positive = left)
LEFT_ANGLE_MIN  = 0.2       # ~11 degrees
LEFT_ANGLE_MAX  = 1.5       # ~86 degrees
RIGHT_ANGLE_MIN = -1.5      # ~86 degrees
RIGHT_ANGLE_MAX = -0.2      # ~11 degrees

# Max range to consider — ignore distant walls, focus on immediate row boundaries
MAX_RANGE = 5.0             # meters

# Minimum valid points required on each side to use that reading
MIN_POINTS = 5

# How long to hold the last correction when walls disappear (seconds)
COAST_DURATION = 2.0


class RowFollower:
    def __init__(self):
        rospy.init_node('row_follower')

        self.cmd_pub  = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.scan_sub = rospy.Subscriber('/scan_fixed', LaserScan, self.scan_callback)
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        self.odom_sub = rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)

        self.latest_scan    = None
        self.last_angular   = 0.0       # last valid correction — used during coasting
        self.last_seen_time = None      # when we last had valid readings on both sides
        self.current_goal   = None      # active move_base goal position
        self.current_pos    = None      # current robot position from odometry

        rospy.loginfo("Row follower started.")
        rospy.loginfo(f"Speed: {FORWARD_SPEED} m/s  |  Gain: {ANGULAR_GAIN}  |  Max range: {MAX_RANGE}m")
        rospy.loginfo(f"Goal yield radius: {GOAL_YIELD_RADIUS} m")

    def scan_callback(self, msg):
        self.latest_scan = msg

    def goal_callback(self, msg):
        self.current_goal = msg.pose.position

    def odom_callback(self, msg):
        self.current_pos = msg.pose.pose.position

    def near_goal(self):
        """True when robot is within GOAL_YIELD_RADIUS of the active move_base goal."""
        if self.current_goal is None or self.current_pos is None:
            return False
        dx = self.current_goal.x - self.current_pos.x
        dy = self.current_goal.y - self.current_pos.y
        return math.sqrt(dx * dx + dy * dy) < GOAL_YIELD_RADIUS

    def get_side_distance(self, msg, angle_min, angle_max):
        """Average distance to obstacles within an angular window. Returns None if insufficient points."""
        distances = []
        for i, distance in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            if angle_min <= angle <= angle_max:
                if msg.range_min < distance < min(msg.range_max, MAX_RANGE):
                    if not np.isnan(distance) and not np.isinf(distance):
                        distances.append(distance)
        if len(distances) < MIN_POINTS:
            return None
        return np.mean(distances)

    def run(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if self.latest_scan is None:
                rate.sleep()
                continue

            # Yield to move_base when close to goal — prevents cmd_vel fighting
            if self.near_goal():
                rospy.loginfo_throttle(2.0,
                    f"Near goal — yielding to move_base (within {GOAL_YIELD_RADIUS} m)")
                rate.sleep()
                continue

            scan = self.latest_scan
            left_dist  = self.get_side_distance(scan, LEFT_ANGLE_MIN,  LEFT_ANGLE_MAX)
            right_dist = self.get_side_distance(scan, RIGHT_ANGLE_MIN, RIGHT_ANGLE_MAX)

            twist = Twist()
            twist.linear.x = FORWARD_SPEED

            if left_dist is not None and right_dist is not None:
                # Both walls visible — normal centering
                error = left_dist - right_dist
                twist.angular.z = -ANGULAR_GAIN * error
                twist.angular.z = max(-MAX_ANGULAR, min(MAX_ANGULAR, twist.angular.z))

                self.last_angular   = twist.angular.z
                self.last_seen_time = rospy.Time.now()

                rospy.loginfo_throttle(1.0,
                    f"L={left_dist:.2f}m  R={right_dist:.2f}m  "
                    f"err={error:.3f}  angular={twist.angular.z:.3f}")

            else:
                # One or both walls missing — coast with last correction briefly
                now = rospy.Time.now()
                coast_ok = (self.last_seen_time is not None and
                            (now - self.last_seen_time).to_sec() < COAST_DURATION)

                if coast_ok:
                    # Hold last angular correction — helps navigate corners
                    twist.angular.z = self.last_angular * 0.5  # damped
                    side = "left" if left_dist is None else "right"
                    rospy.logwarn_throttle(1.0,
                        f"Can't see {side} — coasting with angular={twist.angular.z:.3f}")
                else:
                    # No walls for too long — go straight, don't guess
                    twist.angular.z = 0.0
                    rospy.logwarn_throttle(2.0, "No walls detected — going straight")

            self.cmd_pub.publish(twist)
            rate.sleep()

        # Stop cleanly on shutdown
        self.cmd_pub.publish(Twist())
        rospy.loginfo("Row follower stopped.")


if __name__ == '__main__':
    try:
        follower = RowFollower()
        follower.run()
    except rospy.ROSInterruptException:
        pass
