#!/usr/bin/env python3
"""
Row Follower — heading-locked row detection and centering for the Husky.

How it works:
  1. On first valid scan, locks the robot's current yaw from /odom as the
     committed row heading. This is the direction the robot will follow.
  2. Reads /scan_fixed and extracts LEFT and RIGHT point clusters.
  3. Computes lateral offset (centering error) from average distances.
  4. Computes heading error: difference between current yaw and locked row heading.
  5. Angular correction = lateral correction + heading correction.
     The heading lock prevents the robot from being pulled into doorways or
     side corridors — it can only make small corrections around the row direction.
  6. Coasting logic: when walls disappear, hold last correction briefly.

Why heading lock matters for real crop rows:
  - Crop row "walls" are sparse plant stems with gaps — pure wall-following
    gets pulled into gaps or missing plants.
  - Rows are planted in straight lines. Locking to the entry heading and
    making lateral-only corrections is the correct model for crop navigation.
  - Side corridors and doorways are ignored because they require a heading
    change larger than MAX_HEADING_CORRECTION allows.

Run on top of husky_slam_nav.launch:
  1. roslaunch husky_custom_sim husky_slam_nav.launch
  2. rosrun husky_custom_sim row_follower.py
  Point the robot down the row before starting — the first odom reading
  becomes the committed row heading.
"""

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import tf.transformations

# --- Tuning parameters ---
FORWARD_SPEED   = 0.15      # m/s
LATERAL_GAIN    = 1.2       # How aggressively to correct lateral offset
HEADING_GAIN    = 1.5       # How aggressively to correct heading drift
MAX_ANGULAR     = 0.4       # Max turn rate (rad/s)

# Heading lock: robot cannot deviate more than this from committed row heading.
# Prevents being pulled into doorways or side corridors.
# 0.35 rad ≈ 20 degrees — enough for gentle curves, too small for a side turn.
MAX_HEADING_CORRECTION = 0.35   # radians

# --- Scan angle windows ---
# Angles relative to robot forward (0 = straight ahead, positive = left)
LEFT_ANGLE_MIN  = 0.2       # ~11 degrees
LEFT_ANGLE_MAX  = 1.5       # ~86 degrees
RIGHT_ANGLE_MIN = -1.5      # ~86 degrees
RIGHT_ANGLE_MAX = -0.2      # ~11 degrees

# Max range to consider
MAX_RANGE = 5.0             # meters

# Minimum valid points required on each side
MIN_POINTS = 5

# How long to hold the last correction when walls disappear (seconds)
COAST_DURATION = 2.0


def yaw_from_odom(msg):
    """Extract yaw angle from an Odometry message quaternion."""
    q = msg.pose.pose.orientation
    _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
    return yaw


def angle_diff(a, b):
    """Signed difference between two angles, wrapped to [-pi, pi]."""
    diff = a - b
    while diff >  np.pi: diff -= 2 * np.pi
    while diff < -np.pi: diff += 2 * np.pi
    return diff


class RowFollower:
    def __init__(self):
        rospy.init_node('row_follower')

        self.cmd_pub  = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.scan_sub = rospy.Subscriber('/scan_fixed', LaserScan, self.scan_callback)
        self.odom_sub = rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)

        self.latest_scan    = None
        self.current_yaw    = None      # live yaw from odometry
        self.row_heading    = None      # locked row heading — set once on first valid scan
        self.last_angular   = 0.0
        self.last_seen_time = None

        rospy.loginfo("Row follower started. Waiting for scan data...")
        rospy.loginfo(f"Forward speed: {FORWARD_SPEED} m/s")
        rospy.loginfo(f"Lateral gain: {LATERAL_GAIN}  |  Heading gain: {HEADING_GAIN}")
        rospy.loginfo(f"Max heading correction: {np.degrees(MAX_HEADING_CORRECTION):.0f} deg")

    def scan_callback(self, msg):
        self.latest_scan = msg

    def odom_callback(self, msg):
        self.current_yaw = yaw_from_odom(msg)

    def get_side_distance(self, msg, angle_min, angle_max):
        """Average distance to obstacles within an angular window.
        Returns None if insufficient valid points."""
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
            if self.latest_scan is None or self.current_yaw is None:
                rate.sleep()
                continue

            scan = self.latest_scan
            left_dist  = self.get_side_distance(scan, LEFT_ANGLE_MIN,  LEFT_ANGLE_MAX)
            right_dist = self.get_side_distance(scan, RIGHT_ANGLE_MIN, RIGHT_ANGLE_MAX)

            twist = Twist()
            twist.linear.x = FORWARD_SPEED

            if left_dist is not None and right_dist is not None:
                # Lock row heading on first valid reading
                if self.row_heading is None:
                    self.row_heading = self.current_yaw
                    rospy.loginfo(f"Row heading locked: {np.degrees(self.row_heading):.1f} deg")

                # --- Lateral correction ---
                # How far off-center am I between the two walls?
                lateral_error  = left_dist - right_dist
                lateral_cmd    = -LATERAL_GAIN * lateral_error

                # --- Heading correction ---
                # How far has my yaw drifted from the committed row direction?
                heading_error  = angle_diff(self.current_yaw, self.row_heading)
                heading_cmd    = -HEADING_GAIN * heading_error

                # Clamp heading correction — this is the key constraint.
                # The robot cannot turn more than MAX_HEADING_CORRECTION rad
                # from the row heading, so doorways and side corridors are ignored.
                heading_cmd = np.clip(heading_cmd,
                                      -MAX_HEADING_CORRECTION,
                                       MAX_HEADING_CORRECTION)

                # Combine and clamp total angular output
                twist.angular.z = np.clip(lateral_cmd + heading_cmd,
                                          -MAX_ANGULAR, MAX_ANGULAR)

                self.last_angular   = twist.angular.z
                self.last_seen_time = rospy.Time.now()

                rospy.loginfo_throttle(1.0,
                    f"L={left_dist:.2f}m  R={right_dist:.2f}m  "
                    f"lat_err={lateral_error:.3f}  "
                    f"hdg_err={np.degrees(heading_error):.1f}deg  "
                    f"angular={twist.angular.z:.3f}")

            else:
                # One or both walls missing — coast briefly with last correction
                now = rospy.Time.now()
                coast_ok = (self.last_seen_time is not None and
                            (now - self.last_seen_time).to_sec() < COAST_DURATION)

                if coast_ok:
                    twist.angular.z = self.last_angular * 0.5  # damped coast
                    side = "left" if left_dist is None else "right"
                    rospy.logwarn_throttle(1.0,
                        f"Can't see {side} wall — coasting  angular={twist.angular.z:.3f}")
                else:
                    # Coast expired — hold row heading, go straight
                    if self.row_heading is not None and self.current_yaw is not None:
                        heading_error   = angle_diff(self.current_yaw, self.row_heading)
                        twist.angular.z = np.clip(-HEADING_GAIN * heading_error,
                                                  -MAX_HEADING_CORRECTION,
                                                   MAX_HEADING_CORRECTION)
                        rospy.logwarn_throttle(2.0,
                            f"No walls — holding row heading  "
                            f"hdg_err={np.degrees(heading_error):.1f}deg")
                    else:
                        twist.angular.z = 0.0
                        rospy.logwarn_throttle(2.0, "No walls, no heading lock — going straight")

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
