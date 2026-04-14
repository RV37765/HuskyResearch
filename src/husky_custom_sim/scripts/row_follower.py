#!/usr/bin/env python3
"""
Row Follower — keeps the Husky centered between crop rows using live LiDAR data.

How it works:
  1. Reads /scan_fixed (VLP-16 laser scan, corrected by scan_relay)
  2. Extracts points on the LEFT and RIGHT sides of the robot
  3. Computes average distance to obstacles on each side
  4. If left is closer than right → steer right, and vice versa
  5. Publishes a Twist to /cmd_vel to correct heading

This runs ON TOP of the navigation stack — move_base handles getting the robot
to the right row, row_follower keeps it centered while traveling down it.

To use:
  1. Launch nav stack: roslaunch husky_custom_sim husky_real_nav.launch
  2. Send robot to row entry waypoint via navigate.py
  3. Once inside the row: rosrun husky_custom_sim row_follower.py
  4. Robot will drive forward keeping equal distance to both sides
  5. Ctrl+C to stop when robot reaches end of row
"""

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

# --- Tuning parameters ---
FORWARD_SPEED   = 0.15      # m/s — slow and steady through crop rows
ANGULAR_GAIN    = 1.2       # How aggressively to correct heading (higher = more reactive)
MAX_ANGULAR     = 0.4       # Max turn rate (rad/s) — prevents overcorrection

# --- Scan angle windows ---
# The VLP-16 scan is 360 degrees. We sample left and right side sectors.
# Angles are relative to robot forward (0 rad = straight ahead).
# Positive = left, negative = right (ROS convention).
LEFT_ANGLE_MIN  = 0.2       # ~11 degrees from forward
LEFT_ANGLE_MAX  = 1.5       # ~86 degrees from forward
RIGHT_ANGLE_MIN = -1.5      # ~86 degrees from forward (right)
RIGHT_ANGLE_MAX = -0.2      # ~11 degrees from forward (right)

# Ignore scan points beyond this distance — far obstacles aren't row boundaries
MAX_RANGE = 5.0             # meters — increased to handle wider corridor sections

# Minimum number of valid points required on each side to act
MIN_POINTS = 5


class RowFollower:
    def __init__(self):
        rospy.init_node('row_follower')

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.scan_sub = rospy.Subscriber('/scan_fixed', LaserScan, self.scan_callback)

        self.latest_scan = None
        rospy.loginfo("Row follower started. Waiting for scan data...")
        rospy.loginfo(f"Forward speed: {FORWARD_SPEED} m/s")
        rospy.loginfo(f"Correction gain: {ANGULAR_GAIN}")

    def scan_callback(self, msg):
        self.latest_scan = msg

    def get_side_distance(self, msg, angle_min, angle_max):
        """
        Average distance to obstacles within an angular window.
        Returns None if fewer than MIN_POINTS valid readings exist.
        """
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
        rate = rospy.Rate(10)  # 10 Hz control loop

        while not rospy.is_shutdown():
            if self.latest_scan is None:
                rate.sleep()
                continue

            scan = self.latest_scan

            left_dist  = self.get_side_distance(scan, LEFT_ANGLE_MIN,  LEFT_ANGLE_MAX)
            right_dist = self.get_side_distance(scan, RIGHT_ANGLE_MIN, RIGHT_ANGLE_MAX)

            twist = Twist()
            twist.linear.x = FORWARD_SPEED

            if left_dist is None or right_dist is None:
                # Can't see one side — go straight, don't guess
                twist.angular.z = 0.0
                side = "left" if left_dist is None else "right"
                rospy.logwarn_throttle(2.0, f"Can't see {side} side — going straight")
            else:
                # Positive error = left is closer = steer right (negative angular)
                error = left_dist - right_dist
                twist.angular.z = -ANGULAR_GAIN * error
                twist.angular.z = max(-MAX_ANGULAR, min(MAX_ANGULAR, twist.angular.z))

                rospy.loginfo_throttle(1.0,
                    f"L={left_dist:.2f}m  R={right_dist:.2f}m  "
                    f"err={error:.3f}  angular={twist.angular.z:.3f}")

            self.cmd_pub.publish(twist)
            rate.sleep()

        # Stop the robot when node shuts down
        self.cmd_pub.publish(Twist())
        rospy.loginfo("Row follower stopped.")


if __name__ == '__main__':
    try:
        follower = RowFollower()
        follower.run()
    except rospy.ROSInterruptException:
        pass
