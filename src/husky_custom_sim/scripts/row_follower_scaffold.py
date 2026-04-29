#!/usr/bin/env python3
"""
Row Follower — Scaffolding Edition (test script, do not replace row_follower.py)

Instead of a single L/R measurement at 90°, this script measures the wall
at three angular positions per side (forward, lateral, rear) and converts
each to a true perpendicular wall distance via:

    L_perp = measured_distance * sin(|angle|)

This gives a spatial profile of the corridor rather than a single point.
The weighted average across scaffold points drives centering, and the rear
point anchors heading at junctions where the lateral reading disappears.

Scaffold layout (top-down, robot faces right):

    rear-left   lateral-left   fwd-left
         \\           |          /
          \\          |         /
    ═══════[  HUSKY  ]═══════════  forward →
          /           |         \\
         /            |          \\
    rear-right  lateral-right  fwd-right

Junction detection:
    If lateral reading exceeds MAX_WALL_RANGE but rear is still valid,
    the wall just ended → coast using last correction.
    Rear scaffolding acts as a geometric heading anchor.

Run alongside husky_slam_nav.launch:
    roslaunch husky_custom_sim husky_slam_nav.launch
    rosrun husky_custom_sim row_follower_scaffold.py
"""

import math
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry

# ── Tuning ────────────────────────────────────────────────────────────────────
FORWARD_SPEED    = 0.20     # m/s
ANGULAR_GAIN     = 0.9      # lower than row_follower.py — scaffolding is already smoother
MAX_ANGULAR      = 0.4      # rad/s

# Treat any wall reading beyond this as open space (junction or end of row)
MAX_WALL_RANGE   = 2.5      # meters perpendicular

# Minimum LiDAR returns required in a window to trust it
MIN_POINTS       = 6

# Coasting when walls disappear
COAST_DURATION   = 3.5      # seconds — longer than base script to clear junctions

# Goal proximity — yield to move_base near a waypoint
GOAL_YIELD_RADIUS = 1.5     # meters

# ── Scaffold windows (radians, robot frame: 0=forward, +left, -right) ────────
#
# Three windows per side. Angles chosen so rays hit the wall at roughly
# equal physical spacing along the corridor.
#
#   Forward  (~20°–45°): looks ahead  — earliest warning of corridor shape
#   Lateral  (~50°–85°): looks beside — primary centering signal
#   Rear     (~95°–130°): looks behind — geometric anchor at junctions
#
LEFT_WINDOWS = [
    (np.radians(20),  np.radians(45)),   # forward-left
    (np.radians(50),  np.radians(85)),   # lateral-left
    (np.radians(95),  np.radians(130)),  # rear-left
]

RIGHT_WINDOWS = [
    (np.radians(-45), np.radians(-20)),  # forward-right
    (np.radians(-85), np.radians(-50)),  # lateral-right
    (np.radians(-130),np.radians(-95)),  # rear-right
]

# Weights for weighted average: forward matters most, rear least
SCAFFOLD_WEIGHTS = [1.5, 1.0, 0.5]

# Window labels for logging
LABELS = ['fwd', 'lat', 'rear']


class RowFollowerScaffold:
    def __init__(self):
        rospy.init_node('row_follower_scaffold')

        self.cmd_pub  = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.scan_sub = rospy.Subscriber('/scan_fixed', LaserScan, self.scan_cb)
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_cb)
        self.odom_sub = rospy.Subscriber('/odometry/filtered', Odometry, self.odom_cb)

        self.latest_scan    = None
        self.current_goal   = None
        self.current_pos    = None
        self.last_angular   = 0.0
        self.last_seen_time = None

        rospy.loginfo("=== row_follower_scaffold started ===")
        rospy.loginfo(f"Speed: {FORWARD_SPEED} m/s  Gain: {ANGULAR_GAIN}  "
                      f"MaxWall: {MAX_WALL_RANGE} m  Coast: {COAST_DURATION} s")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def scan_cb(self, msg):
        self.latest_scan = msg

    def goal_cb(self, msg):
        self.current_goal = msg.pose.position

    def odom_cb(self, msg):
        self.current_pos = msg.pose.pose.position

    # ── Helpers ───────────────────────────────────────────────────────────────

    def near_goal(self):
        if self.current_goal is None or self.current_pos is None:
            return False
        dx = self.current_goal.x - self.current_pos.x
        dy = self.current_goal.y - self.current_pos.y
        return math.sqrt(dx * dx + dy * dy) < GOAL_YIELD_RADIUS

    def get_perp_distance(self, msg, angle_min, angle_max):
        """
        Average perpendicular wall distance within an angular window.

        Each ray at angle θ with measured distance d gives:
            perp = d * sin(|θ|)

        This recovers the true lateral wall distance regardless of the
        scan angle, allowing forward/rear scaffold points to report the
        same perpendicular distance as the lateral point if the wall is straight.

        Returns None if fewer than MIN_POINTS valid readings, or if the
        computed perpendicular mean exceeds MAX_WALL_RANGE (open space).
        """
        perp_vals = []
        for i, d in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            if angle_min <= angle <= angle_max:
                if msg.range_min < d < min(msg.range_max, MAX_WALL_RANGE / abs(math.sin(angle)) + 0.1):
                    if not math.isnan(d) and not math.isinf(d):
                        perp = d * abs(math.sin(angle))
                        if perp < MAX_WALL_RANGE:
                            perp_vals.append(perp)
        if len(perp_vals) < MIN_POINTS:
            return None
        return np.mean(perp_vals)

    def measure_scaffolding(self, scan):
        """
        Returns two lists: left_readings, right_readings
        Each list has 3 entries (fwd, lat, rear) — None where wall not detected.
        """
        left  = [self.get_perp_distance(scan, a, b) for a, b in LEFT_WINDOWS]
        right = [self.get_perp_distance(scan, a, b) for a, b in RIGHT_WINDOWS]
        return left, right

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if self.latest_scan is None:
                rate.sleep()
                continue

            if self.near_goal():
                rospy.loginfo_throttle(2.0, "Near goal — yielding to move_base")
                rate.sleep()
                continue

            scan  = self.latest_scan
            left, right = self.measure_scaffolding(scan)

            twist = Twist()
            twist.linear.x = FORWARD_SPEED

            # Build weighted error from scaffold points where both sides valid
            errors, weights = [], []
            for i, (l, r, w) in enumerate(zip(left, right, SCAFFOLD_WEIGHTS)):
                if l is not None and r is not None:
                    errors.append(l - r)
                    weights.append(w)

            if errors:
                # Normal centering — weighted average across valid scaffold points
                error = float(np.average(errors, weights=weights))
                twist.angular.z = float(np.clip(-ANGULAR_GAIN * error,
                                                 -MAX_ANGULAR, MAX_ANGULAR))
                self.last_angular   = twist.angular.z
                self.last_seen_time = rospy.Time.now()

                # Detailed log: show each scaffold point
                l_str = '  '.join(
                    f"{LABELS[i]}-L={left[i]:.2f}" if left[i] else f"{LABELS[i]}-L=---"
                    for i in range(3))
                r_str = '  '.join(
                    f"{LABELS[i]}-R={right[i]:.2f}" if right[i] else f"{LABELS[i]}-R=---"
                    for i in range(3))
                rospy.loginfo_throttle(1.0,
                    f"{l_str}  |  {r_str}  |  err={error:.3f}  w={twist.angular.z:.3f}")

                # Junction warning: lateral gone but rear still present
                l_junction = left[1] is None and left[2] is not None
                r_junction = right[1] is None and right[2] is not None
                if l_junction or r_junction:
                    side = 'LEFT' if l_junction else 'RIGHT'
                    rospy.logwarn_throttle(1.0,
                        f"Junction detected on {side} — lateral gone, rear anchoring")

            else:
                # No paired readings at all — coast or go straight
                now = rospy.Time.now()
                coast_ok = (self.last_seen_time is not None and
                            (now - self.last_seen_time).to_sec() < COAST_DURATION)

                if coast_ok:
                    twist.angular.z = self.last_angular * 0.5
                    rospy.logwarn_throttle(1.0,
                        f"No walls — coasting  angular={twist.angular.z:.3f}")
                else:
                    twist.angular.z = 0.0
                    rospy.logwarn_throttle(2.0, "No walls — going straight")

            self.cmd_pub.publish(twist)
            rate.sleep()

        self.cmd_pub.publish(Twist())
        rospy.loginfo("row_follower_scaffold stopped.")


if __name__ == '__main__':
    try:
        RowFollowerScaffold().run()
    except rospy.ROSInterruptException:
        pass
