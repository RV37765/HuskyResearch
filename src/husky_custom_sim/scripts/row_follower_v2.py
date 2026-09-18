#!/usr/bin/env python3
"""
Row Follower v2 -- row_follower.py's proven centering law, gated by a
confidence-based state machine instead of a fixed coasting timer.

The steering math (get_side_distance, error = L - R, angular = -K*error) is
UNCHANGED from row_follower.py -- proven since Spring at +/-1-3cm lateral
accuracy. What's new is the question asked before that steering is trusted:
"is a row actually here, or am I chasing noise?" That question is answered by
RowConfidence (row_confidence.py) -- a PCA linearity check on a forward
LiDAR strip, smoothed over a rolling window.

STATE MACHINE
--------------
    ACQUIRE   -- cold start / after a reset. Creep forward, steer on
                 whatever the sensor gives, wait for a SUSTAINED high
                 confidence reading before trusting FOLLOWING.
    FOLLOWING -- normal centering at full speed. Confidence is holding.
    CREEPING  -- confidence just dropped below the gate. Slow to
                 CREEP_SPEED but keep centering -- don't stop, don't guess.
                 A plant gap is short: if confidence recovers, we were just
                 crossing a gap and go back to FOLLOWING. If it *doesn't*
                 recover for window_size consecutive low-confidence scans
                 (~1.5s at the VLP-16's 9.9Hz), the row has genuinely ended.
    ROW_END   -- stop. The turnaround sequence is a later phase (Week 3) --
                 today this state just stops the robot and logs it.

Why "creep instead of coast": the original coasting held the last angular
correction while still driving at full FORWARD_SPEED the whole time. A peer
running the same VLP-16 on similar row-crop hardware found that a plant gap
and a genuine row end look identical for the first several scans -- the only
way to tell them apart is to keep moving, slowly, and see whether the signal
comes back. See claude/action_plan.md, Week 2 (updated 2026-09-18).

TWO CONFIDENCE SIGNALS, TWO DIFFERENT JOBS:
  - rc.is_confident() -- the SMOOTHED value (mean over the last window_size
    scans). Gates "am I ready to trust FOLLOWING." Slow to rise, slow to
    fall -- exactly what you want for a readiness gate.
  - instant confidence (this tick's raw score, no smoothing) -- drives the
    CREEPING -> ROW_END streak counter. A row end needs window_size
    CONSECUTIVE bad raw scans, not just a smoothed average dipping low.
    A single good raw scan resets the streak to zero.

Run standalone (same pattern as row_follower.py and row_confidence.py):
  1. roslaunch husky_custom_sim husky_real_row_map.launch   (no move_base)
  2. rosrun husky_custom_sim row_follower_v2.py

RowConfidence loads its own params from the "~row_confidence" private
namespace -- same YAML (config/row_confidence.yaml), same mechanism already
validated by row_confidence.py's diagnostic node. No new config file needed.
Without a roslaunch <rosparam> tag, it falls back to its own DEFAULTS dict
(same numbers as the YAML) -- exactly how row_confidence.py was tested via
plain `rosrun` today.

SAFETY: unlike row_confidence.py (read-only, never touched /cmd_vel), this
node WILL drive the robot once confidence is reached. Have the PS4 dead-man's
switch (L1 + left stick) ready before running this on real hardware.
"""

import math
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry

from row_confidence import RowConfidence

# --- Tuning parameters (unchanged from row_follower.py) ---
FORWARD_SPEED   = 0.20      # m/s -- full-speed centering, once confident
ANGULAR_GAIN    = 1.2
MAX_ANGULAR     = 0.4
GOAL_YIELD_RADIUS = 1.5     # meters -- yield to move_base near an active goal

LEFT_ANGLE_MIN  = 0.2       # ~11 degrees
LEFT_ANGLE_MAX  = 1.5       # ~86 degrees
RIGHT_ANGLE_MIN = -1.5
RIGHT_ANGLE_MAX = -0.2

MAX_RANGE  = 5.0            # meters
MIN_POINTS = 5              # per side, for get_side_distance to trust it

# --- New for v2: state machine ---
CREEP_SPEED = 0.05          # m/s -- slow, cautious speed for ACQUIRE / CREEPING

STATE_ACQUIRE   = "ACQUIRE"
STATE_FOLLOWING = "FOLLOWING"
STATE_CREEPING  = "CREEPING"
STATE_ROW_END   = "ROW_END"


class RowFollowerV2:
    def __init__(self):
        rospy.init_node('row_follower_v2')

        self.cmd_pub  = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.scan_sub = rospy.Subscriber('/scan_fixed', LaserScan, self.scan_callback)
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        self.odom_sub = rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)

        self.latest_scan  = None
        self.last_angular  = 0.0
        self.current_goal  = None
        self.current_pos   = None

        self.rc = RowConfidence("~row_confidence")

        self.state = STATE_ACQUIRE
        self.low_conf_streak = 0   # consecutive low-confidence RAW scans, CREEPING only

        rospy.loginfo("Row follower v2 started. State: %s", self.state)
        rospy.loginfo(
            "Speed: %.2f m/s | Creep: %.2f m/s | Gain: %.1f | Confidence gate: %.2f",
            FORWARD_SPEED, CREEP_SPEED, ANGULAR_GAIN, self.rc.confidence_threshold)

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
        """Unchanged from row_follower.py -- the proven centering input."""
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

    def compute_steering(self, scan):
        """Same centering law as v1. Falls back to a damped last-known
        correction when a side is missing this tick -- identical fallback
        to v1's coasting, just no longer the ONLY thing gating row-end."""
        left_dist  = self.get_side_distance(scan, LEFT_ANGLE_MIN,  LEFT_ANGLE_MAX)
        right_dist = self.get_side_distance(scan, RIGHT_ANGLE_MIN, RIGHT_ANGLE_MAX)

        if left_dist is not None and right_dist is not None:
            error = left_dist - right_dist
            angular = -ANGULAR_GAIN * error
            angular = max(-MAX_ANGULAR, min(MAX_ANGULAR, angular))
            self.last_angular = angular
            rospy.loginfo_throttle(1.0,
                "[%s] L=%.2fm R=%.2fm err=%.3f angular=%.3f",
                self.state, left_dist, right_dist, error, angular)
            return angular

        return self.last_angular * 0.5

    def run(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if self.latest_scan is None:
                rate.sleep()
                continue

            if self.near_goal():
                rospy.loginfo_throttle(2.0,
                    "Near goal -- yielding to move_base (within %.1f m)", GOAL_YIELD_RADIUS)
                rate.sleep()
                continue

            scan = self.latest_scan

            # update() runs score(scan) internally, pushes it into RowConfidence's
            # own rolling window, and returns the smoothed value. The raw,
            # un-smoothed score from that same call is sitting in last_debug --
            # free to read, no extra computation.
            temporal_conf = self.rc.update(scan)
            instant_conf  = self.rc.last_debug.get("instant", 0.0)
            confident     = self.rc.is_confident()

            angular = self.compute_steering(scan)
            twist = Twist()

            if self.state == STATE_ACQUIRE:
                twist.linear.x  = CREEP_SPEED
                twist.angular.z = angular
                if confident:
                    rospy.loginfo("Confidence reached (%.2f) -- entering FOLLOWING", temporal_conf)
                    self.state = STATE_FOLLOWING

            elif self.state == STATE_FOLLOWING:
                twist.linear.x  = FORWARD_SPEED
                twist.angular.z = angular
                if not confident:
                    rospy.logwarn("Confidence dropped (%.2f) -- entering CREEPING", temporal_conf)
                    self.state = STATE_CREEPING
                    self.low_conf_streak = 0

            elif self.state == STATE_CREEPING:
                twist.linear.x  = CREEP_SPEED
                twist.angular.z = angular

                if confident:
                    rospy.loginfo("Confidence recovered -- back to FOLLOWING (was a gap)")
                    self.state = STATE_FOLLOWING
                    self.low_conf_streak = 0
                elif instant_conf < self.rc.confidence_threshold:
                    self.low_conf_streak += 1
                    rospy.loginfo_throttle(0.5,
                        "CREEPING: low-confidence streak %d/%d",
                        self.low_conf_streak, self.rc.window_size)
                    if self.low_conf_streak >= self.rc.window_size:
                        rospy.logwarn("Row end confirmed after %d consecutive low scans",
                                       self.low_conf_streak)
                        self.state = STATE_ROW_END
                else:
                    # This tick's raw score was fine even though the smoothed
                    # average hasn't caught up yet -- don't count it against
                    # the streak. A single good scan breaks a bad run.
                    self.low_conf_streak = 0

            elif self.state == STATE_ROW_END:
                twist.linear.x  = 0.0
                twist.angular.z = 0.0
                rospy.loginfo_throttle(2.0,
                    "ROW_END reached -- stopped. Turnaround not yet implemented (Week 3).")

            self.cmd_pub.publish(twist)
            rate.sleep()

        self.cmd_pub.publish(Twist())
        rospy.loginfo("Row follower v2 stopped.")


if __name__ == '__main__':
    try:
        follower = RowFollowerV2()
        follower.run()
    except rospy.ROSInterruptException:
        pass
