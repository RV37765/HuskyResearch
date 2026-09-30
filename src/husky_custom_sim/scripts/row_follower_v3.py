#!/usr/bin/env python3
"""
Row Follower v3 -- row_line_fit.py's geometric centerline, in the same
confidence-gated state machine shape as row_follower_v2.py.

WHY THIS EXISTS
---------------
row_follower_v2.py steers on L_avg - R_avg (row_confidence.py's PCA gate
decides WHEN to trust it) and drives off /scan_fixed. This is the geometric
successor: row_line_fit.py fits an explicit line through each row edge from
raw 3D points, and centerline_error() gives a true cross-track + heading
error against the fitted centerline instead of a mean-distance proxy. See
claude/action_plan.md, "Architecture Update" (2026-09-24) for why this is
expected to be the bigger accuracy win, not just a row-end-detection fix.

WHY HALLWAY TESTING FIRST, EVEN THOUGH THE ALGORITHM TARGETS COTTON ROWS
--------------------------------------------------------------------------
Ryan's framing (2026-09-30): a hallway is a strictly EASIER case than sparse
cotton plants -- dense, continuous wall returns instead of sparse, gappy
plant-top clumps. If clustering + centroid + RANSAC can't reliably find a
line and detect row-end in a hallway, it has no chance in a real field. So:
develop and drive this with best-guess parameters (no field data exists yet
to tune against) and get it working end-to-end on hallway walls first --
that clears a real, necessary bar, just not a sufficient one.

ONE HONEST CAVEAT: dense, continuous wall points don't exercise everything.
The k-means "one row split into two fake rows" bug fixed 2026-09-28 (see
row_line_fit.py's GOTCHA comments) came from SPARSE, gappy points -- a solid
wall is nowhere near ambiguous enough to trigger it. A clean hallway pass
proves the state machine, the ROS plumbing, and the RANSAC fit/steering math
all work end-to-end. It does NOT prove the canopy-height slice or the
clustering step will behave the same way on real, sparse plant returns.

STATE MACHINE (identical shape to row_follower_v2.py)
--------------------------------------------------------
    ACQUIRE   -- cold start. Creep, steer on whatever centerline_error()
                 gives (even a one-sided estimate), wait for both_valid()
                 before trusting FOLLOWING.
    FOLLOWING -- full speed, steer on cross_track + heading_error.
    CREEPING  -- any_valid() just went False (BOTH sides lost, not just
                 one). Slow down, keep steering on whatever's left. Seeing
                 EITHER side again -- even a one-sided estimate -- means it
                 was a gap, not a row end: reset the streak immediately.
                 Only a fully-blind scan (neither side seen) counts toward
                 the row-end streak -- matches RowLineFit.any_valid()'s own
                 documented contract, same reasoning row_follower_v2.py
                 applied to row_confidence.py's instant_conf.
    ROW_END   -- stop. Turnaround is a separate, later phase.

The transition logic itself lives in step_state() below as a plain function
with no ROS or robot dependency, specifically so it can be exercised offline
against synthetic RowLineFit output before ever running on hardware -- see
test_row_follower_v3.py.

STEERING
--------
angular = -(K_LATERAL * cross_track + K_HEADING * heading_error), clipped.
Both gains are BEST-GUESS starting points -- no field data exists yet to
tune them against. K_LATERAL matches row_follower_v2's ANGULAR_GAIN since
cross_track is the same kind of meters-scale error; K_HEADING is a fresh
guess for how hard to correct a given heading error. Expect to retune both
from the very first hallway run.

SAFETY: this node WILL drive the robot once centerline_error() starts
returning values. Have the PS4 dead-man's switch (L1 + left stick) ready
before running this on real hardware -- same rule as row_follower_v2.py.

Run standalone:
    roslaunch husky_custom_sim husky_real_row_map.launch   (no move_base)
    rosrun husky_custom_sim row_follower_v3.py

row_line_fit.py loads no ROS params itself (deliberately rospy-free -- see
its module docstring); this node reads config/row_line_fit.yaml under the
"row_line_fit" namespace the same way row_line_fit_node.py does.
"""

import math

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2

from row_follower_state import (
    STATE_ACQUIRE, STATE_CREEPING, STATE_FOLLOWING, STATE_ROW_END, step_state,
)
from row_line_fit import DEFAULTS, RowLineFit

# --- Tuning parameters ---
FORWARD_SPEED = 0.20    # m/s -- matches row_follower_v2's full-speed value
CREEP_SPEED = 0.05      # m/s -- matches row_follower_v2's creep value
K_LATERAL = 1.2         # best guess: matches row_follower_v2's ANGULAR_GAIN (same units, meters of error)
K_HEADING = 1.5         # best guess -- no field data yet to tune against
MAX_ANGULAR = 0.4


def _yaw_from_quaternion(q):
    """Standard quaternion -> yaw formula -- avoids adding a tf dependency
    for one number, same spirit as row_line_fit_node.py."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _load_overrides(ns="~row_line_fit"):
    return {key: rospy.get_param("%s/%s" % (ns, key), default) for key, default in DEFAULTS.items()}


class RowFollowerV3:
    def __init__(self):
        rospy.init_node("row_follower_v3")

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.cloud_topic = rospy.get_param("~cloud_topic", "/velodyne_points")
        self.odom_topic = rospy.get_param("~odom_topic", "/odometry/filtered")

        self.cloud_sub = rospy.Subscriber(self.cloud_topic, PointCloud2, self._on_cloud, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self._on_odom, queue_size=1)

        self.pose = (0.0, 0.0, 0.0)
        self._have_odom = False
        self.last_angular = 0.0

        self.rlf = RowLineFit(overrides=_load_overrides())
        self.state = STATE_ACQUIRE
        self.low_conf_streak = 0

        rospy.loginfo("Row follower v3 started. State: %s", self.state)
        rospy.loginfo(
            "Speed: %.2f m/s | Creep: %.2f m/s | K_lateral: %.2f | K_heading: %.2f (best-guess, untuned)",
            FORWARD_SPEED, CREEP_SPEED, K_LATERAL, K_HEADING,
        )
        rospy.logwarn("SAFETY: this node drives the robot. Keep the PS4 dead-man's switch ready.")

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        self.pose = (p.x, p.y, _yaw_from_quaternion(msg.pose.pose.orientation))
        self._have_odom = True

    def compute_steering(self, err):
        """err is centerline_error()'s dict, or None. Falls back to a
        damped last-known correction when nothing's available -- identical
        fallback pattern to row_follower.py's coasting / v2's compute_steering."""
        if err is None:
            return self.last_angular * 0.5
        angular = -(K_LATERAL * err["cross_track"] + K_HEADING * err["heading_error"])
        angular = max(-MAX_ANGULAR, min(MAX_ANGULAR, angular))
        self.last_angular = angular
        return angular

    def _on_cloud(self, msg):
        if not self._have_odom:
            rospy.logwarn_throttle(5.0, "row_follower_v3: waiting for first %s message", self.odom_topic)
            return

        raw = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        pts = np.array(raw, dtype=float) if raw else np.empty((0, 3))

        self.rlf.update(pts, pose=self.pose)
        any_valid = self.rlf.any_valid()
        both_valid = self.rlf.both_valid()
        err = self.rlf.centerline_error(self.pose)

        angular = self.compute_steering(err)
        prev_state = self.state
        self.state, self.low_conf_streak = step_state(
            self.state, any_valid, both_valid, self.low_conf_streak, self.rlf.window_size,
        )
        if self.state != prev_state:
            rospy.loginfo("%s -> %s", prev_state, self.state)

        twist = Twist()
        if self.state == STATE_ACQUIRE or self.state == STATE_CREEPING:
            twist.linear.x = CREEP_SPEED
            twist.angular.z = angular
        elif self.state == STATE_FOLLOWING:
            twist.linear.x = FORWARD_SPEED
            twist.angular.z = angular
        else:  # STATE_ROW_END
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            rospy.loginfo_throttle(2.0, "ROW_END reached -- stopped. Turnaround not yet implemented.")

        if self.state == STATE_CREEPING:
            rospy.loginfo_throttle(
                0.5, "CREEPING: no valid fit, streak %d/%d", self.low_conf_streak, self.rlf.window_size,
            )

        err_str = "xt=%.3f hd=%.3f%s" % (
            err["cross_track"], err["heading_error"], " (est)" if err["estimated"] else "",
        ) if err is not None else "none"
        rospy.loginfo_throttle(1.0, "[%s] err=%s angular=%.3f", self.state, err_str, angular)

        self.cmd_pub.publish(twist)


if __name__ == "__main__":
    try:
        RowFollowerV3()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
