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
angular = -(k_lateral * cross_track + k_heading * heading_error), clipped to
+/-max_angular -- implemented in row_follower_state.steering_command(). Both
errors are "robot relative to the row" (see centerline_error), so both terms
are negative feedback. Gains are ROS params (~k_lateral, ~k_heading,
~max_angular), defaulting to the 2026-10-08 live-tuned values.

FRAMES
------
Points are transformed from the cloud's frame into ~target_frame (default
base_link) via TF before RowLineFit sees them -- the VLP-16 is mounted
upside-down, so raw sensor-frame "left" can be the robot's right. See
frame_utils.py. If the transform can't be looked up, this node does not drive.
That also means z_min/z_max are heights above base_link (~0.13 m above
ground), not distances relative to the sensor.

SAFETY: this node WILL drive the robot once centerline_error() starts
returning values. Have the PS4 dead-man's switch (L1 + left stick) ready
before running this on real hardware -- same rule as row_follower_v2.py.

Run it through the launch file, NOT plain rosrun:
    roslaunch husky_custom_sim row_follower_v3.launch env:=hallway

row_line_fit.py loads no ROS params itself (deliberately rospy-free). This
node reads them from its private ~row_line_fit namespace -- and plain
`rosrun` loads NOTHING into that namespace, silently falling back to the
code DEFAULTS (cotton values). That's how the 2026-10-08 hallway ROI fix got
dropped on 2026-10-22. The launch file loads config/row_line_fit.yaml plus an
environment overlay (row_line_fit_<env>.yaml) so that can't happen again.
"""

import math
import time

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool

from frame_utils import apply_rigid, cloud_xyz, describe, rigid_from_transform_msg
from odom_drift import YawDriftCorrector
from row_follower_state import (
    STATE_ACQUIRE, STATE_CREEPING, STATE_FOLLOWING, STATE_ROW_END, apply_baseline_gate, step_state, steering_command,
)
from row_line_fit import DEFAULTS, RowLineFit
import safety

# --- Tuning parameters ---
FORWARD_SPEED = 0.20    # m/s -- matches row_follower_v2's full-speed value
CREEP_SPEED = 0.05      # m/s -- matches row_follower_v2's creep value

# Steering gains are ROS params (~k_lateral, ~k_heading, ~max_angular) so they
# can be changed per run on the command line instead of sed-editing this file
# on the robot -- that's how the live-tuned values ended up uncommitted and
# easy to lose. These defaults ARE those live-tuned values (2026-10-08). They
# were chosen while the heading-sign bug (fixed 2026-10-28, see
# centerline_error) was still destabilizing the loop, so they're probably more
# conservative than the fixed controller needs -- retune upward deliberately,
# one run at a time. The closed-loop sim converges with both these and the
# old 1.2/1.5/0.4 defaults once the sign is fixed.
DEFAULT_K_LATERAL = 0.3
DEFAULT_K_HEADING = 0.3
DEFAULT_MAX_ANGULAR = 0.08


def _yaw_from_quaternion(q):
    """Standard quaternion -> yaw formula -- avoids adding a tf dependency
    for one number, same spirit as row_line_fit_node.py."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _load_overrides(ns="~row_line_fit"):
    return {key: rospy.get_param("%s/%s" % (ns, key), default) for key, default in DEFAULTS.items()}


class RowFollowerV3:
    def __init__(self):
        rospy.init_node("row_follower_v3")

        self.cloud_topic = rospy.get_param("~cloud_topic", "/velodyne_points")
        self.odom_topic = rospy.get_param("~odom_topic", "/odometry/filtered")
        self.k_lateral = float(rospy.get_param("~k_lateral", DEFAULT_K_LATERAL))
        self.k_heading = float(rospy.get_param("~k_heading", DEFAULT_K_HEADING))
        self.max_angular = float(rospy.get_param("~max_angular", DEFAULT_MAX_ANGULAR))
        self._proc_ms_ema = None

        # Points are moved into target_frame (REP-103: x fwd, y LEFT, z up)
        # before RowLineFit sees them -- see frame_utils.py for why. Empty
        # string = legacy raw-sensor-frame behavior; only for comparison runs.
        self.target_frame = rospy.get_param("~target_frame", "base_link")

        # Forward obstacle stop (safety.py): on by default. Calibrated on both
        # 2026-10-28 hallway runs (zero false triggers in 2,968 scans).
        self.obstacle_stop = bool(rospy.get_param("~obstacle_stop", True))
        self.obstacle_cfg = {k: rospy.get_param("~obstacle/%s" % k, v) for k, v in safety.DEFAULTS.items()}
        self.obstacle_gate = safety.ObstacleGate(
            self.obstacle_cfg["min_points"], self.obstacle_cfg["trigger_scans"], self.obstacle_cfg["clear_scans"])

        # Gyro drift correction (odom_drift.py). On by default since the 2026-11-06 same-day
        # A/B: fix off, the robot drove 2.8 cm right of center; fix on, 0.4 cm (wall fit).
        # The node holds still for up to startup_hold_sec so the drift is learned before
        # it drives. Pass drift_correct:=false to reproduce the uncorrected behavior.
        self.drift_correct = bool(rospy.get_param("~drift_correct", True))
        self.startup_hold_sec = float(rospy.get_param("~startup_hold_sec", 4.0))
        # After a gap in scans longer than this the buffered centroids are stale and the
        # odometry yaw has drifted meanwhile (2026-11-03: a 38.7 s LiDAR dropout, then a
        # 1 m cross-track spike): reset the detector and re-acquire. 0 disables.
        self.stall_reset_sec = float(rospy.get_param("~stall_reset_sec", 1.0))
        self._sensor_tf = None
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        # Everything the callbacks below touch must exist BEFORE the
        # subscribers are created -- a subscriber can start firing its
        # callback on a background thread as soon as it's constructed,
        # concurrently with the rest of this __init__ still running. See
        # the module-level note in row_follower_v3's history: this ordering
        # fixes a real (if narrow) startup race where an early message
        # crashed _on_cloud() on self.rlf before this line had run.
        self.pose = (0.0, 0.0, 0.0)
        self._have_odom = False
        self.last_angular = 0.0
        self.paused = False
        self.drift = YawDriftCorrector() if self.drift_correct else None
        self._first_odom_t = None
        self._last_cloud_t = None
        self._drift_warned = False

        self.rlf = RowLineFit(overrides=_load_overrides())
        self.state = STATE_ACQUIRE
        self.low_conf_streak = 0

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.cloud_sub = rospy.Subscriber(self.cloud_topic, PointCloud2, self._on_cloud, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self._on_odom, queue_size=1)
        self.pause_sub = rospy.Subscriber("~pause", Bool, self._on_pause, queue_size=1)

        rospy.loginfo("Row follower v3 started. State: %s", self.state)
        rospy.loginfo(
            "Speed: %.2f m/s | Creep: %.2f m/s | K_lateral: %.2f | K_heading: %.2f | max_angular: %.3f",
            FORWARD_SPEED, CREEP_SPEED, self.k_lateral, self.k_heading, self.max_angular,
        )
        rospy.loginfo(
            "row_line_fit: roi_y_abs_max=%.2f row_half_width=%.2f z=[%.2f, %.2f] cluster_mode=%s",
            self.rlf.roi_y_abs_max, self.rlf.row_half_width, self.rlf.z_min, self.rlf.z_max,
            self.rlf.cluster_mode,
        )
        rospy.loginfo("Obstacle stop: %s (box x=[%.1f, %.1f] |y|<=%.2f, min_points=%d)",
                      "ON" if self.obstacle_stop else "OFF", self.obstacle_cfg["x_min"],
                      self.obstacle_cfg["x_max"], self.obstacle_cfg["half_width"], self.obstacle_cfg["min_points"])
        rospy.loginfo("Pause: rostopic pub -1 %s std_msgs/Bool \"data: true\" (false resumes)",
                      rospy.resolve_name("~pause"))
        rospy.loginfo("Gyro drift correction: %s | detector reset after %.1f s without scans",
                      "ON (startup hold up to %.0f s)" % self.startup_hold_sec if self.drift_correct else "OFF",
                      self.stall_reset_sec)
        rospy.logwarn("SAFETY: this node drives the robot. Keep the PS4 dead-man's switch ready.")

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        yaw = _yaw_from_quaternion(msg.pose.pose.orientation)
        t = msg.header.stamp.to_sec()
        if self.drift is not None:
            self.pose = self.drift.update(t, p.x, p.y, yaw, msg.twist.twist.linear.x)
        else:
            self.pose = (p.x, p.y, yaw)
        if self._first_odom_t is None:
            self._first_odom_t = t
        self._have_odom = True

    def _on_pause(self, msg):
        """Hands-free hold, e.g. while taking tape-measure ground truth. The
        PS4 override only holds while L1 stays pressed, which leaves no hands
        for the tape. Same effect as the obstacle hold: motion is zeroed, the
        detector and state machine keep running. Not a safety device -- the
        dead-man's switch and the e-stop are."""
        self.paused = bool(msg.data)
        rospy.logwarn("row_follower_v3: %s", "PAUSED -- holding" if self.paused else "RESUMED")

    def compute_steering(self, err):
        """Thin wrapper over row_follower_state.steering_command() -- the law
        itself (and the 2026-10-22 decay fix) lives there so it's testable
        without ROS and shared with replay_bag_log.py."""
        angular, self.last_angular = steering_command(
            err, self.last_angular, self.k_lateral, self.k_heading, self.max_angular,
        )
        return angular

    def _on_cloud(self, msg):
        if not self._have_odom:
            rospy.logwarn_throttle(5.0, "row_follower_v3: waiting for first %s message", self.odom_topic)
            return

        stamp = msg.header.stamp.to_sec()
        if self._last_cloud_t is not None and 0.0 < self.stall_reset_sec < stamp - self._last_cloud_t:
            rospy.logwarn("row_follower_v3: %.1f s without scans -- detector reset, re-acquiring",
                          stamp - self._last_cloud_t)
            self.rlf.reset()
            self.state, self.low_conf_streak, self.last_angular = STATE_ACQUIRE, 0, 0.0
        self._last_cloud_t = stamp

        if self.target_frame and self._sensor_tf is None:
            try:
                tf_msg = self._tf_buffer.lookup_transform(
                    self.target_frame, msg.header.frame_id, rospy.Time(0), rospy.Duration(0.1))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as exc:
                # Fail safe: never steer on points whose frame we can't verify.
                # Not publishing lets twist_mux's 0.5s timeout stop the robot.
                rospy.logwarn_throttle(
                    2.0, "row_follower_v3: no TF %s <- %s yet (%s) -- not driving",
                    self.target_frame, msg.header.frame_id, exc)
                return
            self._sensor_tf = rigid_from_transform_msg(tf_msg.transform)
            rospy.loginfo("Sensor transform %s <- %s: %s",
                          self.target_frame, msg.header.frame_id, describe(*self._sensor_tf))

        t0 = time.monotonic()
        pts = cloud_xyz(msg)
        if self._sensor_tf is not None:
            pts = apply_rigid(pts, *self._sensor_tf)

        c = self.obstacle_cfg
        path_pts = safety.points_in_path(pts, c["x_min"], c["x_max"], c["half_width"], c["z_min"], c["z_max"])
        blocked = self.obstacle_gate.update(path_pts) and self.obstacle_stop

        dbg = self.rlf.update(pts, pose=self.pose)
        any_valid = self.rlf.any_valid()
        err, both_valid = apply_baseline_gate(
            self.rlf.centerline_error(self.pose), self.rlf.both_valid(), self.rlf.baseline_ok())

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

        if blocked:
            # Detector and state machine keep running; only motion is held.
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            rospy.logwarn_throttle(1.0, "OBSTACLE in path (%d points) -- holding", path_pts)

        if self.paused:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        if self.drift is not None and not self.drift.learned:
            if stamp - self._first_odom_t < self.startup_hold_sec:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                rospy.loginfo_throttle(1.0, "Holding still to learn the gyro drift...")
            elif not self._drift_warned:
                rospy.logwarn("row_follower_v3: gyro drift not learned (was the robot moving?) -- "
                              "running uncorrected until it next stops for 2 s")
                self._drift_warned = True

        if self.state == STATE_CREEPING:
            rospy.loginfo_throttle(
                0.5, "CREEPING: no valid fit, streak %d/%d", self.low_conf_streak, self.rlf.window_size,
            )

        # Per-scan budget at ~9.9Hz is ~101ms. If proc approaches that, the
        # Python point-cloud conversion (not RowLineFit) is the likely cost.
        proc_ms = (time.monotonic() - t0) * 1000.0
        self._proc_ms_ema = proc_ms if self._proc_ms_ema is None else 0.9 * self._proc_ms_ema + 0.1 * proc_ms

        err_str = "xt=%.3f hd=%.3f%s" % (
            err["cross_track"], err["heading_error"], " (est)" if err["estimated"] else "",
        ) if err is not None else "none"
        rospy.loginfo_throttle(
            1.0, "[%s] err=%s angular=%.3f | clusters=%s n_canopy=%d | path=%d%s%s | proc=%.0fms%s",
            self.state, err_str, angular, dbg["status"], dbg["n_canopy"], path_pts,
            " BLOCKED" if blocked else "", " PAUSED" if self.paused else "", self._proc_ms_ema,
            " | drift=%+.4f" % self.drift.rate if self.drift is not None else "",
        )

        self.cmd_pub.publish(twist)


if __name__ == "__main__":
    try:
        RowFollowerV3()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
