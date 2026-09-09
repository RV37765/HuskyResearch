#!/usr/bin/env python3
"""
row_confidence.py — "is there actually a row ahead?" confidence from LiDAR.

WHY THIS EXISTS
---------------
row_follower.py centers the robot by taking the MEAN range in a left wedge and a
right wedge (see row_follower.py get_side_distance). That works in a clean
hallway. On real cotton rows the returns are sparse and gappy: miss a few plants
and the wedge fills with points from the next row over or the tree line past the
headland, the mean lurches, and the robot chases noise. Worse, a plain mean
gives no signal about whether a row is even there — five points off a fence post
look just like five points off a real row edge.

This module answers a different question: do the points in a forward-facing strip
form a LINE (a row) or a BLOB (open ground, clutter, nothing)? It does that with
PCA on the point cluster:

    - Build the 2x2 covariance of the (x, y) points in the strip.
    - Its two eigenvalues are the variance along the cluster's long axis
      (lambda_max) and its short axis (lambda_min).
    - linearity = 1 - lambda_min / lambda_max
        ~1.0  -> spread is almost entirely along one axis  -> a row
        ~0.0  -> spread is equal in all directions         -> a blob

That per-scan number is noisy, so we push it through a rolling window of the last
N scans and report the smoothed value as the confidence. row_follower_v2 will
require a sustained high confidence (default 0.80, cross-checked by a peer
running the same VLP-16) before it trusts the row estimate and enters FOLLOW.

WHAT IT IS
----------
A plain class, RowConfidence, that any node can import and feed scans to. Running
this file directly also starts a small diagnostic node that republishes the
confidence on /row_confidence/confidence so you can watch it in rqt_plot or
Foxglove before wiring it into a controller.

GOTCHA — ROI aspect ratio
-------------------------
A long, thin ROI strip (default 3.7 m deep x 0.70 m half-width) is itself
elongated, so even diffuse / random returns inside it score a moderate linearity
(~0.6 in testing) rather than ~0. Clean rows score ~0.98, so the 0.80 FOLLOW
gate still separates them, but don't read a mid-range score as "half a row" —
it's mostly the strip's own shape. Shrinking roi_x_max tightens this. Tune the
ROI and confidence_threshold together in sim before trusting them on the robot.

PARAMETERS
----------
Every tunable is loaded with rospy.get_param() from a private sub-namespace
(default "~row_confidence"), so config/row_confidence.yaml can be loaded with
    <rosparam file=".../row_confidence.yaml" command="load" ns="row_confidence"/>
inside whatever node uses it, with no risk of colliding with that node's own
parameters. See config/row_confidence.yaml for the full list and defaults.

Run standalone:
    rosrun husky_custom_sim row_confidence.py
    rosrun husky_custom_sim row_confidence.py _scan_topic:=/scan_fixed
    rqt_plot /row_confidence/confidence /row_confidence/confidence_instant
"""

from collections import deque

import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32


# --- Parameter defaults -------------------------------------------------------
# These mirror config/row_confidence.yaml. They live here too so the class is
# usable without a param server (offline tests, unit checks) and so the file is
# self-documenting.
DEFAULTS = {
    # Forward-facing ROI strip, in the sensor frame (x = forward, y = left).
    "roi_x_min": 0.3,          # m — ignore points right on top of the robot
    "roi_x_max": 4.0,          # m — far edge of the strip we analyse
    # half-width. Cotton row spacing ~0.9 m -> the two bounding rows sit ~0.45 m
    # off center; 0.70 keeps them in and excludes the next rows out (~1.35 m).
    "roi_y_abs_max": 0.70,     # m

    # Height band, ONLY applied on the score_xyz() path (LaserScan has no z).
    # Crops grow through the season, so these must be retunable without a code
    # change. Defaults match the crop pointcloud_to_laserscan slice in the
    # SLAM/nav launches.
    "z_min": 0.05,             # m
    "z_max": 0.70,             # m

    "max_range": 5.0,          # m — matches row_follower.MAX_RANGE

    # How many points we need before PCA is meaningful.
    "min_points": 12,          # used when side_split is False
    "min_points_per_side": 6,  # used when side_split is True

    # Split the strip into a left cluster and a right cluster and score each,
    # instead of scoring one merged cluster. More robust when one row is sparse.
    "side_split": True,
    "side_combine": "mean",    # "mean" (average the two) or "min" (stricter)
    "one_side_penalty": 0.7,   # multiplier when only one side has enough points

    # Temporal smoothing.
    "window_size": 10,         # scans (~1 s at the VLP-16's ~9.9 Hz)
    "require_full_window": True,   # report 0.0 until the window has filled
    "temporal_mode": "mean",      # "mean" of the window, or "fraction" ...
    "per_frame_threshold": 0.6,   # ... of scans at/above this (fraction mode)

    "confidence_threshold": 0.80,  # is_confident() gate — the FOLLOW threshold
}

_EPS = 1e-12


class RowConfidence:
    """Rolling PCA-linearity confidence that a crop row is present ahead.

    Typical use inside a controller::

        rc = RowConfidence("~row_confidence")
        ...
        def on_scan(msg):
            conf = rc.update(msg)            # smoothed confidence in [0, 1]
            if rc.is_confident():
                ...                          # safe to enter / stay in FOLLOW

    ``score(scan)`` gives the raw per-scan value if you want it without touching
    the rolling window. ``reset()`` clears the window when you leave FOLLOW so a
    stale history can't wave you straight back in.
    """

    def __init__(self, param_ns="~row_confidence", overrides=None):
        """Load parameters.

        param_ns:  private namespace to read params from, e.g. "~row_confidence".
        overrides: optional dict of {key: value}. When given, values come from
                   here instead of the param server — used by offline tests so
                   the class runs with no ROS master.
        """
        self.param_ns = param_ns.rstrip("/")
        self._overrides = overrides

        self.roi_x_min = float(self._get("roi_x_min"))
        self.roi_x_max = float(self._get("roi_x_max"))
        self.roi_y_abs_max = float(self._get("roi_y_abs_max"))

        self.z_min = float(self._get("z_min"))
        self.z_max = float(self._get("z_max"))

        self.max_range = float(self._get("max_range"))

        self.min_points = int(self._get("min_points"))
        self.min_points_per_side = int(self._get("min_points_per_side"))

        self.side_split = bool(self._get("side_split"))
        self.side_combine = str(self._get("side_combine")).lower()
        self.one_side_penalty = float(self._get("one_side_penalty"))

        self.window_size = max(1, int(self._get("window_size")))
        self.require_full_window = bool(self._get("require_full_window"))
        self.temporal_mode = str(self._get("temporal_mode")).lower()
        self.per_frame_threshold = float(self._get("per_frame_threshold"))

        self.confidence_threshold = float(self._get("confidence_threshold"))

        if self.side_combine not in ("mean", "min"):
            self._warn("side_combine=%r not recognised, using 'mean'"
                       % self.side_combine)
            self.side_combine = "mean"
        if self.temporal_mode not in ("mean", "fraction"):
            self._warn("temporal_mode=%r not recognised, using 'mean'"
                       % self.temporal_mode)
            self.temporal_mode = "mean"

        self._window = deque(maxlen=self.window_size)
        # Filled in by every score() call; handy for logging / debugging.
        self.last_debug = {}

    # -- parameter / logging helpers (kept ROS-master-optional) --------------

    def _get(self, key):
        default = DEFAULTS[key]
        if self._overrides is not None:
            return self._overrides.get(key, default)
        return rospy.get_param("%s/%s" % (self.param_ns, key), default)

    def _warn(self, msg):
        try:
            rospy.logwarn("RowConfidence: %s", msg)
        except Exception:
            print("RowConfidence WARN: %s" % msg)

    # -- public API --------------------------------------------------------

    def score(self, scan):
        """Instantaneous linearity confidence in [0, 1] for one LaserScan.

        Returns 0.0 when the strip is too sparse to judge.
        """
        pts_xy = self._scan_to_xy(scan)
        return self.score_xy(pts_xy)

    def score_xyz(self, pts_xyz):
        """Instantaneous confidence for an (N, 3) array of points.

        Applies the z_min / z_max height band, drops z, then scores the (x, y).
        This is the entry point for a future BEV projection of /velodyne_points
        in row_follower_v2 — the one place a real per-point height exists.
        """
        pts = np.asarray(pts_xyz, dtype=float)
        if pts.ndim != 2 or pts.shape[1] < 3 or pts.shape[0] == 0:
            return self.score_xy(np.empty((0, 2)))
        finite = np.all(np.isfinite(pts[:, :3]), axis=1)
        band = (pts[:, 2] >= self.z_min) & (pts[:, 2] <= self.z_max)
        keep = finite & band
        return self.score_xy(pts[keep, :2])

    def score_xy(self, pts_xy):
        """Instantaneous confidence for an (N, 2) array of sensor-frame points.

        x = forward, y = left (REP-103). Applies the forward ROI strip mask,
        then PCA linearity (optionally split left/right). Fills ``last_debug``.
        """
        pts = np.asarray(pts_xy, dtype=float).reshape(-1, 2)
        if pts.shape[0]:
            pts = pts[np.all(np.isfinite(pts), axis=1)]

        in_roi = (
            (pts[:, 0] >= self.roi_x_min)
            & (pts[:, 0] <= self.roi_x_max)
            & (np.abs(pts[:, 1]) <= self.roi_y_abs_max)
        ) if pts.shape[0] else np.zeros(0, dtype=bool)
        roi = pts[in_roi]

        debug = {
            "n_in": int(pts.shape[0]),
            "n_roi": int(roi.shape[0]),
            "side_split": self.side_split,
        }

        if self.side_split:
            left = roi[roi[:, 1] > 0.0]
            right = roi[roi[:, 1] < 0.0]
            lin_l, ev_l = self._linearity(left)
            lin_r, ev_r = self._linearity(right)
            ok_l = left.shape[0] >= self.min_points_per_side
            ok_r = right.shape[0] >= self.min_points_per_side

            if ok_l and ok_r:
                if self.side_combine == "min":
                    conf = min(lin_l, lin_r)
                else:
                    conf = 0.5 * (lin_l + lin_r)
            elif ok_l:
                conf = lin_l * self.one_side_penalty
            elif ok_r:
                conf = lin_r * self.one_side_penalty
            else:
                conf = 0.0

            debug.update({
                "n_left": int(left.shape[0]), "n_right": int(right.shape[0]),
                "lin_left": lin_l if ok_l else None,
                "lin_right": lin_r if ok_r else None,
                "eig_left": ev_l, "eig_right": ev_r,
            })
        else:
            if roi.shape[0] >= self.min_points:
                conf, ev = self._linearity(roi)
            else:
                conf, ev = 0.0, (0.0, 0.0)
            debug.update({"lin": conf, "eig": ev})

        conf = float(np.clip(conf, 0.0, 1.0))
        debug["instant"] = conf
        self.last_debug = debug
        return conf

    def update(self, scan):
        """score() the scan, push it into the rolling window, return the
        smoothed temporal confidence."""
        self._window.append(self.score(scan))
        conf = self.temporal_confidence()
        self.last_debug["temporal"] = conf
        return conf

    def push(self, instant_score):
        """Feed an already-computed instantaneous score into the window.

        For callers that run score_xy / score_xyz themselves (e.g. a BEV path)
        and still want the temporal smoothing."""
        self._window.append(float(np.clip(instant_score, 0.0, 1.0)))
        return self.temporal_confidence()

    def temporal_confidence(self):
        """Smoothed confidence over the rolling window, in [0, 1].

        0.0 until the window has filled when ``require_full_window`` is set — we
        don't want a controller entering FOLLOW off two lucky scans.
        """
        n = len(self._window)
        if n == 0:
            return 0.0
        if self.require_full_window and n < self.window_size:
            return 0.0
        vals = np.fromiter(self._window, dtype=float, count=n)
        if self.temporal_mode == "fraction":
            return float(np.mean(vals >= self.per_frame_threshold))
        return float(np.mean(vals))

    def is_confident(self):
        """True when the smoothed confidence is at/above the FOLLOW threshold."""
        return self.temporal_confidence() >= self.confidence_threshold

    def reset(self):
        """Drop the rolling-window history (call when leaving FOLLOW)."""
        self._window.clear()

    @property
    def window_fill(self):
        """(current, capacity) of the rolling window — for diagnostics."""
        return len(self._window), self.window_size

    # -- internals -------------------------------------------------------

    def _scan_to_xy(self, scan):
        """LaserScan -> (N, 2) sensor-frame points, range-filtered.

        Same validity gate as row_follower.get_side_distance (range_min < r <
        min(range_max, max_range), no nan/inf), just vectorised.
        """
        ranges = np.asarray(scan.ranges, dtype=float)
        n = ranges.size
        if n == 0:
            return np.empty((0, 2))

        angles = scan.angle_min + np.arange(n) * scan.angle_increment
        far = min(scan.range_max, self.max_range)
        valid = (
            np.isfinite(ranges)
            & (ranges > scan.range_min)
            & (ranges < far)
        )
        r = ranges[valid]
        a = angles[valid]
        # REP-103: x forward, y left. LaserScan angle 0 = forward, +ve = CCW.
        return np.column_stack((r * np.cos(a), r * np.sin(a)))

    def _linearity(self, pts):
        """PCA linearity of an (N, 2) cluster.

        Returns (linearity in [0, 1], (lambda_min, lambda_max)).
        linearity = 1 - lambda_min / lambda_max: 1.0 for a perfect line,
        0.0 for an isotropic blob. 0.0 when there are too few points or the
        cluster has no spread.
        """
        if pts.shape[0] < 2:
            return 0.0, (0.0, 0.0)
        cov = np.cov(pts.T)  # 2x2
        if not np.all(np.isfinite(cov)):
            return 0.0, (0.0, 0.0)
        w = np.linalg.eigvalsh(cov)  # ascending, real (cov is symmetric)
        l_min = float(max(w[0], 0.0))
        l_max = float(max(w[-1], 0.0))
        if l_max <= _EPS:
            return 0.0, (l_min, l_max)
        lin = 1.0 - (l_min / l_max)
        return float(np.clip(lin, 0.0, 1.0)), (l_min, l_max)


# --- Diagnostic node --------------------------------------------------------

class _RowConfidenceNode:
    """Thin ROS wrapper: subscribe to a scan, publish the confidence.

    Publishes:
        ~confidence          (std_msgs/Float32)  smoothed / temporal
        ~confidence_instant  (std_msgs/Float32)  raw per-scan
    """

    def __init__(self):
        rospy.init_node("row_confidence")
        self.rc = RowConfidence("~row_confidence")

        scan_topic = rospy.get_param("~scan_topic", "/scan_fixed")
        self.pub = rospy.Publisher("~confidence", Float32, queue_size=1)
        self.pub_instant = rospy.Publisher(
            "~confidence_instant", Float32, queue_size=1)
        self.sub = rospy.Subscriber(
            scan_topic, LaserScan, self._on_scan, queue_size=1)

        fill, cap = self.rc.window_fill
        rospy.loginfo("row_confidence up. scan_topic=%s", scan_topic)
        rospy.loginfo(
            "ROI x[%.2f, %.2f] |y|<=%.2f  window=%d  threshold=%.2f  "
            "side_split=%s(%s)",
            self.rc.roi_x_min, self.rc.roi_x_max, self.rc.roi_y_abs_max, cap,
            self.rc.confidence_threshold, self.rc.side_split,
            self.rc.side_combine)

    def _on_scan(self, msg):
        # PCA on a few hundred points is microseconds — safe to do inline,
        # no work is deferred and there is no sleep in this callback.
        temporal = self.rc.update(msg)
        instant = self.rc.last_debug.get("instant", 0.0)
        self.pub.publish(Float32(data=temporal))
        self.pub_instant.publish(Float32(data=instant))

        d = self.rc.last_debug
        fill, cap = self.rc.window_fill
        rospy.loginfo_throttle(
            1.0,
            "conf=%.2f (inst=%.2f)  roi=%d/%d pts  L=%s R=%s  win=%d/%d  %s",
            temporal, instant, d.get("n_roi", 0), d.get("n_in", 0),
            _fmt(d.get("lin_left")), _fmt(d.get("lin_right")),
            fill, cap, "FOLLOW-ok" if self.rc.is_confident() else "hold")


def _fmt(v):
    return "--" if v is None else ("%.2f" % v)


if __name__ == "__main__":
    try:
        _RowConfidenceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
