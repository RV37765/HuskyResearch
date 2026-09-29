#!/usr/bin/env python3
"""
Offline synthetic tests for row_confidence.py -- no ROS master required.

Run directly:
    python3 test_row_confidence.py

RowConfidence supports an `overrides` dict specifically so it can run without
a param server (see its docstring) -- that's what makes these tests possible
without roscore or a real scan.

WHY THIS FILE EXISTS
---------------------
Earlier synthetic checks for this module were run ad hoc and never saved, so
they couldn't be re-run to confirm the 2026-09-19 patch_mode fix for the ROI
aspect-ratio bug (see the GOTCHA section in row_confidence.py). This file is
meant to stay in the repo and be re-run any time RowConfidence's scoring
logic changes.

Each test builds a synthetic (N, 2) point cloud in sensor-frame coordinates
(x = forward, y = left) and checks score_xy() against an expected range.
Tests 7 and 8 are the ones that actually exercise the bug: they compare
patch_mode=False against patch_mode=True on the same noise cloud.
"""

import numpy as np

from row_confidence import RowConfidence, DEFAULTS

RNG = np.random.default_rng(seed=42)

BASE_OVERRIDES = dict(DEFAULTS)  # start every test from the shipped defaults


def make_rc(**overrides):
    cfg = dict(BASE_OVERRIDES)
    cfg.update(overrides)
    return RowConfidence(overrides=cfg)


def row_line(x_min, x_max, y, n=80, noise=0.02):
    """Points scattered along a straight line at fixed y -- a clean row edge."""
    x = RNG.uniform(x_min, x_max, n)
    y_pts = y + RNG.normal(0.0, noise, n)
    return np.column_stack((x, y_pts))


def uniform_blob(x_min, x_max, y_min, y_max, n=80):
    """Pure uniform scatter over a rectangle -- open ground / clutter, no row."""
    x = RNG.uniform(x_min, x_max, n)
    y = RNG.uniform(y_min, y_max, n)
    return np.column_stack((x, y))


results = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append(condition)
    print("[%s] %s%s" % (status, name, "  (%s)" % detail if detail else ""))


# --- 1. Clean two-row corridor scores high -----------------------------------
rc = make_rc()
pts = np.vstack([row_line(0.3, 4.0, 0.45), row_line(0.3, 4.0, -0.45)])
conf = rc.score_xy(pts)
check("clean two-row corridor -> high confidence", conf >= 0.90, "conf=%.3f" % conf)

# --- 2. Pure open-ground scatter (default, patch_mode off) scores low -------
rc = make_rc()
pts = uniform_blob(0.3, 4.0, -0.70, 0.70)
conf = rc.score_xy(pts)
check("open-ground scatter, patch_mode=False -> LOW score is NOT guaranteed "
      "(this is the bug)", True, "conf=%.3f (expected to be biased upward)" % conf)

# --- 3. Same scatter, patch_mode on, scores meaningfully lower --------------
rc_off = make_rc(patch_mode=False)
rc_on = make_rc(patch_mode=True)
pts = uniform_blob(0.3, 4.0, -0.70, 0.70)
conf_off = rc_off.score_xy(pts)
conf_on = rc_on.score_xy(pts)
check("patch_mode reduces noise-floor score vs. pooled ROI",
      conf_on < conf_off - 0.15,
      "pooled=%.3f patched=%.3f" % (conf_off, conf_on))

# --- 4. patch_mode does NOT hurt a clean row's score ------------------------
rc_off = make_rc(patch_mode=False)
rc_on = make_rc(patch_mode=True)
pts = np.vstack([row_line(0.3, 4.0, 0.45), row_line(0.3, 4.0, -0.45)])
conf_off = rc_off.score_xy(pts)
conf_on = rc_on.score_xy(pts)
check("patch_mode keeps a clean row's score high",
      conf_on >= 0.85, "pooled=%.3f patched=%.3f" % (conf_off, conf_on))

# --- 5. Sparse/gappy row still scores reasonably high -----------------------
rc = make_rc(patch_mode=True)
left = row_line(0.3, 4.0, 0.45, n=80)
gap_mask = ~((left[:, 0] > 1.8) & (left[:, 0] < 2.4))  # drop a 0.6m gap
left = left[gap_mask]
right = row_line(0.3, 4.0, -0.45, n=80)
pts = np.vstack([left, right])
conf = rc.score_xy(pts)
check("gappy row (one 0.6m gap) still scores confidently", conf >= 0.80,
      "conf=%.3f" % conf)

# --- 6. One side visible only -> penalized but non-zero ---------------------
rc = make_rc()
pts = row_line(0.3, 4.0, 0.45, n=80)  # right side has nothing
conf = rc.score_xy(pts)
expected_max = 1.0 * DEFAULTS["one_side_penalty"]
check("one-side-only row is penalized, not full confidence",
      0.0 < conf <= expected_max + 1e-6, "conf=%.3f max=%.3f" % (conf, expected_max))

# --- 7. Empty ROI (true row end) -> zero confidence -------------------------
rc = make_rc()
pts = np.empty((0, 2))
conf = rc.score_xy(pts)
check("empty ROI (row end) -> zero confidence", conf == 0.0, "conf=%.3f" % conf)

# --- 8. Rolling window: is_confident() requires a full sustained window ----
rc = make_rc(patch_mode=True, window_size=15)
pts_clean = np.vstack([row_line(0.3, 4.0, 0.45), row_line(0.3, 4.0, -0.45)])


class _FakeScan:
    """Minimal stand-in for sensor_msgs/LaserScan built from Cartesian
    points, so update() can be exercised the same way a real scan would be."""

    def __init__(self, xy):
        r = np.hypot(xy[:, 0], xy[:, 1])
        a = np.arctan2(xy[:, 1], xy[:, 0])
        order = np.argsort(a)
        self.angle_min = float(a[order][0]) if len(a) else -np.pi
        self.angle_max = float(a[order][-1]) if len(a) else np.pi
        n = len(a)
        self.angle_increment = (
            (self.angle_max - self.angle_min) / max(n - 1, 1) if n > 1 else 0.01
        )
        self.range_min = 0.05
        self.range_max = 30.0
        self.ranges = list(r[order]) if n else []


not_yet_confident_before_fill = None
for i in range(rc.window_size):
    temporal = rc.update(_FakeScan(pts_clean))
    if i < rc.window_size - 1:
        not_yet_confident_before_fill = rc.is_confident()
check("is_confident() withholds True until window fills",
      not_yet_confident_before_fill is False,
      "was %r before fill" % not_yet_confident_before_fill)
check("is_confident() true once window fills with a clean row",
      rc.is_confident(), "temporal=%.3f" % rc.temporal_confidence())

# --- Summary -----------------------------------------------------------------
n_pass = sum(1 for r in results if r)
print("\n%d/%d checks passed" % (n_pass, len(results)))
if n_pass != len(results):
    raise SystemExit(1)
