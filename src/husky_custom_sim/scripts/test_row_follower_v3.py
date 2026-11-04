#!/usr/bin/env python3
"""
Offline tests for row_follower_v3.py -- no ROS, no robot required.

Run directly:
    python3 test_row_follower_v3.py

Two layers, same spirit as test_row_line_fit.py:
  1. step_state() unit tests -- the pure state-transition function, checked
     against every transition directly, no RowLineFit involved.
  2. An integrated dry run -- a REAL RowLineFit instance fed a synthetic
     "hallway then it ends" scan sequence, driving step_state() the same way
     row_follower_v3.py's _on_cloud() does, checking the full state sequence
     ACQUIRE -> FOLLOWING -> CREEPING -> ROW_END actually happens before this
     ever touches a real sensor.
"""

import math

import numpy as np

from row_follower_state import (
    apply_baseline_gate, steering_command,
    STATE_ACQUIRE, STATE_CREEPING, STATE_FOLLOWING, STATE_ROW_END, step_state,
)
from row_line_fit import RowLineFit

results = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append(condition)
    print("[%s] %s%s" % (status, name, "  (%s)" % detail if detail else ""))


# ============================================================
# Layer 1: step_state() unit tests
# ============================================================

# --- ACQUIRE ---
check("ACQUIRE + both_valid -> FOLLOWING",
      step_state(STATE_ACQUIRE, True, True, 0, 15) == (STATE_FOLLOWING, 0))
check("ACQUIRE + one-side-only stays in ACQUIRE",
      step_state(STATE_ACQUIRE, True, False, 0, 15) == (STATE_ACQUIRE, 0))
check("ACQUIRE + nothing stays in ACQUIRE",
      step_state(STATE_ACQUIRE, False, False, 0, 15) == (STATE_ACQUIRE, 0))

# --- FOLLOWING ---
check("FOLLOWING + both_valid stays in FOLLOWING",
      step_state(STATE_FOLLOWING, True, True, 0, 15) == (STATE_FOLLOWING, 0))
check("FOLLOWING + one-side-only stays in FOLLOWING (not a drop to CREEPING)",
      step_state(STATE_FOLLOWING, True, False, 0, 15) == (STATE_FOLLOWING, 0))
check("FOLLOWING + nothing valid -> CREEPING, streak reset",
      step_state(STATE_FOLLOWING, False, False, 7, 15) == (STATE_CREEPING, 0))

# --- CREEPING ---
check("CREEPING + both_valid recovered -> FOLLOWING",
      step_state(STATE_CREEPING, True, True, 5, 15) == (STATE_FOLLOWING, 0))
check("CREEPING + one-side-only counts as recovery (a gap, not an end), streak resets",
      step_state(STATE_CREEPING, True, False, 5, 15) == (STATE_CREEPING, 0))
check("CREEPING + nothing valid increments the streak",
      step_state(STATE_CREEPING, False, False, 5, 15) == (STATE_CREEPING, 6))
check("CREEPING + streak reaches window_size -> ROW_END",
      step_state(STATE_CREEPING, False, False, 14, 15) == (STATE_ROW_END, 15))

# --- ROW_END ---
check("ROW_END is terminal (stays ROW_END regardless of signal)",
      step_state(STATE_ROW_END, True, True, 15, 15) == (STATE_ROW_END, 15))


# ============================================================
# Layer 2: integrated dry run -- real RowLineFit, synthetic hallway
# ============================================================

RNG = np.random.default_rng(seed=11)
HALF_WIDTH = 0.9   # m -- generous hallway half-width, not cotton row spacing
Z_MID = 0.35


def make_hallway_scan(robot_x, n_wall_pts=60, x_range=(0.3, 5.0), y_jitter=0.02, half_width=HALF_WIDTH):
    """Dense, continuous wall returns -- the point of this test is that a
    hallway is EASIER than sparse cotton (see row_follower_v3.py's module
    docstring), so point counts here are deliberately much higher than
    test_row_line_fit.py's sparse per-plant counts."""
    x_l = RNG.uniform(x_range[0], x_range[1], n_wall_pts)
    y_l = half_width + RNG.normal(0.0, y_jitter, n_wall_pts)
    x_r = RNG.uniform(x_range[0], x_range[1], n_wall_pts)
    y_r = -half_width + RNG.normal(0.0, y_jitter, n_wall_pts)
    left = np.column_stack((x_l, y_l, np.full(n_wall_pts, Z_MID)))
    right = np.column_stack((x_r, y_r, np.full(n_wall_pts, Z_MID)))
    return np.vstack([left, right])


def make_empty_scan():
    return np.empty((0, 3))


rlf = RowLineFit(overrides={"roi_y_abs_max": 1.5}, seed=3)  # widen ROI for the hallway's larger half-width
state, streak = STATE_ACQUIRE, 0
robot_x = 0.0
states_seen = [state]

# Drive down a clean hallway long enough to reach FOLLOWING.
for _ in range(30):
    scan = make_hallway_scan(robot_x)
    rlf.update(scan, pose=(robot_x, 0.0, 0.0))
    state, streak = step_state(state, rlf.any_valid(), rlf.both_valid(), streak, rlf.window_size)
    states_seen.append(state)
    robot_x += 0.02

check("integrated: reaches FOLLOWING within 30 clean hallway scans",
      state == STATE_FOLLOWING, "final state=%s, sequence tail=%s" % (state, states_seen[-5:]))

# Now the hallway ends -- feed empty scans and confirm ROW_END is reached,
# through the real RowLineFit + step_state() pipeline together.
#
# IMPORTANT LATENCY NOTE, found via this test (2026-09-30): this takes TWO
# stacked windows, not one. RowLineFit's own accumulation buffer takes
# window_size blind scans to go fully stale before any_valid() even turns
# False (see row_line_fit.py's staleness tracking) -- only THEN does
# step_state()'s own streak start counting, and it needs another
# window_size before declaring ROW_END. Total: ~2*window_size scans
# (~3s at 9.9Hz), not window_size (~1.5s).
#
# row_confidence.py + row_follower_v2.py avoid this by feeding the CREEPING
# streak row_confidence's INSTANT (unsmoothed) per-scan score instead of its
# windowed one -- decoupling the streak from the smoothing window entirely.
# RowLineFit has no equivalent "this scan alone" signal (any_valid() is
# necessarily a function of the accumulated buffer), so this doubling is a
# real, current characteristic of row_follower_v3, not a test artifact.
# Worth revisiting after the first hallway run shows whether ~3s of extra
# creep-forward distance past a real row end actually matters in practice.
row_end_at = None
for i in range(2 * rlf.window_size + 5):
    rlf.update(make_empty_scan(), pose=(robot_x, 0.0, 0.0))
    state, streak = step_state(state, rlf.any_valid(), rlf.both_valid(), streak, rlf.window_size)
    robot_x += 0.02
    if state == STATE_ROW_END and row_end_at is None:
        row_end_at = i

check("integrated: reaches ROW_END after the hallway ends (within 2*window_size+5 blind scans)",
      state == STATE_ROW_END, "final state=%s, reached at blind-scan #%s" % (state, row_end_at))
check("integrated: ROW_END wasn't reached instantly (real dwell, not a hair-trigger)",
      row_end_at is not None and row_end_at >= rlf.window_size - 1,
      "reached at #%s, window_size=%d" % (row_end_at, rlf.window_size))

# --- baseline gate + steering law (2026-10-28) ---
_e = {"cross_track": 0.1, "heading_error": 0.05, "estimated": False}
check("baseline gate passes err and both_valid through once baseline is ok",
      apply_baseline_gate(_e, True, True) == (_e, True))
check("baseline gate drops err and withholds both_valid while the robot has not moved",
      apply_baseline_gate(_e, True, False) == (None, False))
ang, last = steering_command(_e, 0.0, 0.3, 0.3, 0.08)
check("steering is negative feedback on both terms (robot left + pointing left -> turn right)", ang < 0, "ang=%.4f" % ang)
ang, last = steering_command({"cross_track": 5.0, "heading_error": 5.0, "estimated": False}, 0.0, 0.3, 0.3, 0.08)
check("steering is clipped to max_angular", abs(ang) == 0.08 and last == ang, "ang=%.4f" % ang)
ang, last = steering_command(None, -0.08, 0.3, 0.3, 0.08)
ang2, last2 = steering_command(None, last, 0.3, 0.3, 0.08)
check("blind fallback decays toward zero and STORES the decayed value (-0.08 -> -0.04 -> -0.02)",
      abs(ang + 0.04) < 1e-9 and abs(ang2 + 0.02) < 1e-9 and last2 == ang2, "ang=%.4f ang2=%.4f" % (ang, ang2))

# --- forward obstacle stop (safety.py, 2026-10-28) ---
import safety as _safety
_c = _safety.DEFAULTS
_ahead = np.column_stack((np.full(200, 1.0), np.linspace(-0.15, 0.15, 200), np.linspace(0.1, 0.9, 200)))
_walls = np.column_stack((np.linspace(0.3, 5.0, 400), np.r_[np.full(200, 1.2), np.full(200, -1.2)], np.full(400, 0.5)))
check("points_in_path counts a person-sized object 1 m ahead",
      _safety.points_in_path(_ahead, _c["x_min"], _c["x_max"], _c["half_width"], _c["z_min"], _c["z_max"]) == 200)
check("points_in_path ignores hallway walls 1.2 m to the side",
      _safety.points_in_path(_walls, _c["x_min"], _c["x_max"], _c["half_width"], _c["z_min"], _c["z_max"]) == 0)
_g = _safety.ObstacleGate(min_points=20, trigger_scans=3, clear_scans=10)
_seq = [_g.update(n) for n in (200, 200, 200, 200)]
check("gate blocks only after trigger_scans consecutive hits", _seq == [False, False, True, True], "seq=%s" % _seq)
_seq = [_g.update(0) for _ in range(10)]
check("gate stays blocked until clear_scans consecutive clear scans", _seq[:9] == [True] * 9 and _seq[9] is False,
      "seq=%s" % _seq)
_g2 = _safety.ObstacleGate(min_points=20, trigger_scans=3, clear_scans=10)
_seq = [_g2.update(n) for n in (200, 0, 200, 0, 200, 0)]
check("isolated single-scan hits never block (debounce)", not any(_seq), "seq=%s" % _seq)

# --- gyro yaw-drift correction (odom_drift.py, 2026-11-03) ---
from odom_drift import YawDriftCorrector

DRIFT = 0.005   # rad/s: the UM7 bias measured with the robot parked on 2026-11-03


def drifting_odom(segments, drift=DRIFT, hz=50.0):
    """Raw odometry for a robot whose TRUE heading never changes, while its yaw creeps at
    `drift` and its position is integrated along that drifting yaw (what the EKF does).
    segments: [(seconds, forward_speed, extra_turn_rate), ...]. Yields (t, x, y, yaw, speed)."""
    t = x = y = yaw = 0.0
    for seconds, speed, turn in segments:
        for _ in range(int(seconds * hz)):
            yield t, x, y, yaw, speed
            yaw += (drift + turn) / hz
            x += speed / hz * math.cos(yaw)
            y += speed / hz * math.sin(yaw)
            t += 1.0 / hz


_c = YawDriftCorrector()
_out = [(_s[4], _c.update(*_s)) for _s in drifting_odom([(3.0, 0.0, 0.0), (20.0, 0.2, 0.0)])]
_drive = [p for v, p in _out if v > 0]
check("drift: learns the parked yaw rate (within 5%)", abs(_c.rate - DRIFT) < 0.05 * DRIFT, "rate=%.5f" % _c.rate)
check("drift: corrected yaw holds still over a 20 s straight drive (raw drifts 0.1 rad)",
      abs(_drive[-1][2] - _drive[0][2]) < 0.002, "change=%.5f rad" % (_drive[-1][2] - _drive[0][2]))
_travel = math.atan2(_drive[-1][1] - _drive[0][1], _drive[-1][0] - _drive[0][0])
check("drift: corrected path runs along the corrected heading (path and yaw stay consistent)",
      abs(_travel - _drive[-1][2]) < 0.002, "travel=%.4f yaw=%.4f" % (_travel, _drive[-1][2]))
_c = YawDriftCorrector()
_same = all(abs(a - b) < 1e-12 for _s in drifting_odom([(10.0, 0.2, 0.0)]) for a, b in zip(_c.update(*_s), _s[1:4]))
check("drift: never parked -> nothing learned, pose passes through unchanged", _same and not _c.learned)
_c = YawDriftCorrector()
for _s in drifting_odom([(4.0, 0.0, 0.5)]):
    _c.update(*_s)
check("drift: turning in place with the joystick is not mistaken for drift", not _c.learned, "rate=%.4f" % _c.rate)
_c = YawDriftCorrector()
for _s in drifting_odom([(3.0, 0.0, 0.0), (5.0, 0.2, 0.0), (3.0, 0.0, -0.002)]):
    _c.update(*_s)
check("drift: a later stop refreshes the estimate (bias moved 0.005 -> 0.003)",
      abs(_c.rate - 0.003) < 0.0003, "rate=%.5f" % _c.rate)


def hallway_with_drift(correct, n_scans=300, v=0.2, hz=9.9):
    """The 2026-11-03 bug, reproduced: drive dead-center down a straight hallway while the
    odometry yaw drifts. Returns the detector's mean (heading_error, cross_track)."""
    det = RowLineFit(overrides={"roi_y_abs_max": 1.5}, seed=3)
    corr = YawDriftCorrector()
    odom = drifting_odom([(3.0, 0.0, 0.0), (n_scans / hz, v, 0.0)])
    hes, xts = [], []
    for k, s in enumerate(odom):
        pose = corr.update(*s) if correct else s[1:4]
        if s[4] == 0.0 or k % 5:      # odometry at 50 Hz, a scan every 5th sample (~10 Hz)
            continue
        det.update(make_hallway_scan(0.0), pose=pose)   # walls in the TRUE robot frame: centered, parallel
        err = det.centerline_error(pose)
        if err is not None and det._scan_i > 30:
            hes.append(err["heading_error"])
            xts.append(err["cross_track"])
    return (float(np.mean(hes)), float(np.mean(xts)), len(hes)) if hes else (float("nan"), float("nan"), 0)


# The synthetic walls have 60 points each, so the per-scan centroid jitters ~0.2 m along
# the wall; that dilutes the heading tilt here (real walls give thousands of points and
# showed -1.9 deg). The cross-track shift survives either way.
_he, _xt, _n = hallway_with_drift(correct=False)
check("drift bug reproduced: raw drifting odometry biases heading and cross-track",
      _n > 100 and _he < -math.radians(0.3) and _xt > 0.02,
      "mean he=%.2f deg, xt=%.1f cm, n=%d" % (math.degrees(_he), 100 * _xt, _n))
_he, _xt, _n = hallway_with_drift(correct=True)
check("drift corrected: heading and cross-track bias gone (<0.3 deg, <1 cm)",
      _n > 100 and abs(_he) < math.radians(0.3) and abs(_xt) < 0.01,
      "mean he=%.2f deg, xt=%.1f cm, n=%d" % (math.degrees(_he), 100 * _xt, _n))

# --- Summary -----------------------------------------------------------------
n_pass = sum(1 for r in results if r)
print("\n%d/%d checks passed" % (n_pass, len(results)))
if n_pass != len(results):
    raise SystemExit(1)
