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

import numpy as np

from row_follower_state import (
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

# --- Summary -----------------------------------------------------------------
n_pass = sum(1 for r in results if r)
print("\n%d/%d checks passed" % (n_pass, len(results)))
if n_pass != len(results):
    raise SystemExit(1)
