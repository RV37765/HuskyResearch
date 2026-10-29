#!/usr/bin/env python3
"""
row_follower_state.py -- the row-following state machine's pure transition
logic, factored out of row_follower_v3.py specifically so it's testable
without ROS. row_follower_v3.py imports rospy at module level (it's the
ROS-facing node); test_row_follower_v3.py needs to exercise the state
transitions without needing WSL2/ROS just to run a plain-Python check --
same "keep the core logic dependency-free" principle row_line_fit.py
already applies to its own algorithm, extended here to the state machine
built on top of it.

STATE MACHINE (see row_follower_v3.py for the full narrative)
------------------------------------------------------------------
    ACQUIRE   -- cold start. Wait for both_valid() before trusting FOLLOWING.
    FOLLOWING -- full speed. Drops to CREEPING only when BOTH sides are lost.
    CREEPING  -- seeing EITHER side again means it was a gap, not a row end:
                 reset the streak immediately. Only a fully-blind scan
                 (neither side seen) counts toward the row-end streak.
    ROW_END   -- terminal. Turnaround is a separate, later phase.
"""

STATE_ACQUIRE = "ACQUIRE"
STATE_FOLLOWING = "FOLLOWING"
STATE_CREEPING = "CREEPING"
STATE_ROW_END = "ROW_END"


def step_state(state, any_valid, both_valid, low_conf_streak, window_size):
    """Pure state-transition function -- no ROS, no robot, no RowLineFit
    instance. Takes this tick's signals and the current state/streak,
    returns (new_state, new_streak).
    """
    if state == STATE_ACQUIRE:
        if both_valid:
            return STATE_FOLLOWING, 0
        return STATE_ACQUIRE, 0

    if state == STATE_FOLLOWING:
        if not any_valid:
            return STATE_CREEPING, 0
        return STATE_FOLLOWING, 0

    if state == STATE_CREEPING:
        if any_valid:
            return (STATE_FOLLOWING, 0) if both_valid else (STATE_CREEPING, 0)
        streak = low_conf_streak + 1
        if streak >= window_size:
            return STATE_ROW_END, streak
        return STATE_CREEPING, streak

    return STATE_ROW_END, low_conf_streak


def steering_command(err, last_angular, k_lateral, k_heading, max_angular):
    """Pure steering law -- returns (angular, new_last_angular).

    err is RowLineFit.centerline_error()'s dict, or None when neither side
    has a valid fit. Both errors share one convention ("robot relative to
    the row", see centerline_error's docstring), so the law below is
    negative feedback on BOTH terms.

    When blind, decay the last command toward zero instead of holding it.
    The decayed value must be stored back -- returning last_angular * 0.5
    without reassigning it held a constant nonzero turn forever (found in the
    field 2026-10-22: a long blind stretch kept the robot slowly turning
    into a wall).

    Lives here (not in row_follower_v3.py) for the same reason step_state()
    does: so it can be tested without ROS, and so replay_bag_log.py runs the
    identical law instead of a hand-copied formula that could drift.
    """
    if err is None:
        decayed = last_angular * 0.5
        return decayed, decayed
    angular = -(k_lateral * err["cross_track"] + k_heading * err["heading_error"])
    angular = max(-max_angular, min(max_angular, angular))
    return angular, angular


def apply_baseline_gate(err, both_valid, baseline_ok):
    """Don't act on a line DIRECTION that has no travel baseline behind it.

    With the robot parked or just starting, the accumulated centroids all sit
    at one spot, so the fitted heading (and the cross-track measured against
    it) is noise -- 43 of 72 large cross-track spikes in the 2026-10-28 run
    were this. Returns (err_for_steering, both_valid_for_state):
      * err is dropped (-> steering decays toward zero) until baseline_ok;
      * both_valid is withheld so ACQUIRE can't promote to FOLLOWING on a
        direction that isn't trustworthy yet.
    any_valid (the row-end signal) is deliberately NOT gated: a robot that is
    paused still sees its walls, and must not drift toward a false ROW_END.
    """
    if baseline_ok:
        return err, both_valid
    return None, False
