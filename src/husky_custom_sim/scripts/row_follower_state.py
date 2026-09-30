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
