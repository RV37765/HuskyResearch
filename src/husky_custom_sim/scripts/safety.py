#!/usr/bin/env python3
"""
safety.py -- forward obstacle stop for the row follower. Plain numpy, no rospy.

WHY THIS EXISTS
---------------
row_follower_v3 only reasons about the two row boundaries. Nothing looked at
the space directly in front of the robot: on 2026-10-28 it "avoided" the
hallway trash cans only because they shifted the fitted wall, and a person
stepping in front mid-row would not have stopped it. This is required before
any unattended run.

HOW IT DECIDES
--------------
Count points (base_link frame) inside a box in the robot's path. Block after
`trigger_scans` consecutive scans with at least `min_points`; unblock after
`clear_scans` consecutive clear scans (hysteresis, so it doesn't stutter).

CALIBRATION (2026-10-28, both real hallway runs, 2,968 scans)
-------------------------------------------------------------
The default box (0.3-1.5 m ahead, |y| <= 0.435 m = half the 670 mm Husky plus
10 cm, 0.05-1.2 m above base_link) held at most 3 points in any scan; the
nearest in-path point was never closer than 1.44 m. A person-sized object at
1 m returns hundreds of points, so min_points=20 separates the two with a
wide margin and gave zero false triggers on all recorded data.

LIMITS
------
* The VLP-16 sees only +/-15 degrees vertically from ~0.51 m up, so very low
  objects right at the bumper are invisible (at 0.5 m it sees ~0.38-0.64 m
  above ground). It is a person/large-object stop, not a curb detector.
* The driver's min_range drops returns closer than that range; the box
  extends to 1.5 m so an approaching obstacle is caught before then.
* In a crop row, weeds or leaning plants in the inter-row will land in the
  box; the field overlay may need a narrower box or a higher min_points.
"""

import numpy as np

DEFAULTS = {
    "x_min": 0.3,
    "x_max": 1.5,
    "half_width": 0.435,
    "z_min": 0.05,
    "z_max": 1.2,
    "min_points": 20,
    "trigger_scans": 3,
    "clear_scans": 10,
}


def points_in_path(pts, x_min, x_max, half_width, z_min, z_max):
    """Number of (N, >=3) base_link points inside the forward box."""
    pts = np.asarray(pts)
    if pts.size == 0:
        return 0
    inside = ((pts[:, 0] >= x_min) & (pts[:, 0] <= x_max)
              & (np.abs(pts[:, 1]) <= half_width)
              & (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max))
    return int(np.count_nonzero(inside))


class ObstacleGate:
    """Debounced blocked/clear decision from per-scan point counts."""

    def __init__(self, min_points=20, trigger_scans=3, clear_scans=10):
        self.min_points = int(min_points)
        self.trigger_scans = int(trigger_scans)
        self.clear_scans = int(clear_scans)
        self.blocked = False
        self._hits = 0
        self._clear = 0

    def update(self, n_points):
        """Feed this scan's in-path count; returns True while the path is blocked."""
        if n_points >= self.min_points:
            self._hits += 1
            self._clear = 0
            if self._hits >= self.trigger_scans:
                self.blocked = True
        else:
            self._clear += 1
            self._hits = 0
            if self.blocked and self._clear >= self.clear_scans:
                self.blocked = False
        return self.blocked
