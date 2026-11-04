#!/usr/bin/env python3
"""
odom_drift.py -- remove the gyro's yaw drift from the odometry pose the
detector uses. rospy-free, so row_follower_v3.py and replay_bag_log.py share
the identical code and test_row_follower_v3.py can exercise it without ROS.

WHY THIS EXISTS (2026-11-03)
----------------------------
The robot's EKF (/odometry/filtered) takes its yaw rate from the UM7 gyro,
and that gyro read +0.005 rad/s with the robot parked: the EKF's yaw turned
+17 deg/min while nothing moved (wheel odometry read 0). RowLineFit places
each scan's wall centroids ~1.3 m ahead using this pose, so a drifting yaw
swings successive centroids sideways. The fitted row line comes out tilted
(heading error -1.9 deg) and, extended back to the robot, shifted
(cross-track +4 cm). The steering law then holds the robot where the two
cancel: the detector reports ~+3 cm while the robot is really ~1 cm right
of center. The effect scales with drift rate * look-ahead / speed, so at
creep speed it is 4x larger. Wheel odometry is not the fix either: it does
not drift while parked, but it drifts the other way while driving.

HOW
---
While the robot is parked, the odometry's yaw rate IS the drift. Learn it,
integrate the learned rate into a correction, and subtract that from the
yaw. The position is rebuilt from the odometry's own increments rotated by
the same correction -- correcting the yaw alone would leave the path bending
away from the heading a little more every second, which is worse.

"Parked" means forward speed below park_speed for at least park_sec. Every
such stretch refreshes the estimate: the bias moves slowly with temperature
(+11 to +14 deg/min on Oct 28, +17 on Nov 3, steady within a session). A
parked stretch with a fast turn in it is someone rotating the robot in place
with the joystick, not drift, so it is thrown away.
"""

import math


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class YawDriftCorrector:
    def __init__(self, park_speed=0.01, park_sec=2.0, max_rate=0.02, spike_rate=0.1):
        self.park_speed = park_speed   # m/s; slower than this counts as parked
        self.park_sec = park_sec       # s of parking before an estimate is trusted
        self.max_rate = max_rate       # rad/s; a larger "drift" is a real turn, not bias
        self.spike_rate = spike_rate   # rad/s between samples; above this the stretch restarts
        self.rate = 0.0                # learned drift, rad/s (+ = yaw creeps counter-clockwise)
        self.learned = False           # True once any parked stretch produced an estimate
        self._corr = 0.0               # integrated correction, rad
        self._prev = None              # previous raw sample (t, x, y, yaw)
        self._pose = None              # corrected (x, y, yaw)
        self._yaw_u = 0.0              # unwrapped raw yaw
        self._park = None              # (t, unwrapped yaw) where the current parked stretch began

    def update(self, t, x, y, yaw, speed):
        """Feed one raw odometry sample (time s, x m, y m, yaw rad, forward speed m/s).
        Returns the corrected (x, y, yaw)."""
        if self._prev is None:
            self._prev = (t, x, y, yaw)
            self._yaw_u = yaw
            self._pose = (x, y, yaw)
            return self._pose
        t0, x0, y0, yaw0 = self._prev
        dt = t - t0
        if dt <= 0.0:
            return self._pose
        dyaw = _wrap(yaw - yaw0)
        self._yaw_u += dyaw
        self._corr += self.rate * dt

        # Rotate this raw position increment by -corr and add it to the corrected path.
        c, s = math.cos(self._corr), math.sin(self._corr)
        dx, dy = x - x0, y - y0
        px, py, _ = self._pose
        self._pose = (px + c * dx + s * dy, py - s * dx + c * dy, _wrap(yaw - self._corr))

        if abs(speed) >= self.park_speed or abs(dyaw / dt) > self.spike_rate:
            self._park = None                       # driving, or turning in place: no evidence
        elif self._park is None:
            self._park = (t, self._yaw_u)
        else:
            dur = t - self._park[0]
            if dur >= self.park_sec:
                est = (self._yaw_u - self._park[1]) / dur
                if abs(est) <= self.max_rate:
                    self.rate = est
                    self.learned = True
        self._prev = (t, x, y, yaw)
        return self._pose
