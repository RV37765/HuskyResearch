#!/usr/bin/env python3
"""
row_line_fit_node.py -- read-only ROS diagnostic wrapper for row_line_fit.py.

WHY A SEPARATE, READ-ONLY NODE FIRST
--------------------------------------
row_line_fit.py has been validated offline (11/19 -- now 17/17 -- synthetic
tests, see test_row_line_fit.py) but has never touched a real sensor. This
node's only job is to change that safely: subscribe to the real
/velodyne_points and /odometry/filtered, run them through RowLineFit, and
republish the fit status so it can be watched in rqt_plot or Foxglove.

It does NOT publish to /cmd_vel. This mirrors exactly how row_confidence.py
was introduced -- a read-only diagnostic node first, and only once THAT was
trusted did row_follower_v2.py wire it into anything that drives the robot.
The same caution applies here, more so: row_confidence.py's core algorithm
had already run on real hardware before its bug was found; row_line_fit.py's
algorithm has not been near a sensor yet.

rospy LIVES HERE, NOT IN row_line_fit.py
-------------------------------------------
row_line_fit.py is deliberately rospy-free so its offline tests need nothing
but plain Python (see its module docstring). It also doesn't read the ROS
param server itself, unlike row_confidence.py's RowConfidence class -- this
node reads config/row_line_fit.yaml under the "row_line_fit" namespace and
passes the values in as a plain overrides dict.

Run standalone (sensors only, no move_base, no driving):
    roslaunch husky_custom_sim husky_real_row_map.launch
    rosrun husky_custom_sim row_line_fit_node.py
    rosrun husky_custom_sim row_line_fit_node.py _row_line_fit/cluster_mode:=dbscan

Watch it:
    rqt_plot /row_line_fit/cross_track /row_line_fit/heading_error
"""

import math

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Float32

from row_line_fit import DEFAULTS, RowLineFit


def _yaw_from_quaternion(q):
    """Standard quaternion -> yaw formula. Avoids adding a tf dependency for
    one number -- same "no unnecessary deps" spirit as row_line_fit.py
    itself avoiding rospy/sklearn."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _load_overrides(ns="~row_line_fit"):
    """Read every RowLineFit DEFAULTS key from the param server under ns,
    falling back to RowLineFit's own default for anything not set there."""
    return {key: rospy.get_param("%s/%s" % (ns, key), default) for key, default in DEFAULTS.items()}


class RowLineFitNode:
    def __init__(self):
        rospy.init_node("row_line_fit_node")

        overrides = _load_overrides()
        self.rlf = RowLineFit(overrides=overrides)

        self.cloud_topic = rospy.get_param("~cloud_topic", "/velodyne_points")
        self.odom_topic = rospy.get_param("~odom_topic", "/odometry/filtered")

        self.pose = (0.0, 0.0, 0.0)
        self._have_odom = False

        self.pub_any_valid = rospy.Publisher("~any_valid", Bool, queue_size=1)
        self.pub_both_valid = rospy.Publisher("~both_valid", Bool, queue_size=1)
        self.pub_estimated = rospy.Publisher("~estimated", Bool, queue_size=1)
        self.pub_cross_track = rospy.Publisher("~cross_track", Float32, queue_size=1)
        self.pub_heading_error = rospy.Publisher("~heading_error", Float32, queue_size=1)

        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self._on_odom, queue_size=1)
        self.cloud_sub = rospy.Subscriber(self.cloud_topic, PointCloud2, self._on_cloud, queue_size=1)

        rospy.loginfo(
            "row_line_fit_node up. cloud_topic=%s odom_topic=%s cluster_mode=%s window=%d",
            self.cloud_topic, self.odom_topic, self.rlf.cluster_mode, self.rlf.window_size,
        )
        rospy.loginfo("READ-ONLY diagnostic -- does not publish /cmd_vel. Safe to run anytime.")

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        yaw = _yaw_from_quaternion(msg.pose.pose.orientation)
        self.pose = (p.x, p.y, yaw)
        self._have_odom = True

    def _on_cloud(self, msg):
        if not self._have_odom:
            rospy.logwarn_throttle(5.0, "row_line_fit_node: waiting for first %s message", self.odom_topic)
            return

        raw = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        pts = np.array(raw, dtype=float) if raw else np.empty((0, 3))

        dbg = self.rlf.update(pts, pose=self.pose)
        any_valid = self.rlf.any_valid()
        both_valid = self.rlf.both_valid()
        err = self.rlf.centerline_error(self.pose)

        self.pub_any_valid.publish(Bool(data=any_valid))
        self.pub_both_valid.publish(Bool(data=both_valid))
        if err is not None:
            self.pub_cross_track.publish(Float32(data=err["cross_track"]))
            self.pub_heading_error.publish(Float32(data=err["heading_error"]))
            self.pub_estimated.publish(Bool(data=err["estimated"]))

        err_str = (
            "xt=%.3f hd=%.3f%s" % (err["cross_track"], err["heading_error"], " (est)" if err["estimated"] else "")
            if err is not None else "none"
        )
        rospy.loginfo_throttle(
            1.0,
            "status=%s n_canopy=%d any_valid=%s both_valid=%s err=%s",
            dbg.get("status"), dbg.get("n_canopy", 0), any_valid, both_valid, err_str,
        )


if __name__ == "__main__":
    try:
        RowLineFitNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
