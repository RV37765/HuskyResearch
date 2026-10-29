#!/usr/bin/env python3
"""
measure_corridor.py -- measure the left/right boundary geometry from the LiDAR
itself instead of guessing it. READ-ONLY: never publishes /cmd_vel.

Stand the robot still between two walls (hallway) or two crop rows (field),
run this, and it reports what the sensor actually sees:
  * the sensor -> base_link transform, and whether raw left/right is mirrored
  * distance from base_link to the left and right boundary lines
  * boundary-to-boundary spacing and half-width  -> row_half_width
  * the robot's offset from center and yaw relative to the boundaries
  * suggested roi_y_abs_max, plus the height distribution of boundary points

In the hallway it sets row_line_fit_hallway.yaml from measurement. In the
field (Stage 3) the same command measures real row spacing from where the
robot actually sits -- the number Stage 2 can only estimate.

Usage (robot's system bringup must be running; no launch file needed):
    rosrun husky_custom_sim measure_corridor.py
    rosrun husky_custom_sim measure_corridor.py _scans:=20 _z_min:=0.3 _z_max:=1.0
"""

import math

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
from sensor_msgs.msg import PointCloud2

from frame_utils import apply_rigid, describe, rigid_from_transform_msg
from row_line_fit import RowLineFit


def main():
    rospy.init_node("measure_corridor", anonymous=True)
    topic = rospy.get_param("~cloud_topic", "/velodyne_points")
    target = rospy.get_param("~target_frame", "base_link")
    n_scans = int(rospy.get_param("~scans", 10))
    z_min = float(rospy.get_param("~z_min", 0.20))
    z_max = float(rospy.get_param("~z_max", 1.20))
    x_min, x_max = float(rospy.get_param("~x_min", 0.3)), float(rospy.get_param("~x_max", 5.0))
    y_max = float(rospy.get_param("~y_max", 4.0))

    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf)

    rospy.loginfo("Waiting for %d scans on %s ...", n_scans, topic)
    clouds = []
    while len(clouds) < n_scans and not rospy.is_shutdown():
        clouds.append(rospy.wait_for_message(topic, PointCloud2, timeout=10.0))

    frame = clouds[0].header.frame_id
    tf_msg = buf.lookup_transform(target, frame, rospy.Time(0), rospy.Duration(3.0))
    R, t = rigid_from_transform_msg(tf_msg.transform)

    pts = np.vstack([
        apply_rigid(np.array(list(pc2.read_points(c, field_names=("x", "y", "z"), skip_nans=True)),
                             dtype=float), R, t)
        for c in clouds
    ])
    keep = ((pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) & (np.abs(pts[:, 1]) <= y_max)
            & (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max))
    band = pts[keep]

    fitter = RowLineFit(overrides={"ransac_inlier_thresh": 0.05, "ransac_iters": 300}, seed=0)
    sides = {}
    for name, mask in (("left", band[:, 1] > 0), ("right", band[:, 1] < 0)):
        side = band[mask]
        if side.shape[0] < 20:
            sides[name] = None
            continue
        # Keep the nearest boundary on this side: the dominant cluster of |y|.
        near = side[np.abs(side[:, 1]) <= np.percentile(np.abs(side[:, 1]), 60) + 0.3]
        fit = fitter._ransac_line(near[:, :2])
        if not fit.get("valid"):
            sides[name] = None
            continue
        d = fit["direction"] if fit["direction"][0] >= 0 else -fit["direction"]
        normal = np.array([-d[1], d[0]])
        dist = abs(float(np.dot(fit["point"], normal)))   # base_link origin -> line
        yaw = math.degrees(math.atan2(d[1], d[0]))        # line angle vs robot heading
        inl = near[np.abs((near[:, :2] - fit["point"]).dot(normal)) <= 0.05]
        sides[name] = {"dist": dist, "yaw": yaw, "n": int(inl.shape[0]),
                       "z5": float(np.percentile(inl[:, 2], 5)), "z95": float(np.percentile(inl[:, 2], 95))}

    print("\n================ measure_corridor ================")
    print("Sensor transform %s <- %s:\n  %s" % (target, frame, describe(R, t)))
    print("Points in band z=[%.2f, %.2f] x=[%.1f, %.1f]: %d (from %d scans)"
          % (z_min, z_max, x_min, x_max, band.shape[0], len(clouds)))
    for name in ("left", "right"):
        s = sides[name]
        if s is None:
            print("  %-5s boundary: NOT FOUND (too few points / no clean line on this side)" % name)
        else:
            print("  %-5s boundary: %.3f m away, line yaw %+.1f deg, %d inlier pts, heights %.2f-%.2f m"
                  % (name, s["dist"], s["yaw"], s["n"], s["z5"], s["z95"]))
    if sides["left"] and sides["right"]:
        L, Rr = sides["left"]["dist"], sides["right"]["dist"]
        width = L + Rr
        yaw = 0.5 * (sides["left"]["yaw"] + sides["right"]["yaw"])
        print("\n  spacing (boundary to boundary): %.3f m" % width)
        print("  offset from center: %+.3f m (positive = robot is LEFT of center)" % ((Rr - L) / 2.0))
        print("  robot yaw relative to boundaries: %+.1f deg %s" % (
            yaw, "(OK, inside the ~10 deg detector envelope)" if abs(yaw) < 8 else
            "(TOO LARGE -- straighten the robot before testing)"))
        print("\n  Suggested overlay values:")
        print("    row_half_width: %.2f" % (width / 2.0))
        print("    roi_y_abs_max:  %.2f   (farther boundary + 0.6 m margin)" % (max(L, Rr) + 0.6))
    print("===================================================\n")


if __name__ == "__main__":
    main()
