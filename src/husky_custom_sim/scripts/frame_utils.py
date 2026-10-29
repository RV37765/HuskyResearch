#!/usr/bin/env python3
"""
frame_utils.py -- move raw LiDAR points into base_link before anything reasons
about "left" and "right". Plain numpy, no rospy, so it's testable offline.

WHY THIS EXISTS
---------------
The VLP-16 is mounted upside-down (180-degree roll in the URDF). The driver
publishes /velodyne_points in the sensor's own frame, so a point physically on
the robot's LEFT can arrive with a NEGATIVE y, and "up" is flipped. Anything
that reads raw points without applying TF is reasoning in a mirrored world.

Evidence this is real, not hypothetical: the Spring row_follower.py computes
error = left_dist - right_dist from the "+angle = left" sector and steers
angular = -K * error. With un-mirrored data that steers TOWARD the closer wall
and would diverge -- yet it held +/-1-3 cm for a whole semester. That sign is
only stabilizing if the scan is mirrored. row_line_fit.py consumed the same
sensor's raw points with no transform, so its cross_track and heading_error
were computed in that mirrored frame (2026-10-28).

The fix is correct by construction rather than assuming a flip: look up the
real transform from TF once (it's static) and apply it. If the data turns out
not to be mirrored, this is just a small translation; if it is, it un-mirrors.
"""

import math

import numpy as np


def quat_to_rot(qx, qy, qz, qw):
    """Unit quaternion -> 3x3 rotation matrix."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def rot_to_rpy(R):
    """3x3 rotation matrix -> (roll, pitch, yaw) in radians, ZYX convention (same as tf_echo)."""
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(math.cos(pitch)) > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock -- yaw and roll are coupled; report it all as roll
        roll = math.atan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return roll, pitch, yaw


def rigid_from_transform_msg(transform):
    """geometry_msgs/Transform (duck-typed: .translation.{x,y,z}, .rotation.{x,y,z,w})
    -> (R 3x3, t 3,) mapping points FROM the child/source frame INTO the parent/target frame."""
    q = transform.rotation
    t = transform.translation
    return quat_to_rot(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z], dtype=float)


def apply_rigid(pts, R, t):
    """Apply p' = R p + t to an (N, >=3) array; returns (N, 3)."""
    pts = np.asarray(pts, dtype=float)
    if pts.size == 0:
        return np.empty((0, 3))
    return pts[:, :3].dot(R.T) + t


def describe(R, t):
    """One-line human-readable summary for logs: translation + RPY in degrees,
    plus an explicit verdict on whether left/right is mirrored."""
    roll, pitch, yaw = rot_to_rpy(R)
    # R maps sensor -> base_link. The sensor's +y axis expressed in base_link is R[:, 1];
    # if that points toward base_link -y, raw "left" is physically the robot's right.
    mirrored = R[1, 1] < 0
    return ("t=[%.3f, %.3f, %.3f] m  rpy=[%.1f, %.1f, %.1f] deg  -> raw sensor y is %s"
            % (t[0], t[1], t[2], math.degrees(roll), math.degrees(pitch), math.degrees(yaw),
               "MIRRORED (raw +y = robot's RIGHT)" if mirrored else "not mirrored"))


def chain_to_target(static_tfs, source_frame, target_frame):
    """Compose a chain of static transforms into one (R, t) mapping points from
    source_frame into target_frame.

    static_tfs: {child_frame: (parent_frame, R, t)} -- e.g. built from a bag's
    /tf_static, where each entry maps child -> parent. Walks child -> parent
    until target_frame is reached; raises KeyError listing known frames if the
    chain is broken. Offline stand-in for tf2's lookup_transform so the replay
    tool applies the SAME sensor transform the live node did.
    """
    R_tot, t_tot = np.eye(3), np.zeros(3)
    frame, hops = source_frame, 0
    while frame != target_frame:
        if frame not in static_tfs or hops > 32:
            raise KeyError("no static chain %s -> %s (stuck at %r; known children: %s)"
                           % (source_frame, target_frame, frame, sorted(static_tfs)))
        parent, R, t = static_tfs[frame]
        R_tot, t_tot = R.dot(R_tot), R.dot(t_tot) + t
        frame, hops = parent, hops + 1
    return R_tot, t_tot


def cloud_xyz(msg):
    """Fast sensor_msgs/PointCloud2 -> (N, 3) float64 array of finite x, y, z.

    Replaces list(pc2.read_points(...)) + np.array(), which cost ~33 ms per
    26,000-point scan in pure Python on the 2026-10-28 profile; this is a
    vectorized view over the raw byte buffer. Duck-typed on the message
    (fields / point_step / data / is_bigendian) so it needs no ROS import.
    Only FLOAT32 x/y/z fields are supported -- true of the Velodyne driver.
    """
    offs = {}
    for f in msg.fields:
        if f.name in ("x", "y", "z"):
            if f.datatype != 7:  # sensor_msgs/PointField.FLOAT32
                raise ValueError("cloud_xyz: field %s is not FLOAT32 (datatype %d)" % (f.name, f.datatype))
            offs[f.name] = f.offset
    if len(offs) != 3:
        raise ValueError("cloud_xyz: cloud has no x/y/z fields")
    n = msg.width * msg.height
    if n == 0:
        return np.empty((0, 3))
    raw = np.frombuffer(msg.data, dtype=np.uint8, count=n * msg.point_step).reshape(n, msg.point_step)
    dt = ">f4" if msg.is_bigendian else "<f4"
    cols = [np.ascontiguousarray(raw[:, offs[k]:offs[k] + 4]).view(dt).reshape(n).astype(np.float64)
            for k in ("x", "y", "z")]
    pts = np.column_stack(cols)
    return pts[np.all(np.isfinite(pts), axis=1)]
