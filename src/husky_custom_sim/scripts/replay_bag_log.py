#!/usr/bin/env python3
"""
replay_bag_log.py -- offline, per-scan CSV of a recorded row_follower_v3 run.

Replays a rosbag through the SAME pipeline the live node runs (sensor transform
-> RowLineFit -> steering_command -> step_state) and writes EVERY scan's
internal state, instead of the live node's once-a-second log line.

Record in the field:
    rosbag record /velodyne_points /odometry/filtered /cmd_vel /tf_static -O run.bag
(/tf_static is what lets this tool rebuild the sensor -> base_link transform.)

Replay (anywhere ROS is sourced; WSL is fine, no robot needed):
    python3 replay_bag_log.py run.bag run.csv --row-half-width 1.22
    python3 replay_bag_log.py run.bag run.csv --k-lateral 0.6 --k-heading 0.6 --max-angular 0.15
    python3 replay_bag_log.py run.bag run.csv --env field      (once row_line_fit_field.yaml exists)

Defaults match row_follower_v3.launch (hallway overlay, 0.3/0.3/0.08). Pass the
same overrides the run used. Because the replay is deterministic and offline,
you can ALSO change a parameter here and see what the detector would have done
on the identical recorded scans -- parameter experiments without driving.

Diagnostic columns beyond the live node's log:
  pose_age_sec      how stale the odometry pose was vs the scan it was paired with
  cloud_dt_sec      gap since the previous scan (a stall shows up here)
  clusters, n_canopy  why the detector did/didn't find two sides this scan
  cmd_linear/angular  what the live node actually COMMANDED (/cmd_vel in the bag)
  replay_angular    what this replay's steering law computes (should track the above
                    when gains/config match the run -- if not, the run used other settings)
"""

import argparse
import csv
import math
import os

import numpy as np
import rosbag
import yaml

from frame_utils import apply_rigid, chain_to_target, cloud_xyz, describe, rigid_from_transform_msg
from odom_drift import YawDriftCorrector
from row_follower_state import STATE_ACQUIRE, apply_baseline_gate, step_state, steering_command
from row_line_fit import DEFAULTS, RowLineFit
import safety

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config")


def _yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _load_yaml(name):
    path = os.path.join(CONFIG_DIR, name)
    if not os.path.exists(path):
        raise SystemExit("config file not found: %s" % path)
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _parse_overrides(pairs):
    overrides = {}
    for pair in pairs or []:
        key, _, val = pair.partition("=")
        if key not in DEFAULTS:
            raise SystemExit("--override key %r is not a row_line_fit.py DEFAULTS key" % key)
        default = DEFAULTS[key]
        overrides[key] = val if isinstance(default, str) else type(default)(val)
    return overrides


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag_path")
    ap.add_argument("csv_path")
    ap.add_argument("--env", default="hallway", help="config overlay: row_line_fit_<env>.yaml")
    ap.add_argument("--k-lateral", type=float, default=0.3)
    ap.add_argument("--k-heading", type=float, default=0.3)
    ap.add_argument("--max-angular", type=float, default=0.08)
    ap.add_argument("--row-half-width", type=float, default=None)
    ap.add_argument("--roi-y-abs-max", type=float, default=None)
    ap.add_argument("--target-frame", default="base_link")
    ap.add_argument("--cloud-topic", default="/velodyne_points")
    ap.add_argument("--odom-topic", default="/odometry/filtered")
    ap.add_argument("--cmd-topic", default="/cmd_vel")
    ap.add_argument("--no-transform", action="store_true",
                    help="skip the sensor->base_link transform (reproduces the pre-2026-10-28 behavior)")
    ap.add_argument("--override", action="append", metavar="KEY=VALUE",
                    help="any row_line_fit DEFAULTS key, e.g. --override max_cluster_y_spread=0.2; repeatable")
    ap.add_argument("--drift-correct", action="store_true",
                    help="remove the gyro's yaw drift from the odometry (odom_drift.py; matches ~drift_correct)")
    ap.add_argument("--stall-reset-sec", type=float, default=1.0,
                    help="reset the detector after a scan gap longer than this (0 = never; matches ~stall_reset_sec)")
    args = ap.parse_args()

    # Same layering as the launch file: code defaults < base yaml < env overlay < CLI.
    cfg = {}
    cfg.update(_load_yaml("row_line_fit.yaml"))
    cfg.update(_load_yaml("row_line_fit_%s.yaml" % args.env))
    if args.row_half_width is not None:
        cfg["row_half_width"] = args.row_half_width
    if args.roi_y_abs_max is not None:
        cfg["roi_y_abs_max"] = args.roi_y_abs_max
    cfg.update(_parse_overrides(args.override))
    rlf = RowLineFit(overrides=cfg, seed=0)
    print("config: roi_y_abs_max=%.2f row_half_width=%.2f z=[%.2f, %.2f] | gains %.2f/%.2f/%.3f"
          % (rlf.roi_y_abs_max, rlf.row_half_width, rlf.z_min, rlf.z_max,
             args.k_lateral, args.k_heading, args.max_angular))

    bag = rosbag.Bag(args.bag_path)

    # Static transforms: collect from the bag up front (they're latched, recorded once).
    static_tfs = {}
    for _, msg, _ in bag.read_messages(topics=["/tf_static"]):
        for ts in msg.transforms:
            R, t = rigid_from_transform_msg(ts.transform)
            static_tfs[ts.child_frame_id] = (ts.header.frame_id, R, t)
    sensor_rt = None  # resolved on the first cloud, once we know its frame_id

    state, streak, last_angular = STATE_ACQUIRE, 0, 0.0
    sc = safety.DEFAULTS
    gate = safety.ObstacleGate(sc["min_points"], sc["trigger_scans"], sc["clear_scans"])
    drift = YawDriftCorrector() if args.drift_correct else None
    pose, last_odom_stamp = (0.0, 0.0, 0.0), None
    cmd = (float("nan"), float("nan"))
    prev_cloud_t = None
    rows, scan_i, stall_resets = [], 0, 0

    for topic, msg, _ in bag.read_messages(topics=[args.cloud_topic, args.odom_topic, args.cmd_topic]):
        if topic == args.odom_topic:
            p = msg.pose.pose.position
            yaw = _yaw_from_quaternion(msg.pose.pose.orientation)
            if drift is not None:
                pose = drift.update(msg.header.stamp.to_sec(), p.x, p.y, yaw, msg.twist.twist.linear.x)
            else:
                pose = (p.x, p.y, yaw)
            last_odom_stamp = msg.header.stamp
            continue
        if topic == args.cmd_topic:
            cmd = (msg.linear.x, msg.angular.z)
            continue
        if last_odom_stamp is None:
            continue  # same as the live node: no pose yet, nothing to place the scan with

        scan_i += 1
        stamp = msg.header.stamp
        pose_age = (stamp - last_odom_stamp).to_sec()
        cloud_dt = (stamp.to_sec() - prev_cloud_t) if prev_cloud_t is not None else None
        prev_cloud_t = stamp.to_sec()
        if args.stall_reset_sec > 0 and cloud_dt is not None and cloud_dt > args.stall_reset_sec:
            rlf.reset()
            state, streak, last_angular = STATE_ACQUIRE, 0, 0.0
            stall_resets += 1

        if sensor_rt is None and not args.no_transform:
            try:
                sensor_rt = chain_to_target(static_tfs, msg.header.frame_id, args.target_frame)
            except KeyError as exc:
                raise SystemExit("%s\n(was /tf_static in the rosbag record command?)" % exc)
            print("sensor transform %s <- %s: %s" % (args.target_frame, msg.header.frame_id, describe(*sensor_rt)))

        pts = cloud_xyz(msg)
        if sensor_rt is not None:
            pts = apply_rigid(pts, *sensor_rt)

        path_pts = safety.points_in_path(pts, sc["x_min"], sc["x_max"], sc["half_width"], sc["z_min"], sc["z_max"])
        blocked = gate.update(path_pts)
        dbg = rlf.update(pts, pose=pose)
        left_fit, right_fit = rlf.fit_lines()
        any_valid = left_fit["valid"] or right_fit["valid"]
        both_valid = left_fit["valid"] and right_fit["valid"]
        err, both_valid = apply_baseline_gate(rlf.centerline_error(pose), both_valid, rlf.baseline_ok())
        angular, last_angular = steering_command(
            err, last_angular, args.k_lateral, args.k_heading, args.max_angular)

        prev_state = state
        state, streak = step_state(state, any_valid, both_valid, streak, rlf.window_size)

        def dir_deg(fit):
            if not fit.get("valid"):
                return ""
            return round(math.degrees(math.atan2(fit["direction"][1], fit["direction"][0])), 1)

        rows.append({
            "scan_i": scan_i,
            "t_sec": round(stamp.to_sec(), 3),
            "cloud_dt_sec": round(cloud_dt, 4) if cloud_dt is not None else "",
            "pose_age_sec": round(pose_age, 4) if pose_age is not None else "",
            "state": state, "state_changed": state != prev_state,
            "clusters": dbg["status"], "n_canopy": dbg["n_canopy"],
            "path_points": path_pts, "obstacle_blocked": blocked,
            "odom_x": round(pose[0], 3), "odom_y": round(pose[1], 3), "odom_theta_rad": round(pose[2], 4),
            "left_valid": left_fit["valid"], "left_inlier_frac": left_fit.get("inlier_frac", ""),
            "left_n": left_fit.get("n", ""), "left_dir_deg": dir_deg(left_fit),
            "right_valid": right_fit["valid"], "right_inlier_frac": right_fit.get("inlier_frac", ""),
            "right_n": right_fit.get("n", ""), "right_dir_deg": dir_deg(right_fit),
            "cross_track_m": round(err["cross_track"], 4) if err else "",
            "heading_error_rad": round(err["heading_error"], 4) if err else "",
            "estimated_one_sided": err["estimated"] if err else "",
            "replay_angular": round(angular, 4),
            "cmd_linear": round(cmd[0], 3) if cmd[0] == cmd[0] else "",
            "cmd_angular": round(cmd[1], 4) if cmd[1] == cmd[1] else "",
            "low_conf_streak": streak,
            "drift_rate": round(drift.rate, 5) if drift is not None else "",
        })

    bag.close()
    with open(args.csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)

    print("wrote %d scans to %s" % (len(rows), args.csv_path))
    if rows:
        dts = [r["cloud_dt_sec"] for r in rows if r["cloud_dt_sec"] != ""]
        if dts:
            print("cloud_dt_sec: min=%.3f max=%.3f (a value >0.3 means a scan stall)" % (min(dts), max(dts)))
        ages = [r["pose_age_sec"] for r in rows if r["pose_age_sec"] != ""]
        if ages:
            print("pose_age_sec: min=%.4f max=%.4f mean=%.4f" % (min(ages), max(ages), sum(ages) / len(ages)))
        first = next((r["scan_i"] for r in rows if r["state"] != "ACQUIRE"), None)
        print("first non-ACQUIRE scan: %s | final state: %s" % (first, rows[-1]["state"]))
        changes = [(r["scan_i"], r["state"]) for r in rows if r["state_changed"]]
        print("obstacle stop would have held the robot on %d scans (max in-path points %d)"
              % (sum(1 for r in rows if r["obstacle_blocked"]), max(r["path_points"] for r in rows)))
        print("state changes: %s" % changes)
        if drift is not None:
            print("drift correction: learned yaw drift %+.5f rad/s (%+.1f deg/min)%s"
                  % (drift.rate, math.degrees(drift.rate) * 60, "" if drift.learned else " -- NEVER LEARNED (robot never parked 2 s)"))
        if stall_resets:
            print("stall recovery: detector reset %d time(s) after a scan gap > %.1f s" % (stall_resets, args.stall_reset_sec))


if __name__ == "__main__":
    main()
