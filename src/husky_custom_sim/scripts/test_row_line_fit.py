#!/usr/bin/env python3
"""
Offline synthetic tests for row_line_fit.py -- no ROS, no robot required.

Run directly:
    python3 test_row_line_fit.py

Mirrors the spirit of test_row_confidence.py: build synthetic point clouds
in sensor-frame coordinates (x = forward, y = left), drive them through
RowLineFit across several simulated scans as the robot advances along a
straight world-frame corridor, and check the resulting fit against what we
know to be true by construction.
"""

import time

import numpy as np

from row_line_fit import RowLineFit

RNG = np.random.default_rng(seed=7)

ROW_HALF_WIDTH = 0.45   # m -- cotton row spacing ~0.9m -> rows at +/-0.45m
Z_MID = 0.35            # a z value safely inside the default canopy band


def make_scan(robot_x, robot_y=0.0, n_stems=14, x_range=(0.3, 4.0),
              y_jitter=0.02, row_half_width=ROW_HALF_WIDTH,
              drop_left=False, drop_right=False, n_outliers=0):
    """One scan's sensor-frame points for a straight world-frame corridor.

    World-frame rows sit at y = +row_half_width and y = -row_half_width for
    all x. robot_x/robot_y is the robot's current WORLD pose (theta=0, i.e.
    facing +x) -- sensor-frame points are world points shifted into the
    robot's frame. drop_left/drop_right simulate a gap (no returns on that
    side this scan). n_outliers adds uniform scatter anywhere in the ROI
    width, standing in for weeds.
    """
    pts = []
    if not drop_left:
        x = RNG.uniform(x_range[0], x_range[1], n_stems)
        y = (row_half_width - robot_y) + RNG.normal(0.0, y_jitter, n_stems)
        pts.append(np.column_stack((x, y, np.full(n_stems, Z_MID))))
    if not drop_right:
        x = RNG.uniform(x_range[0], x_range[1], n_stems)
        y = (-row_half_width - robot_y) + RNG.normal(0.0, y_jitter, n_stems)
        pts.append(np.column_stack((x, y, np.full(n_stems, Z_MID))))
    if n_outliers:
        x = RNG.uniform(x_range[0], x_range[1], n_outliers)
        y = RNG.uniform(-1.0, 1.0, n_outliers)
        pts.append(np.column_stack((x, y, np.full(n_outliers, Z_MID))))
    if not pts:
        return np.empty((0, 3))
    return np.vstack(pts)


results = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append(condition)
    print("[%s] %s%s" % (status, name, "  (%s)" % detail if detail else ""))


# --- 1. Clean straight corridor: both sides fit, near-zero error ------------
rlf = RowLineFit()
robot_x = 0.0
for _ in range(rlf.window_size):
    scan = make_scan(robot_x)
    rlf.update(scan, pose=(robot_x, 0.0, 0.0))
    robot_x += 0.02  # ~0.2 m/s at ~10 Hz, matches FORWARD_SPEED

left_fit, right_fit = rlf.fit_lines()
err = rlf.centerline_error((robot_x, 0.0, 0.0))
check("clean corridor -> both sides fit", left_fit["valid"] and right_fit["valid"],
      "left_inlier=%.2f right_inlier=%.2f" % (left_fit["inlier_frac"], right_fit["inlier_frac"]))
check("clean corridor, centered robot -> near-zero cross-track",
      err is not None and abs(err["cross_track"]) < 0.05, "err=%s" % err)
check("clean corridor, robot facing along row -> near-zero heading error",
      err is not None and abs(err["heading_error"]) < 0.05, "err=%s" % err)

# --- 2. Off-center robot -> cross-track reflects the true offset ------------
rlf = RowLineFit()
robot_x = 0.0
OFFSET = 0.15
for _ in range(rlf.window_size):
    scan = make_scan(robot_x, robot_y=OFFSET)
    rlf.update(scan, pose=(robot_x, OFFSET, 0.0))
    robot_x += 0.02
err = rlf.centerline_error((robot_x, OFFSET, 0.0))
check("off-center robot (0.15m) -> cross-track recovers the true offset",
      err is not None and abs(abs(err["cross_track"]) - OFFSET) < 0.05, "err=%s" % err)

# --- 3. Momentary gap (fewer scans than window_size) -> stays valid --------
rlf = RowLineFit()
robot_x = 0.0
for i in range(rlf.window_size):
    drop_l = (5 <= i < 8)  # a 3-scan gap on the left, well under window_size
    scan = make_scan(robot_x, drop_left=drop_l)
    rlf.update(scan, pose=(robot_x, 0.0, 0.0))
    robot_x += 0.02
check("momentary 3-scan gap -> any_valid() still True (not a false row-end)",
      rlf.any_valid())

# --- 4. Weed/outlier points -> RANSAC fit stays close to ground truth ------
rlf = RowLineFit()
robot_x = 0.0
for _ in range(rlf.window_size):
    scan = make_scan(robot_x, n_outliers=6)  # ~6 weed-like outliers per scan
    rlf.update(scan, pose=(robot_x, 0.0, 0.0))
    robot_x += 0.02
left_fit, right_fit = rlf.fit_lines()
# ground truth: both lines run parallel to world x-axis -> direction ~ (+/-1, 0)
left_angle = abs(np.degrees(np.arctan2(left_fit["direction"][1], left_fit["direction"][0])))
left_angle = min(left_angle, 180 - left_angle)  # direction sign is arbitrary
check("weed outliers -> RANSAC still recovers a near-horizontal row line",
      left_fit["valid"] and left_angle < 5.0,
      "direction angle=%.2f deg, inlier_frac=%.2f" % (left_angle, left_fit["inlier_frac"]))

# --- 5. Contrast: a concentrated weed clump (lever-arm outlier) ------------
# Uniform scattered noise doesn't reliably bias a least-squares slope -- a
# REAL weed clump does, because it's concentrated off to one side, which is
# exactly the case RANSAC exists to reject. Isolated single-scan comparison,
# independent of the k-means/temporal-window pipeline, to test the line-fit
# step itself head to head.
clean_x = RNG.uniform(0.3, 4.0, 14)
clean_y = ROW_HALF_WIDTH + RNG.normal(0.0, 0.02, 14)
clump_x = RNG.uniform(3.5, 4.0, 5)
clump_y = np.full(5, 1.0)  # a weed clump, well off the true row line
pts = np.column_stack((np.concatenate([clean_x, clump_x]), np.concatenate([clean_y, clump_y])))

lstsq_slope, _ = np.polyfit(pts[:, 0], pts[:, 1], 1)
lstsq_angle = abs(np.degrees(np.arctan(lstsq_slope)))

isolated = RowLineFit()
ransac_fit = isolated._ransac_line(pts)
ransac_angle = abs(np.degrees(np.arctan2(ransac_fit["direction"][1], ransac_fit["direction"][0])))
ransac_angle = min(ransac_angle, 180 - ransac_angle)  # direction sign is arbitrary

check("weed clump drags a naive least-squares fit off the true row line (true=0deg)",
      lstsq_angle > 5.0, "lstsq angle=%.2f deg" % lstsq_angle)
check("weed clump: RANSAC correctly rejects it, stays near the true line",
      ransac_fit["valid"] and ransac_angle < 3.0,
      "RANSAC angle=%.2f deg, inlier_frac=%.2f" % (ransac_angle, ransac_fit["inlier_frac"]))
check("RANSAC clearly outperforms naive least-squares on this outlier clump",
      ransac_angle < lstsq_angle - 3.0,
      "lstsq=%.2f deg vs RANSAC=%.2f deg" % (lstsq_angle, ransac_angle))

# --- 6. True row end: canopy goes empty for longer than window_size -------
rlf = RowLineFit()
robot_x = 0.0
for _ in range(rlf.window_size):  # build up a valid fit first
    scan = make_scan(robot_x)
    rlf.update(scan, pose=(robot_x, 0.0, 0.0))
    robot_x += 0.02
setup_ok = rlf.any_valid()
for _ in range(rlf.window_size + 2):  # now the row genuinely ends
    rlf.update(np.empty((0, 3)), pose=(robot_x, 0.0, 0.0))
    robot_x += 0.02
check("sanity: fit was valid before the row ended", setup_ok)
check("sustained empty canopy (> window_size scans) -> any_valid() goes False",
      not rlf.any_valid())

# --- 7. One-side fallback: only one row visible, centerline still estimable -
rlf = RowLineFit()
robot_x = 0.0
OFFSET2 = 0.15
for _ in range(rlf.window_size):
    scan = make_scan(robot_x, robot_y=OFFSET2, drop_right=True)
    rlf.update(scan, pose=(robot_x, OFFSET2, 0.0))
    robot_x += 0.02
err = rlf.centerline_error((robot_x, OFFSET2, 0.0))
check("one-side fallback: only left row visible, centerline_error is no longer None",
      err is not None, "err=%s" % err)
check("one-side fallback: cross-track still roughly matches the true 0.15m offset",
      err is not None and abs(abs(err["cross_track"]) - OFFSET2) < 0.08, "err=%s" % err)
check("one-side fallback: result is correctly flagged as estimated",
      err is not None and err["estimated"] is True, "err=%s" % err)

# --- 8. DBSCAN rejects an isolated outlier that k-means would absorb -------
# Dense, evenly-spaced rows so DBSCAN's default eps/min_samples reliably
# chain into one cluster per row -- this test isolates the clustering step
# itself, not the sparse-return realism the earlier tests use.
xs = np.linspace(0.3, 4.0, 40)
row_a = np.column_stack((xs, ROW_HALF_WIDTH + RNG.normal(0, 0.015, 40)))
row_b = np.column_stack((xs, -ROW_HALF_WIDTH + RNG.normal(0, 0.015, 40)))
lone_outlier = np.array([[2.0, 1.3]])  # far from both rows, alone -- not dense enough to cluster
pts2d = np.vstack([row_a, row_b, lone_outlier])

rlf_db = RowLineFit(overrides={"cluster_mode": "dbscan"})
left_db, right_db, ok_db = rlf_db.cluster_two_rows_dbscan(pts2d)
rlf_km = RowLineFit()
left_km, right_km, ok_km = rlf_km.cluster_two_rows(pts2d)

check("DBSCAN: still finds both rows with a lone outlier present", ok_db)
check("DBSCAN: left centroid unaffected by the outlier (stays near true 0.45m)",
      ok_db and abs(left_db[1] - ROW_HALF_WIDTH) < 0.03,
      "DBSCAN left centroid y=%.3f (true=%.2f)" % (left_db[1], ROW_HALF_WIDTH))
check("k-means: DOES get pulled toward the outlier (contrast, not a k-means bug)",
      ok_km and abs(left_km[1] - ROW_HALF_WIDTH) > abs(left_db[1] - ROW_HALF_WIDTH),
      "k-means y=%.3f vs DBSCAN y=%.3f (true=%.2f)" % (left_km[1], left_db[1], ROW_HALF_WIDTH))

# --- 9. Compute-cost benchmark: closing a previously-flagged unknown -------
# k-means (multiple restarts) + RANSAC (100 iterations) every scan at ~9.9Hz
# was flagged in action_plan.md as an unexamined real-time cost risk. This
# doesn't prove the ROS node will hit budget (message overhead, real point
# counts, and the rest of the control loop aren't modeled here), but it's a
# real number instead of a guess.
def benchmark(cluster_mode, n_scans=60):
    rlf_bench = RowLineFit(overrides={"cluster_mode": cluster_mode})
    bx = 0.0
    for _ in range(rlf_bench.window_size):  # fill the window so fit_lines() does real work
        rlf_bench.update(make_scan(bx, n_outliers=3), pose=(bx, 0.0, 0.0))
        bx += 0.02
    t0 = time.perf_counter()
    for _ in range(n_scans):
        rlf_bench.update(make_scan(bx, n_outliers=3), pose=(bx, 0.0, 0.0))
        rlf_bench.fit_lines()
        bx += 0.02
    return (time.perf_counter() - t0) / n_scans

budget_s = 1.0 / 9.9  # one scan period at the VLP-16's rate
for mode in ("kmeans", "dbscan"):
    avg_s = benchmark(mode)
    check("compute cost (%s) leaves headroom in the ~101ms/scan budget at 9.9Hz" % mode,
          avg_s < budget_s * 0.5,
          "%.2fms/scan (budget %.1fms)" % (avg_s * 1000, budget_s * 1000))

# --- 10. Yaw invariance: a straight wall at 8 deg yaw must still cluster (2026-10-28) --
# The spread gate used to measure y-spread in the ROBOT frame, so a perfectly
# straight wall seen at ~4+ deg yaw looked like a "blend" and blinded the detector.
import math as _math
def _wall_scan_yawed(yaw_deg, n=40):
    th = _math.radians(yaw_deg); pts = []
    for wy in (0.9, -0.9):
        wx = RNG.uniform(-0.5, 5.5, n); wyy = np.full(n, wy) + RNG.normal(0, 0.01, n)
        rx = _math.cos(-th) * wx - _math.sin(-th) * wyy
        ry = _math.sin(-th) * wx + _math.cos(-th) * wyy
        keep = (rx >= 0.3) & (rx <= 5.0)
        pts.append(np.column_stack((rx[keep], ry[keep], np.full(keep.sum(), Z_MID))))
    return np.vstack(pts)
for yaw in (0, 4, 8):
    rlf = RowLineFit(overrides={"roi_y_abs_max": 2.5}, seed=0)
    modes = [rlf.update(_wall_scan_yawed(yaw), pose=(0.0, 0.0, _math.radians(yaw)))["status"] for _ in range(10)]
    check("yaw invariance: straight walls at %d deg yaw still cluster both sides" % yaw,
          modes.count("both") >= 9, "modes=%s" % sorted(set(modes)))

# --- 11. Sign convention: heading_error is robot-relative-to-row, like cross_track (2026-10-28) --
rlf = RowLineFit(seed=0)
for k in range(rlf.window_size):
    rlf.update(make_scan(0.02 * k, robot_y=0.0), pose=(0.02 * k, 0.0, 0.0))
th = _math.radians(5.0)   # robot yawed 5 deg LEFT of a row that runs along +x
err = rlf.centerline_error((0.3, 0.0, th))
check("heading_error > 0 when the robot points left of the row", err is not None and err["heading_error"] > 0.05,
      "err=%s" % err)
err = rlf.centerline_error((0.3, 0.2, 0.0))
check("cross_track > 0 when the robot is left of the centerline", err is not None and err["cross_track"] > 0.1,
      "err=%s" % err)

# --- 12. One clean wall + one wall with a protrusion -> one_side, not none (2026-10-28) --
rlf = RowLineFit(seed=0); modes = []
for k in range(25):
    pts = make_scan(0.02 * k, robot_y=0.0)
    obst = np.column_stack((RNG.uniform(0.5, 1.5, 40), np.full(40, -0.35) + RNG.normal(0, 0.03, 40), np.full(40, Z_MID)))
    obst2 = np.column_stack((RNG.uniform(2.5, 3.5, 40), np.full(40, -0.30) + RNG.normal(0, 0.03, 40), np.full(40, Z_MID)))
    modes.append(rlf.update(np.vstack((pts, obst, obst2)), pose=(0.02 * k, 0.0, 0.0))["status"])
check("clean left wall + cluttered right wall -> detector keeps a usable side",
      modes[-10:].count("none") <= 2, "last-10 modes=%s" % modes[-10:])
check("...and the buffer still yields a valid fit on the clean side", rlf.any_valid())

# --- 13. fit cache: three queries per scan share one RANSAC run, results unchanged ---
rlf = RowLineFit(seed=0)
for k in range(rlf.window_size):
    rlf.update(make_scan(0.02 * k), pose=(0.02 * k, 0.0, 0.0))
a = rlf.fit_lines(); b = rlf.fit_lines()
check("fit_lines() is cached within a scan (same objects returned)", a is b)
rlf.update(make_scan(0.4), pose=(0.4, 0.0, 0.0))
check("fit cache is invalidated by update()", rlf.fit_lines() is not a)

# --- 14. clustering subsample cap keeps both sides at full density (2026-10-28) ---
rlf = RowLineFit(overrides={"max_cluster_points": 300}, seed=0)
dense = make_scan(0.0, n_stems=2000)            # ~4000 canopy points >> cap
modes = [rlf.update(dense, pose=(0.02 * k, 0.0, 0.0))["status"] for k in range(15)]
check("subsampling a dense scan to the cap still finds both rows", modes.count("both") >= 14, "modes=%s" % sorted(set(modes)))

# --- 15. travel baseline: ~0 when parked, grows with motion (2026-10-28) ---
rlf = RowLineFit(seed=0)
for _ in range(15): rlf.update(make_scan(0.0), pose=(0.0, 0.0, 0.0))
parked = rlf.baseline()
rlf = RowLineFit(seed=0)
for k in range(15): rlf.update(make_scan(0.02 * k), pose=(0.02 * k, 0.0, 0.0))
moving = rlf.baseline()
check("baseline is tiny while parked and above min_baseline_m after ~0.3 m of travel",
      parked < rlf.min_baseline_m and moving >= rlf.min_baseline_m, "parked=%.3f moving=%.3f" % (parked, moving))

# --- 16. cloud_xyz equals the pc2 conversion on a synthetic PointCloud2-shaped message ---
import struct, types
from frame_utils import cloud_xyz
_pts = np.array([[1.0, 2.0, 3.0], [np.nan, 0.0, 0.0], [4.5, -1.5, 0.25], [0.0, np.inf, 1.0]], dtype=np.float32)
_buf = b"".join(struct.pack("<ffff", x, y, z, 7.0) for x, y, z in _pts)     # x,y,z + intensity, point_step 16
_fields = [types.SimpleNamespace(name=n, offset=o, datatype=7, count=1) for n, o in (("x", 0), ("y", 4), ("z", 8), ("intensity", 12))]
_msg = types.SimpleNamespace(fields=_fields, point_step=16, width=4, height=1, data=_buf, is_bigendian=False)
_out = cloud_xyz(_msg)
check("cloud_xyz keeps finite points and drops NaN/inf rows exactly",
      _out.shape == (2, 3) and np.allclose(_out, [[1, 2, 3], [4.5, -1.5, 0.25]]), "out=%s" % _out.tolist())

# --- 17. Clean evenly-spread rows must split reliably even when sparse (2026-10-28) ---
# Legacy Euclidean k-means failed this split in ~48% of scans at 20 pts/side.
_bad = 0
for _ in range(200):
    _rl = RowLineFit(seed=0)
    if _rl.update(make_scan(0.0, n_stems=20), pose=(0, 0, 0))["status"] != "both":
        _bad += 1
check("20 evenly-spread points per side split into two rows in >=98% of scans (legacy: ~52%)",
      _bad <= 4, "failures=%d/200 (cluster_x_scale=%.2f)" % (_bad, RowLineFit().cluster_x_scale))

# --- Summary -----------------------------------------------------------------
n_pass = sum(1 for r in results if r)
print("\n%d/%d checks passed" % (n_pass, len(results)))
if n_pass != len(results):
    raise SystemExit(1)
