#!/usr/bin/env python3
"""
row_line_fit.py -- explicit row-line model from raw 3D LiDAR points.

WHY THIS EXISTS
---------------
row_confidence.py asks "does this look like a row?" with a single smoothed
number checked against a threshold. Ryan's literature research (2026-09-24;
see claude/action_plan.md, "Architecture Update") found that this class of
design -- threshold + dwell time -- is exactly what a real peer-reviewed
LiDAR row-detection pipeline avoids (Liu, Yandun, Kantor 2024, "Towards
Over-Canopy Autonomous Navigation," arXiv:2403.17774, verified against the
actual paper text, not just the abstract). That pipeline never waits for a
score to cross a line: every cycle it tries to fit an ACTUAL row model from
current data, and the absence of a valid fit -- not a smoothed score
dwelling low -- is the row-end signal.

This module is that model. Given raw 3D points (meant to come from
/velodyne_points, NOT the already-flattened /scan_fixed row_confidence.py
uses), it:

    1. slices to a canopy-height band (isolates plant-top returns from
       stalks/soil),
    2. splits the slice into a left-row cluster and a right-row cluster via
       k-means (k=2) -- adaptive to the robot's actual lateral offset, unlike
       a hard y>0/y<0 split,
    3. accumulates each side's cluster centroid over a short scan window,
       aligned into a shared frame using the robot's own odometry pose (so
       the robot's own motion doesn't get mistaken for row curvature),
    4. RANSAC-fits a line through each side's accumulated centroids --
       robust to weed/outlier points, unlike a plain least-squares fit,
    5. reports whether that fit is VALID (enough inliers, low residual).
       "no valid fit on either side" is the row-end signal, in place of
       row_confidence.py's "smoothed confidence stayed below 0.80."

A valid fit on both sides also produces an explicit line equation per row
edge -- the ingredient needed to steer on true geometric cross-track/heading
error against the row centerline (centerline_error()), instead of
row_follower_v2.py's current L_avg - R_avg mean-distance proxy. See
action_plan.md for why that's the bigger expected accuracy win.

STALENESS, NOT JUST WINDOW LENGTH
----------------------------------
A naive rolling buffer (append on success, evict oldest on overflow) has a
bug for row-end detection: if clustering starts failing every scan (a real
row end), nothing new gets pushed, so nothing gets evicted either -- the
buffer keeps quietly reporting a valid fit from data that's actually stale.
This module tracks a monotonically increasing scan counter and timestamps
every buffered centroid with it; fit_lines() drops anything older than
window_size scans *before* fitting, regardless of whether new data has
arrived to push it out. call update() every scan, whether or not clustering
succeeded that scan, so the counter (and therefore staleness) stays correct.

DELIBERATELY NO ROSPY IMPORT
-----------------------------
row_confidence.py imports rospy unconditionally at module level, which means
its own offline synthetic tests need a full ROS environment (WSL2 +
`source /opt/ros/noetic/setup.bash`) just to run pure-numpy checks. This
module stays plain numpy so it can be developed and tested with nothing but
`python3 test_row_line_fit.py`, anywhere, no ROS required. A thin ROS node
wrapper (subscribing to /velodyne_points and /odometry/filtered) is a
separate, later piece, once this algorithm is validated -- not needed to
validate the algorithm itself.

Run the offline tests:
    python3 test_row_line_fit.py

KNOWN SIMPLIFICATION
---------------------
k-means with k=2 assumes roughly compact (not too elongated/curved) clusters
per side within a single scan's ROI. This holds for the straight and gently
curved cases the reference paper tests, but a sharply curved row within one
ROI window could confuse it. Not a concern for the offline prototype; flag
for field validation.
"""

import time
from collections import deque

import numpy as np

_EPS = 1e-9


DEFAULTS = {
    # Canopy height band, sensor/base_link frame (z up). Cotton grows through
    # the season -- must be retunable without a code change. Same rationale
    # and starting values as row_confidence.py's z_min/z_max.
    "z_min": 0.05,
    "z_max": 0.70,

    # Forward-facing ROI, sensor frame (x = forward, y = left). Wider than
    # row_confidence.py's since k-means -- not a hard ROI wall -- does the
    # actual row separation; this box only excludes points too close to
    # trust or too far to be useful.
    "roi_x_min": 0.3,
    "roi_x_max": 5.0,
    "roi_y_abs_max": 1.2,

    "min_points_for_cluster": 12,   # total canopy points needed to attempt clustering
    "kmeans_iters": 25,
    "kmeans_restarts": 4,           # random restarts; keep the lowest-inertia result

    # Which clustering method splits the canopy slice into left/right.
    # "kmeans" forces every point into one of 2 clusters (including outliers).
    # "dbscan" can reject sparse/outlier points as noise before they ever
    # reach a cluster -- literature comparison found it handles irregular
    # field conditions better (see action_plan.md, 2026-09-27 entry). Not
    # yet the default: k-means is what the 2026-09-24 offline validation
    # ran against.
    "cluster_mode": "kmeans",       # "kmeans" or "dbscan"
    "dbscan_eps": 0.25,             # m -- neighborhood radius for density clustering
    "dbscan_min_samples": 4,        # points (incl. itself) required to be a core point

    # Temporal accumulation of centroids into the fit buffer.
    "window_size": 15,              # scans (~1.5s at ~9.9Hz) -- matches row_confidence.py

    # RANSAC line fit.
    "ransac_iters": 100,
    "ransac_inlier_thresh": 0.08,     # m -- perpendicular distance to count as inlier
    "ransac_min_inlier_frac": 0.6,    # fraction of buffered centroids that must agree
    "min_fit_points": 4,              # centroids needed (after staleness filtering) to fit

    # One-side fallback for centerline_error() -- half the expected row
    # spacing, used to estimate a centerline when only one side has a valid
    # fit. Cotton row spacing ~0.9m -> rows sit ~0.45m off center.
    "row_half_width": 0.45,

    # Reject a left/right split where both clusters land too close together
    # in y to plausibly be two different rows. Needed because k-means (and
    # DBSCAN, if it finds >=2 dense groups) is happy to split a SINGLE row
    # lengthwise into two chunks that each still look like a valid line to
    # RANSAC -- without this check, "only one row visible" silently produces
    # a fake second "row" instead of correctly reporting ok=False. Set well
    # below true row spacing (~0.9m -> ~0.45m each side) but above what two
    # lengthwise chunks of the same row would ever show.
    "min_cluster_separation_y": 0.3,

    # Reject a cluster whose OWN y-spread is too big to be one row's
    # plant-to-plant jitter. Found necessary via testing (2026-09-28): the
    # separation check above isn't enough on its own -- a k-means split that
    # cuts across BOTH rows in an uneven ratio (some near-side points grouped
    # with some far-side points) produces two "blended" clusters whose means
    # can happen to land more than min_cluster_separation_y apart purely by
    # chance, even though neither cluster is a real row. A real row's points
    # stay tightly banded in y; a blended cluster's y-spread is elevated
    # because it's really a mix of two different y populations.
    #
    # Measured as scaled median absolute deviation (MAD), NOT plain std --
    # also found via testing. Plain std is outlier-sensitive, which fights
    # the whole point of running RANSAC downstream: a handful of real weed
    # points swept into an otherwise-clean row cluster inflates std past any
    # reasonable threshold even though the cluster is still fine (RANSAC
    # rejects those weeds just fine on its own). MAD stays small as long as
    # the *majority* of a cluster's points are tightly banded -- a few
    # outliers barely move it -- while a genuine ~50/50 mix of two rows
    # still shows a large MAD, since even the median splits the two
    # populations. Set well above realistic plant-jitter noise but well
    # below what mixing two rows ~0.9m apart would produce.
    "max_cluster_y_spread": 0.12,

    # Reject a cluster whose points don't span most of the canopy slice's
    # forward (x) range. Found necessary via testing (2026-09-28) as a gap
    # in max_cluster_y_spread: an UNEVEN blend (say 9 real left-row points +
    # 5 real right-row points landing in one cluster, instead of a clean
    # 50/50 split) still has a small MAD, because MAD only measures whether
    # the MAJORITY is tight -- it can't tell "5 genuine outliers" from "5
    # points that are actually the other row." The reliable signature is
    # geometric, not statistical: a k-means split that cuts wrongly along x
    # instead of by row confines each resulting cluster to roughly HALF the
    # forward distance range (a "near" half and a "far" half), while a real
    # row's points are spread along its entire visible length. Expressed as
    # a fraction of the full canopy slice's x-range this scan.
    "min_cluster_x_coverage": 0.6,
}


def _wrap_angle(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _robust_spread(values):
    """Scaled median absolute deviation -- an outlier-robust stand-in for
    std(). The 1.4826 factor makes it comparable to a standard deviation for
    normally-distributed data. See max_cluster_y_spread in DEFAULTS for why
    plain std() is the wrong tool here."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)))


class RowLineFit:
    """Rolling k-means + RANSAC row-line model. See module docstring.

    Typical use inside a controller::

        rlf = RowLineFit()
        ...
        def on_scan(points_xyz, odom_pose):
            rlf.update(points_xyz, odom_pose)
            if not rlf.is_row_present():
                ...                      # feed into the existing state machine's
                                          # low-confidence streak, same as
                                          # row_confidence.py's instant_conf did
            err = rlf.centerline_error(odom_pose)
            if err is not None:
                ...                      # steer on err["cross_track"] / ["heading_error"]
    """

    def __init__(self, overrides=None, seed=None):
        """overrides: dict of DEFAULTS keys to override.

        seed: RNG seed for the internal k-means/RANSAC randomness. Pass a
        fixed int for reproducible tests. Left as None (the default), this
        seeds from time.time_ns() rather than letting numpy pull from OS
        entropy -- found necessary via testing (2026-09-28): an unseeded
        default_rng() occasionally took a long time to return on this dev
        machine, almost certainly an OS entropy-pool stall, not a numpy bug.
        The same failure mode is a real (if rare) risk on embedded/robot
        hardware, where /dev/urandom-style sources can block early after
        boot when the entropy pool hasn't filled yet. None of this needs
        cryptographic randomness -- RANSAC/k-means just need practical
        variety between restarts -- so there's no reason to risk a stall for
        it, here or in the field.
        """
        cfg = dict(DEFAULTS)
        if overrides:
            cfg.update(overrides)
        for key, value in cfg.items():
            setattr(self, key, value)

        if self.cluster_mode not in ("kmeans", "dbscan"):
            print("RowLineFit: cluster_mode=%r not recognised, using 'kmeans'" % self.cluster_mode)
            self.cluster_mode = "kmeans"

        self._rng = np.random.default_rng(seed if seed is not None else time.time_ns() % (2**32))
        self._left_buf = deque(maxlen=self.window_size)   # list of (scan_i, xy)
        self._right_buf = deque(maxlen=self.window_size)
        self._scan_i = 0
        self.last_debug = {}

    def reset(self):
        """Drop accumulated history (call when leaving FOLLOWING / re-acquiring)."""
        self._left_buf.clear()
        self._right_buf.clear()
        self._scan_i = 0

    # -- pipeline stage 1: canopy slice --------------------------------------

    def slice_canopy(self, pts_xyz):
        """Height-band + forward ROI filter on raw (N, 3) sensor-frame points."""
        pts = np.asarray(pts_xyz, dtype=float)
        if pts.ndim != 2 or pts.shape[1] < 3 or pts.shape[0] == 0:
            return np.empty((0, 3))
        finite = np.all(np.isfinite(pts[:, :3]), axis=1)
        band = (pts[:, 2] >= self.z_min) & (pts[:, 2] <= self.z_max)
        roi = (
            (pts[:, 0] >= self.roi_x_min) & (pts[:, 0] <= self.roi_x_max)
            & (np.abs(pts[:, 1]) <= self.roi_y_abs_max)
        )
        return pts[finite & band & roi]

    # -- pipeline stage 2: k-means split into left/right ---------------------

    def _single_row_centroid(self, pts):
        """Check whether pts (taken all together) look like ONE coherent row
        -- tight in y, spanning most of the configured ROI depth.

        Needed because k-means and DBSCAN each have their own way of being
        unable to say "I only found one real row": k-means is forced to
        always produce exactly 2 clusters (see the GOTCHA in
        cluster_two_rows), and DBSCAN might legitimately find only 0 or 1
        dense group. Either way, this checks the whole slice directly
        instead of trying to coax a second cluster out of nothing. Returns
        the centroid, or None if it doesn't look like a single real row
        (e.g. it's scattered noise instead).
        """
        if _robust_spread(pts[:, 1]) > self.max_cluster_y_spread:
            return None
        roi_span = self.roi_x_max - self.roi_x_min
        x_range = pts[:, 0].max() - pts[:, 0].min()
        if x_range < self.min_cluster_x_coverage * roi_span:
            return None
        return pts.mean(axis=0)

    def cluster_two_rows(self, pts_xy):
        """K-means (k=2) split of a canopy slice into left/right row clusters.

        Returns (left_centroid_or_None, right_centroid_or_None, status),
        where status is "both", "one_side", or "none". Seeds the first
        attempt from sign(y) -- usually already close to the true split --
        then tries a few random-pair seeds, keeping whichever converged run
        has the lowest inertia. Protects against a bad seed collapsing both
        means onto one row when returns are sparse.

        GOTCHA (found via testing, 2026-09-28): plain lowest-inertia
        selection is not safe here. A single row is far more spread out
        along x (its own length) than across y (its width), so splitting
        purely by x -- cutting the row into a "near" and "far" half instead
        of separating it from the other row -- can have LOWER total k-means
        inertia than the correct split, even though it's structurally wrong.
        A restart that stumbles onto that split will win on inertia alone.
        Candidates are therefore only considered if their two centroids are
        at least min_cluster_separation_y apart in y -- a real second row,
        not the same row cut lengthwise.
        """
        pts = np.asarray(pts_xy, dtype=float)
        if pts.shape[0] < self.min_points_for_cluster:
            return None, None, "none"

        full_x_range = pts[:, 0].max() - pts[:, 0].min()
        min_x_range = self.min_cluster_x_coverage * full_x_range

        best_centroids, best_inertia = None, np.inf

        for attempt in range(self.kmeans_restarts):
            if attempt == 0:
                has_pos = np.any(pts[:, 1] >= 0)
                has_neg = np.any(pts[:, 1] < 0)
                c0 = pts[pts[:, 1] >= 0].mean(axis=0) if has_pos else pts[0]
                c1 = pts[pts[:, 1] < 0].mean(axis=0) if has_neg else pts[-1]
                centroids = np.vstack([c0, c1])
            else:
                idx = self._rng.choice(pts.shape[0], size=2, replace=False)
                centroids = pts[idx].copy()

            for _ in range(self.kmeans_iters):
                d0 = np.linalg.norm(pts - centroids[0], axis=1)
                d1 = np.linalg.norm(pts - centroids[1], axis=1)
                assign0 = d0 <= d1
                if not np.any(assign0) or np.all(assign0):
                    break  # collapsed onto one cluster -- abandon this restart
                new_centroids = np.vstack([pts[assign0].mean(axis=0), pts[~assign0].mean(axis=0)])
                converged = np.allclose(new_centroids, centroids)
                centroids = new_centroids
                if converged:
                    break

            d0 = np.linalg.norm(pts - centroids[0], axis=1)
            d1 = np.linalg.norm(pts - centroids[1], axis=1)
            assign0 = d0 <= d1
            if not np.any(assign0) or np.all(assign0):
                continue  # this restart collapsed -- skip it

            if abs(centroids[0][1] - centroids[1][1]) < self.min_cluster_separation_y:
                continue  # structurally implausible as two different rows -- see GOTCHA above
            if (_robust_spread(pts[assign0, 1]) > self.max_cluster_y_spread
                    or _robust_spread(pts[~assign0, 1]) > self.max_cluster_y_spread):
                continue  # a real row is tightly banded in y; this candidate is a blend of both
            x_range0 = pts[assign0, 0].max() - pts[assign0, 0].min() if np.any(assign0) else 0.0
            x_range1 = pts[~assign0, 0].max() - pts[~assign0, 0].min() if np.any(~assign0) else 0.0
            if x_range0 < min_x_range or x_range1 < min_x_range:
                continue  # confined to half the forward range -- a wrong-axis (x) split

            inertia = (
                np.sum(d0[assign0] ** 2) + np.sum(d1[~assign0] ** 2)
            )
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids.copy()

        if best_centroids is None:
            single = self._single_row_centroid(pts)
            if single is None:
                return None, None, "none"
            return (single, None, "one_side") if single[1] >= 0 else (None, single, "one_side")

        # sensor frame REP-103: y = left. Larger y -> the left row.
        if best_centroids[0][1] >= best_centroids[1][1]:
            left, right = best_centroids[0], best_centroids[1]
        else:
            left, right = best_centroids[1], best_centroids[0]
        return left, right, "both"

    def _dbscan(self, pts, eps, min_samples):
        """Plain-numpy DBSCAN. Returns an (N,) int array of cluster labels,
        -1 for noise. O(n^2) pairwise distance -- fine at the point counts a
        single scan's canopy slice produces (tens, not thousands)."""
        n = pts.shape[0]
        if n == 0:
            return np.empty(0, dtype=int)
        dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        neighbor_idx = [np.where(dist[i] <= eps)[0] for i in range(n)]

        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            if len(neighbor_idx[i]) < min_samples:
                continue  # not a core point -- left as noise unless reached later
            labels[i] = cluster_id
            seeds = list(neighbor_idx[i])
            k = 0
            while k < len(seeds):
                j = seeds[k]
                k += 1
                if not visited[j]:
                    visited[j] = True
                    if len(neighbor_idx[j]) >= min_samples:
                        seeds.extend(neighbor_idx[j])
                if labels[j] == -1:
                    labels[j] = cluster_id
            cluster_id += 1
        return labels

    def cluster_two_rows_dbscan(self, pts_xy):
        """DBSCAN split of a canopy slice into left/right row clusters.

        Unlike k-means, points that don't belong near any dense group (a
        weed, a stray reflection) can end up labelled noise (-1) and never
        touch either row estimate at all -- rejected at the clustering stage
        itself, before RANSAC ever sees them. Returns (left_centroid_or_None,
        right_centroid_or_None, status), same interface as
        cluster_two_rows(): status is "both", "one_side", or "none".
        """
        pts = np.asarray(pts_xy, dtype=float)
        if pts.shape[0] < self.min_points_for_cluster:
            return None, None, "none"

        labels = self._dbscan(pts, self.dbscan_eps, self.dbscan_min_samples)
        unique = [lbl for lbl in set(labels.tolist()) if lbl != -1]
        if len(unique) < 2:
            # 0 or 1 dense cluster found. Unlike k-means, DBSCAN isn't forced
            # to invent a second one -- but it also can't tell us "this one
            # cluster IS a real row" on its own, so check it the same way
            # k-means' one-sided fallback does.
            candidate = pts[labels == unique[0]] if unique else pts
            single = self._single_row_centroid(candidate)
            if single is None:
                return None, None, "none"
            return (single, None, "one_side") if single[1] >= 0 else (None, single, "one_side")

        counts = sorted(
            ((lbl, int(np.sum(labels == lbl))) for lbl in unique),
            key=lambda t: t[1], reverse=True,
        )
        group0 = pts[labels == counts[0][0]]
        group1 = pts[labels == counts[1][0]]
        if (_robust_spread(group0[:, 1]) > self.max_cluster_y_spread
                or _robust_spread(group1[:, 1]) > self.max_cluster_y_spread):
            return None, None, "none"
        full_x_range = pts[:, 0].max() - pts[:, 0].min()
        min_x_range = self.min_cluster_x_coverage * full_x_range
        x_range0 = group0[:, 0].max() - group0[:, 0].min()
        x_range1 = group1[:, 0].max() - group1[:, 0].min()
        if x_range0 < min_x_range or x_range1 < min_x_range:
            return None, None, "none"
        c0, c1 = group0.mean(axis=0), group1.mean(axis=0)

        if c0[1] >= c1[1]:
            left, right = c0, c1
        else:
            left, right = c1, c0

        if abs(left[1] - right[1]) < self.min_cluster_separation_y:
            return None, None, "none"
        return left, right, "both"

    # -- pipeline stage 3: accumulate into a shared frame --------------------

    def update(self, pts_xyz, pose=(0.0, 0.0, 0.0)):
        """Feed one scan's raw 3D points (sensor frame) + the robot's current
        (x, y, theta) pose. Always advances the internal scan counter, even
        when clustering fails, so staleness tracking stays correct -- call
        this every scan, not just the successful ones.
        """
        self._scan_i += 1
        canopy = self.slice_canopy(pts_xyz)
        if canopy.shape[0]:
            if self.cluster_mode == "dbscan":
                left_c, right_c, status = self.cluster_two_rows_dbscan(canopy[:, :2])
            else:
                left_c, right_c, status = self.cluster_two_rows(canopy[:, :2])
        else:
            left_c, right_c, status = None, None, "none"

        debug = {
            "scan_i": self._scan_i, "n_canopy": int(canopy.shape[0]),
            "status": status, "clustered": status != "none",
        }
        x, y, theta = pose
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        if left_c is not None:
            self._left_buf.append((self._scan_i, rot.dot(left_c) + np.array([x, y])))
            debug["left_local"] = left_c
        if right_c is not None:
            self._right_buf.append((self._scan_i, rot.dot(right_c) + np.array([x, y])))
            debug["right_local"] = right_c

        self.last_debug = debug
        return debug

    def _fresh_points(self, buf):
        """Entries from buf newer than window_size scans ago, as an (N, 2) array."""
        cutoff = self._scan_i - self.window_size
        pts = [xy for (i, xy) in buf if i > cutoff]
        return np.array(pts) if pts else np.empty((0, 2))

    # -- pipeline stage 4: RANSAC line fit ------------------------------------

    def _ransac_line(self, pts):
        """Fit a 2D line to pts via RANSAC.

        Returns a dict: valid, point (a point on the line), direction (unit
        vector), inlier_frac, residual_rms, n.
        """
        n = pts.shape[0]
        if n < self.min_fit_points:
            return {"valid": False, "reason": "too few points", "n": n}

        best = None
        for _ in range(self.ransac_iters):
            i, j = self._rng.choice(n, size=2, replace=False)
            p0, p1 = pts[i], pts[j]
            d = p1 - p0
            norm = np.linalg.norm(d)
            if norm < _EPS:
                continue
            d = d / norm
            normal = np.array([-d[1], d[0]])
            dist = np.abs((pts - p0).dot(normal))
            inliers = dist <= self.ransac_inlier_thresh
            count = int(np.sum(inliers))
            if best is None or count > best["count"]:
                best = {"count": count, "inliers": inliers}

        if best is None or best["count"] < 2:
            return {"valid": False, "reason": "degenerate sample", "n": n}

        inlier_pts = pts[best["inliers"]]
        mean = inlier_pts.mean(axis=0)
        centered = inlier_pts - mean
        # Principal direction of the inliers, via SVD, is the least-squares
        # line direction through them -- refit on inliers only, same spirit
        # as standard RANSAC (initial 2-point sample just finds consensus).
        _, _, vt = np.linalg.svd(centered)
        direction = vt[0]
        normal = np.array([-direction[1], direction[0]])
        residuals = centered.dot(normal)
        residual_rms = float(np.sqrt(np.mean(residuals ** 2))) if len(residuals) else float("inf")
        inlier_frac = best["count"] / n

        return {
            "valid": bool(inlier_frac >= self.ransac_min_inlier_frac),
            "point": mean,
            "direction": direction,
            "inlier_frac": float(inlier_frac),
            "residual_rms": residual_rms,
            "n": n,
        }

    def fit_lines(self):
        """RANSAC-fit the left and right accumulated buffers, after dropping
        stale entries. Returns (left_fit, right_fit) dicts."""
        left_fit = self._ransac_line(self._fresh_points(self._left_buf))
        right_fit = self._ransac_line(self._fresh_points(self._right_buf))
        return left_fit, right_fit

    def any_valid(self):
        """True when at least one side has a valid fit right now.

        This is the row-end signal: False means neither side currently has
        a believable row model -- feed this into the SAME dwell-based state
        machine row_follower_v2.py already has (CREEPING's low_conf_streak),
        just as the per-cycle boolean instead of row_confidence.py's
        instant_conf < threshold. The dwell/streak logic for telling a real
        gap from a real row-end doesn't go away -- it's still a real,
        unavoidable problem -- but the per-cycle signal feeding it is now a
        genuine geometric fit instead of a biased scalar score.
        """
        left_fit, right_fit = self.fit_lines()
        return left_fit["valid"] or right_fit["valid"]

    def both_valid(self):
        """True only when BOTH sides have a valid fit -- a stricter check,
        analogous to row_confidence.py's confidence_threshold gate for
        entering FOLLOWING with full confidence in the centerline."""
        left_fit, right_fit = self.fit_lines()
        return left_fit["valid"] and right_fit["valid"]

    # -- pipeline stage 5: the steering payoff --------------------------------

    def _average_line(self, left_fit, right_fit, robot_fwd):
        """Average two valid fits into one centerline (direction, anchor)."""
        d_l, d_r = left_fit["direction"], right_fit["direction"]
        if d_l.dot(d_r) < 0:
            d_r = -d_r  # SVD direction sign is arbitrary; align before averaging
        direction = d_l + d_r
        direction = direction / np.linalg.norm(direction)
        if direction.dot(robot_fwd) < 0:
            direction = -direction  # resolve the remaining forward/backward ambiguity
        anchor = 0.5 * (left_fit["point"] + right_fit["point"])
        return direction, anchor

    def centerline_error(self, robot_pose):
        """Geometric cross-track + heading error of robot_pose=(x, y, theta)
        against the row centerline.

        This is the piece that would replace row_follower_v2.py's
        L_avg - R_avg proxy with a true geometric error. Three cases:

        - both sides valid: centerline is the midline between the two
          fitted lines (most trustworthy).
        - one side valid: the centerline is ESTIMATED by offsetting the
          single visible row inward by row_half_width. Real fields often
          have one row sparser than the other -- row_confidence.py already
          tolerates this via one_side_penalty; this was a real gap here
          before (see action_plan.md, 2026-09-27 entry). The returned dict
          has "estimated": True so a caller can weight it differently (e.g.
          fine to keep using once already FOLLOWING, not trustworthy enough
          to enter FOLLOWING on).
        - neither side valid: returns None.
        """
        left_fit, right_fit = self.fit_lines()
        x, y, theta = robot_pose
        pos = np.array([x, y])
        robot_fwd = np.array([np.cos(theta), np.sin(theta)])

        if left_fit["valid"] and right_fit["valid"]:
            direction, anchor = self._average_line(left_fit, right_fit, robot_fwd)
            estimated = False
        elif left_fit["valid"] or right_fit["valid"]:
            fit = left_fit if left_fit["valid"] else right_fit
            # sensor frame REP-103: the left row sits at +y relative to
            # center, the right row at -y -- side_sign picks which way
            # "inward" is for whichever single row we actually have.
            side_sign = 1.0 if left_fit["valid"] else -1.0
            direction = fit["direction"]
            if direction.dot(robot_fwd) < 0:
                direction = -direction
            normal = np.array([-direction[1], direction[0]])
            anchor = fit["point"] - side_sign * self.row_half_width * normal
            estimated = True
        else:
            return None

        normal = np.array([-direction[1], direction[0]])
        cross_track = float((pos - anchor).dot(normal))
        row_heading = float(np.arctan2(direction[1], direction[0]))
        heading_error = _wrap_angle(row_heading - theta)

        return {"cross_track": cross_track, "heading_error": heading_error, "estimated": estimated}
