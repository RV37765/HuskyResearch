#!/usr/bin/env python3
"""
report_card.py -- one-screen summary of a replay_bag_log.py CSV.

WHY THIS EXISTS
---------------
A run's bag is ~1 GB, and WiFi scp moved ~1.3 MB/s on 2026-10-28, so copying
it to a laptop before deciding the next run costs 10+ minutes. The replay CSV
is a few MB and this summary is a screenful of text. Run both ON THE ROBOT
right after a run and paste the output; copy the bag itself later.

    cd ~/HuskyRealWork/src/husky_custom_sim/scripts
    python3 replay_bag_log.py ~/bags/run.bag ~/bags/run.csv
    python3 report_card.py ~/bags/run.csv

If the run used non-default gains, give them to the replay AND the cap here:
    python3 replay_bag_log.py run.bag run.csv --k-lateral 0.6 --k-heading 0.7 --max-angular 0.15
    python3 report_card.py run.csv --max-angular 0.15

Plain Python (csv module only) -- runs on the robot, in WSL or on Windows.

Every number here is what the DETECTOR believed. Real accuracy needs a tape
measure; the PAUSES section prints the detector's cross-track and heading just
before each stop, to compare against tape readings taken at that stop.
"""

import argparse
import csv
import math

MOVING_SPEED = 0.05   # m/s; at or above this the robot counts as driving
PAUSE_SPEED = 0.01    # m/s; below this it counts as stopped
PAUSE_MIN_SEC = 2.0   # shorter stops are not listed
PRE_PAUSE_SCANS = 10  # ~1 s of scans before a stop: the detector's belief at that stop
STALL_SEC = 0.3       # a gap between clouds longer than this is a scan stall
SETTLED_M = 0.05      # |cross-track| below this counts as on the centerline
STEP_START_M = 0.15   # a run starting at least this far off-center gets a STEP line
STEP_WINDOW_M = 6.0   # overshoot is looked for this far past the first centerline crossing
OPEN_MIN_M = 1.0      # a hole this long in one side's points, row resuming beyond, is an opening
OPEN_MERGE_SCANS = 5  # sightings of an opening this few scans apart are the same opening


def _num(cell):
    """CSV cell -> float, or None for an empty cell."""
    return float(cell) if cell not in ("", None) else None


def _quantile(vals, q):
    """Linearly interpolated quantile (numpy's default method) of a non-empty list."""
    s = sorted(vals)
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _pct(n, d):
    return 100.0 * n / d if d else 0.0


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    s = 0.0
    for i, r in enumerate(rows):
        r["t"] = float(r["t_sec"])
        r["x"], r["y"] = float(r["odom_x"]), float(r["odom_y"])
        if i:
            s += math.hypot(r["x"] - rows[i - 1]["x"], r["y"] - rows[i - 1]["y"])
        r["s"] = s  # distance driven so far, by odometry
        r["xt"] = _num(r["cross_track_m"])
        r["he"] = _num(r["heading_error_rad"])
        r["ang"] = float(r["replay_angular"])
        r["cmd_lin"] = _num(r["cmd_linear"])
        r["cmd_ang"] = _num(r["cmd_angular"])
        r["dt"] = _num(r["cloud_dt_sec"])
        r["left"] = r["left_valid"] == "True"
        r["right"] = r["right_valid"] == "True"
        r["est"] = r["estimated_one_sided"] == "True"
        # CSVs from before the obstacle stop (2026-10-29) don't have these two.
        r["blocked"] = r.get("obstacle_blocked") == "True"
        r["path"] = int(r.get("path_points") or 0)
        # CSVs from before 2026-11-06 don't have the opening columns.
        for side in ("left", "right"):
            r["open_" + side] = _num(r.get("open_%s_m" % side)) or 0.0
            r["open_from_" + side] = _num(r.get("open_%s_from" % side))
            r["open_to_" + side] = _num(r.get("open_%s_to" % side))
        # Unwrapped odometry yaw, so changes across +/-180 deg add up correctly.
        th = float(r["odom_theta_rad"])
        r["thu"] = th if not i else rows[i - 1]["thu"] + math.atan2(
            math.sin(th - rows[i - 1]["thu"]), math.cos(th - rows[i - 1]["thu"]))
    return rows


def speeds(rows, half=0.5):
    """Speed at each scan from odometry, averaged over +/- half seconds."""
    out, k, j, n = [], 0, 0, len(rows)
    for i in range(n):
        t = rows[i]["t"]
        while rows[k]["t"] < t - half:
            k += 1
        while j + 1 < n and rows[j + 1]["t"] <= t + half:
            j += 1
        span = rows[j]["t"] - rows[k]["t"]
        out.append((rows[j]["s"] - rows[k]["s"]) / span if span > 0 else 0.0)
    return out


def segments(flags):
    """(first, last) index pairs of the contiguous True stretches in flags."""
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def merge(segs, max_gap):
    """Join (first, last) segments separated by at most max_gap indices."""
    out = []
    for a, b in segs:
        if out and a - out[-1][1] - 1 <= max_gap:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def pause_cause(rows, a, b):
    """Best guess at why the robot stopped, from what the follower was doing."""
    if any(r["state"] == "ROW_END" for r in rows[a:a + 5]):
        return "ROW_END"
    if any(r["blocked"] for r in rows[max(0, a - 10):a + 5]):
        return "obstacle stop"
    if b == len(rows) - 1:
        # Ctrl-C on the node looks like an override (its last command lingers in the bag).
        return "stopped till bag end"
    lin = [r["cmd_lin"] for r in rows[a:b + 1] if r["cmd_lin"] is not None]
    if lin and sum(lin) / len(lin) > 0.01:
        # The follower kept commanding forward speed but the robot sat still.
        return "override (L1 or e-stop)"
    return "follower sent 0 (pause?)"


def report(rows, max_angular):
    t0 = rows[0]["t"]
    v = speeds(rows)
    n = len(rows)

    print("RUN        %d scans, %.1f s, %.1f m driven (odometry)" % (n, rows[-1]["t"] - t0, rows[-1]["s"]))
    stalls = [(r["dt"], r["t"] - t0) for r in rows if r["dt"] is not None and r["dt"] > STALL_SEC]
    print("           scan stalls (> %.1f s between clouds): %d%s" % (
        STALL_SEC, len(stalls), ", worst %.2f s at t=%.1f s" % max(stalls) if stalls else ""))

    # -- states ---------------------------------------------------------------
    per_state = {}
    for a, b in zip(rows, rows[1:]):
        per_state[a["state"]] = per_state.get(a["state"], 0.0) + (b["t"] - a["t"])
    print("STATES     " + " | ".join("%s %.1f s" % kv for kv in per_state.items()))
    print("           t=%6.1f s  s=%5.1f m  start in %s" % (0.0, 0.0, rows[0]["state"]))
    for r in rows:
        if r["state_changed"] == "True":
            print("           t=%6.1f s  s=%5.1f m  -> %s" % (r["t"] - t0, r["s"], r["state"]))

    # -- detection --------------------------------------------------------------
    fol = [r for r in rows if r["state"] == "FOLLOWING"]
    if not fol:
        print("DETECTION  never reached FOLLOWING")
    else:
        both = sum(1 for r in fol if r["left"] and r["right"])
        one = sum(1 for r in fol if r["left"] != r["right"])
        print("DETECTION  in FOLLOWING (%d scans): both sides %.0f%% | one side %.0f%% | neither %.0f%%"
              % (len(fol), _pct(both, len(fol)), _pct(one, len(fol)), _pct(len(fol) - both - one, len(fol))))

    # -- openings: a hole in one side with that side resuming beyond (2026-11-06) ----
    # The detector bridges these on purpose (a missing plant is not a row end); this
    # shows that it SAW each one, and what the robot was doing while passing it.
    # Path position: s + the hole's robot-frame x. The far edge stays in view, so
    # s + to holds still; the near edge is clipped once it passes the ROI start, so
    # the earliest s + from is its true position.
    if "open_left_m" in rows[0]:
        events = []
        for side in ("left", "right"):
            flags = [r["open_" + side] >= OPEN_MIN_M for r in rows]
            events += [(a, b, side) for a, b in merge(segments(flags), OPEN_MERGE_SCANS)]
        print("OPENINGS   %d hole(s) of %.1f m or more in one side, with that side resuming beyond"
              % (len(events), OPEN_MIN_M))
        for a, b, side in sorted(events):
            ev = [r for r in rows[a:b + 1] if r["open_" + side] >= OPEN_MIN_M]
            start = min(r["s"] + r["open_from_" + side] for r in ev)
            end = _quantile([r["s"] + r["open_to_" + side] for r in ev], 0.5)
            states = {}
            for r in rows[a:b + 1]:
                states[r["state"]] = states.get(r["state"], 0) + 1
            print("           %-5s %.1f m wide at s=%5.1f-%5.1f m | in view %.1f m of travel | %s | %s line valid %.0f%%, one-sided %.0f%%"
                  % (side, end - start, start, end, rows[b]["s"] - rows[a]["s"],
                     " ".join("%s %.0f%%" % (k, _pct(c, b - a + 1)) for k, c in states.items()),
                     side, _pct(sum(1 for r in rows[a:b + 1] if r[side]), b - a + 1),
                     _pct(sum(1 for r in rows[a:b + 1] if r["est"]), b - a + 1)))

    # -- tracking (detector's belief) -------------------------------------------
    trk = [r for i, r in enumerate(rows)
           if r["state"] == "FOLLOWING" and v[i] >= MOVING_SPEED and r["xt"] is not None]
    if trk:
        xts = [r["xt"] * 100.0 for r in trk]
        axt = [abs(x) for x in xts]
        hes = [math.degrees(r["he"]) for r in trk]
        ahe = [abs(h) for h in hes]
        print("TRACKING   FOLLOWING and moving, %d scans (detector's belief, not ground truth)" % len(trk))
        print("           cross-track: mean %+.1f cm | RMS %.1f | median |xt| %.1f | p95 %.1f | max %.1f cm"
              % (sum(xts) / len(xts), math.sqrt(sum(x * x for x in xts) / len(xts)),
                 _quantile(axt, 0.5), _quantile(axt, 0.95), max(axt)))
        print("           heading:     mean %+.1f deg | median |he| %.1f | p95 %.1f deg | one-sided estimates %.0f%%"
              % (sum(hes) / len(hes), _quantile(ahe, 0.5), _quantile(ahe, 0.95),
                 _pct(sum(1 for r in trk if r["est"]), len(trk))))

    # -- step response: only for runs started deliberately off-center (gain sweep)
    moving = [r for i, r in enumerate(rows) if v[i] >= MOVING_SPEED and r["xt"] is not None]
    if moving and abs(moving[0]["xt"]) >= STEP_START_M:
        first = moving[0]
        x0 = first["xt"]
        hit = next((r for r in moving if abs(r["xt"]) < SETTLED_M), None)
        overshoot = 0.0
        if hit is not None:
            after = [r["xt"] for r in moving if 0.0 <= r["s"] - hit["s"] <= STEP_WINDOW_M]
            overshoot = max([0.0] + [abs(x) for x in after if x * x0 < 0])
        print("STEP       started %+.1f cm off; |xt| < %.0f cm %s; overshoot %.1f cm (within %.0f m)"
              % (x0 * 100.0, SETTLED_M * 100.0,
                 "after %.1f s / %.1f m" % (hit["t"] - first["t"], hit["s"] - first["s"]) if hit else "never",
                 overshoot * 100.0, STEP_WINDOW_M))

    # -- steering (driving scans only: a parked robot's commands mean nothing) -----
    drive = [r for i, r in enumerate(rows) if r["state"] == "FOLLOWING" and v[i] >= MOVING_SPEED]
    if drive:
        angs = [r["ang"] for r in drive]
        mags = [abs(a) for a in angs]
        # A steady nonzero signed mean is the controller leaning against something:
        # a robot that pulls to one side, or a biased heading estimate.
        print("STEERING   while driving: mean |angular| %.4f rad/s | signed mean %+.4f | at the %.2f cap %.0f%%"
              % (sum(mags) / len(mags), sum(angs) / len(angs), max_angular,
                 _pct(sum(1 for m in mags if m >= 0.999 * max_angular), len(mags))))
    # /cmd_vel in the bag at scan i+1 is the live node's answer to scan i. Driving
    # scans only: after Ctrl-C the bag keeps the node's last command forever.
    diffs = [abs(b["cmd_ang"] - a["ang"]) for i, (a, b) in enumerate(zip(rows, rows[1:]))
             if b["cmd_ang"] is not None and b["cmd_lin"] and a["state"] != "ROW_END" and v[i] >= MOVING_SPEED]
    if diffs:
        print("           live vs replay steering: mean |diff| %.4f rad/s over %d scans"
              " (near 0 = replay ran the run's settings)" % (sum(diffs) / len(diffs), len(diffs)))

    # -- obstacle stop -------------------------------------------------------------
    eps = segments([r["blocked"] for r in rows])
    print("OBSTACLE   %d forward-stop episode(s) by the replay's gate" % len(eps))
    for a, b in eps:
        print("           t=%6.1f s  s=%5.1f m  held %.1f s, up to %d points in the path box"
              % (rows[a]["t"] - t0, rows[a]["s"], rows[b]["t"] - rows[a]["t"], max(r["path"] for r in rows[a:b + 1])))

    # -- pauses (ground truth is taken at these) -----------------------------------------
    first_move = next((i for i, x in enumerate(v) if x >= MOVING_SPEED), n)
    stopped = [i >= first_move and x < PAUSE_SPEED for i, x in enumerate(v)]
    pauses = [(a, b) for a, b in segments(stopped) if rows[b]["t"] - rows[a]["t"] >= PAUSE_MIN_SEC]
    print("PAUSES     %d stop(s) of %.0f s or more after the robot first moved" % (len(pauses), PAUSE_MIN_SEC))
    for k, (a, b) in enumerate(pauses, 1):
        pre = [r for r in rows[max(0, a - PRE_PAUSE_SCANS):a + 1] if r["xt"] is not None]
        belief = ("xt %+.1f cm, he %+.1f deg (%d scans)" % (
            100.0 * sum(r["xt"] for r in pre) / len(pre),
            math.degrees(sum(r["he"] for r in pre) / len(pre)), len(pre))) if pre else "no fit"
        print("  #%-2d      t=%6.1f s  s=%5.1f m  %4.0f s  %-9s %-24s before stop: %s"
              % (k, rows[a]["t"] - t0, rows[a]["s"], rows[b]["t"] - rows[a]["t"], rows[a]["state"],
                 pause_cause(rows, a, b), belief))

    # -- odometry yaw: a parked robot's yaw should not move ------------------------
    # The detector places centroids ~1-2 m ahead using the odometry pose, so a
    # drifting odometry yaw tilts the fitted row direction (a heading bias),
    # which the steering law turns into a steady sideways offset.
    drv = [i for i, x in enumerate(v) if x >= MOVING_SPEED]
    if drv:
        a, b = drv[0], drv[-1]
        print("ODOMETRY   yaw changed %+.1f deg while driving %.1f m (a straight hallway should give ~0)"
              % (math.degrees(rows[b]["thu"] - rows[a]["thu"]), rows[b]["s"] - rows[a]["s"]))
    parked_t = parked_d = 0.0
    for a, b in pauses:
        if rows[b]["t"] - rows[a]["t"] >= 5.0:
            parked_t += rows[b]["t"] - rows[a]["t"]
            parked_d += rows[b]["thu"] - rows[a]["thu"]
    if parked_t:
        print("           yaw drift while parked: %+.1f deg/min over %.0f s parked (should be ~0)"
              % (math.degrees(parked_d) / parked_t * 60.0, parked_t))
    learned = [r["drift_rate"] for r in rows if r.get("drift_rate") not in ("", None)]
    if learned:
        print("           drift correction ON: learned gyro drift %+.5f rad/s (%+.1f deg/min) by the end"
              % (float(learned[-1]), math.degrees(float(learned[-1])) * 60.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path")
    ap.add_argument("--max-angular", type=float, default=0.08,
                    help="the run's steering cap (row_follower_v3.launch default 0.08)")
    args = ap.parse_args()
    rows = load(args.csv_path)
    if not rows:
        raise SystemExit("no scans in %s" % args.csv_path)
    report(rows, args.max_angular)


if __name__ == "__main__":
    main()
