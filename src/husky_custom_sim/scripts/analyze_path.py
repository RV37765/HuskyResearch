#!/usr/bin/env python3
"""
Planned vs. Actual Path Analysis
=================================
Reads odometry CSV files recorded with:
    rostopic echo -p /odometry/filtered > <name>.csv

Outputs:
  - Planned vs. actual path overlay plot (PNG)
  - Cross-track error over distance plot (PNG)
  - Summary statistics printed to terminal

Usage:
    python3 analyze_path.py straight_trial1.csv straight_trial2.csv ...
    python3 analyze_path.py left_corner_trial1.csv

Place CSV files in the same directory as this script, or pass full paths.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_odom_csv(csv_path):
    """Load a rostopic echo -p CSV and return x, y, yaw arrays."""
    df = pd.read_csv(csv_path)

    # rostopic echo -p uses these column names
    x = df['field.pose.pose.position.x'].values
    y = df['field.pose.pose.position.y'].values

    # Extract yaw from quaternion if columns exist
    yaw = None
    if all(c in df.columns for c in ['field.pose.pose.orientation.x',
                                      'field.pose.pose.orientation.y',
                                      'field.pose.pose.orientation.z',
                                      'field.pose.pose.orientation.w']):
        from tf.transformations import euler_from_quaternion
        qx = df['field.pose.pose.orientation.x'].values
        qy = df['field.pose.pose.orientation.y'].values
        qz = df['field.pose.pose.orientation.z'].values
        qw = df['field.pose.pose.orientation.w'].values
        yaw = np.array([euler_from_quaternion([qx[i], qy[i], qz[i], qw[i]])[2]
                        for i in range(len(qx))])

    # Normalize to start at origin
    x = x - x[0]
    y = y - y[0]

    return x, y, yaw


def cross_track_errors(x, y):
    """
    Compute perpendicular (cross-track) distance from each point to the
    straight line connecting start and end of the run.
    """
    start = np.array([x[0], y[0]])
    end   = np.array([x[-1], y[-1]])
    vec   = end - start
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return np.zeros(len(x))
    direction = vec / length
    normal = np.array([-direction[1], direction[0]])
    errors = np.array([(np.array([x[i], y[i]]) - start).dot(normal)
                       for i in range(len(x))])
    return errors


def distance_along_path(x, y):
    """Cumulative distance traveled along actual path."""
    diffs = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    return np.concatenate([[0], np.cumsum(diffs)])


def analyze(csv_path, label=None):
    if label is None:
        label = os.path.splitext(os.path.basename(csv_path))[0]

    x, y, yaw = load_odom_csv(csv_path)
    ct_errors  = cross_track_errors(x, y)
    dist_along = distance_along_path(x, y)
    total_dist = dist_along[-1]

    rms_cm = np.sqrt(np.mean(ct_errors**2)) * 100
    max_cm = np.max(np.abs(ct_errors)) * 100
    final_err_cm = np.abs(ct_errors[-1]) * 100

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Distance traveled:       {total_dist:.2f} m")
    print(f"  RMS cross-track error:   {rms_cm:.1f} cm")
    print(f"  Max cross-track error:   {max_cm:.1f} cm")
    print(f"  Final cross-track error: {final_err_cm:.1f} cm")

    # --- Plot 1: Planned vs Actual Path ---
    end_x, end_y = x[-1], y[-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{label}', fontsize=14, fontweight='bold')

    ax = axes[0]
    planned_x = np.array([0, end_x])
    planned_y = np.array([0, end_y])
    ax.plot(planned_x, planned_y, 'b--', linewidth=2, label='Planned path')
    ax.plot(x, y, 'r-', linewidth=2, label='Actual path')
    ax.scatter([0], [0], color='green', s=100, zorder=5, label='Start')
    ax.scatter([end_x], [end_y], color='red', s=100, zorder=5, label='End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Planned vs. Actual Path')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Cross-track error over distance ---
    ax2 = axes[1]
    ax2.plot(dist_along, ct_errors * 100, 'r-', linewidth=1.5)
    ax2.axhline(0, color='blue', linestyle='--', linewidth=1.5, label='Planned')
    ax2.fill_between(dist_along, ct_errors * 100, 0, alpha=0.2, color='red')
    ax2.set_xlabel('Distance traveled (m)')
    ax2.set_ylabel('Cross-track error (cm)')
    ax2.set_title(f'Cross-Track Error\nRMS: {rms_cm:.1f} cm | Max: {max_cm:.1f} cm')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    out_path = f'{label}_analysis.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {out_path}")
    plt.show()

    return {
        'label': label,
        'distance_m': round(total_dist, 2),
        'rms_cm': round(rms_cm, 1),
        'max_cm': round(max_cm, 1),
        'final_err_cm': round(final_err_cm, 1),
    }


def summary_table(results):
    """Print a clean summary table of all trials."""
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Trial':<30} {'Dist (m)':>10} {'RMS (cm)':>10} {'Max (cm)':>10} {'Final (cm)':>12}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    for r in results:
        print(f"  {r['label']:<30} {r['distance_m']:>10.2f} {r['rms_cm']:>10.1f} "
              f"{r['max_cm']:>10.1f} {r['final_err_cm']:>12.1f}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_path.py <trial1.csv> [trial2.csv ...]")
        print("Example: python3 analyze_path.py straight_trial1.csv straight_trial2.csv")
        sys.exit(1)

    results = []
    for csv_path in sys.argv[1:]:
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            continue
        result = analyze(csv_path)
        results.append(result)

    if len(results) > 1:
        summary_table(results)
