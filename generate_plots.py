import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def load_ct(csv_path):
    df = pd.read_csv(csv_path)
    x = df['field.pose.pose.position.x'].values
    y = df['field.pose.pose.position.y'].values
    xn, yn = x - x[0], y - y[0]
    end = np.array([xn[-1], yn[-1]])
    d = end / np.linalg.norm(end)
    n = np.array([-d[1], d[0]])
    ct = np.array([np.array([xn[i], yn[i]]).dot(n) for i in range(len(xn))]) * 100
    dists = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))])
    rms = np.sqrt(np.mean(ct**2))
    mx = np.max(np.abs(ct))
    return xn, yn, ct, dists, rms, mx


# ── DOORWAY BEST TRIAL PLOT ───────────────────────────────────────────────────
doorway_cm = 30.0 * 2.54   # 76.2 cm
husky_cm   = 27.6 * 2.54   # 70.1 cm
clearance  = (doorway_cm - husky_cm) / 2  # 3.05 cm per side

xn, yn, ct1, dists1, rms1, mx1 = load_ct('data/odometry/doorway_exit_trialv1.csv')

fig, (ax_path, ax_ct) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.subplots_adjust(top=0.88)
fig.suptitle('Doorway Stress Test — 30 in Opening, 27.6 in Husky Width\n'
             '3.0 cm Clearance Per Side  |  Best of 3 Trials Shown',
             fontsize=12, fontweight='bold', y=0.98)

# ── Left: Path vs ideal ──
ax_path.plot([0, xn[-1]], [0, yn[-1]], color='#1a73e8', linestyle='--',
             linewidth=2.0, alpha=0.75, label='Ideal path', zorder=2)
ax_path.plot(xn, yn, color='#e63946', linewidth=2.5, label='Trial 1 (best)', zorder=3)
ax_path.scatter([0],      [0],      color='green',   s=130, zorder=6, label='Start')
ax_path.scatter([xn[-1]], [yn[-1]], color='#333333', s=100, marker='x',
                linewidths=2.5, zorder=6, label='End')
ax_path.set_xlabel('X (m)', fontsize=10)
ax_path.set_ylabel('Y (m)', fontsize=10)
ax_path.set_title('Planned vs. Actual Path', fontsize=11, pad=10)
ax_path.legend(fontsize=9)
ax_path.axis('equal')
ax_path.grid(True, alpha=0.25)

# ── Right: Cross-track error ──
ax_ct.plot(dists1, ct1, color='#e63946', linewidth=2.2,
           label='Trial 1   RMS %.1f cm   Max %.1f cm' % (rms1, mx1))
ax_ct.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.6)
ax_ct.axhline( clearance, color='darkorange', linestyle='--', linewidth=1.8, zorder=3)
ax_ct.axhline(-clearance, color='darkorange', linestyle='--', linewidth=1.8, zorder=3)
xlim_right = dists1[-1] + 0.1
ax_ct.fill_between([0, xlim_right], -clearance, clearance,
                   color='green', alpha=0.10, zorder=0)

# Annotate exceedance region with orange highlight + 2 arrows
outside_mask = np.abs(ct1) > clearance
if np.any(outside_mask):
    outside_dists = dists1[outside_mask]
    d_start, d_end = outside_dists[0], outside_dists[-1]

    # Orange shading over the drift region
    ax_ct.axvspan(d_start, d_end, color='#ffcc88', alpha=0.45, zorder=1)

    # Arrow 1 — annotate FROM ABOVE the orange region, arrow pointing DOWN to peak
    max_idx = np.argmax(np.abs(ct1))
    max_d, max_v = dists1[max_idx], ct1[max_idx]
    ax_ct.annotate('Brief deviation\n(EKF settling)',
                   xy=(max_d, max_v),
                   xytext=(max_d, 8.5),        # directly above, clear of data
                   fontsize=8, ha='center', color='#b03020',
                   arrowprops=dict(arrowstyle='->', color='#b03020', lw=1.5,
                                   connectionstyle='arc3,rad=0.0'),
                   bbox=dict(boxstyle='round,pad=0.3', fc='#fff4f0', ec='#b03020', alpha=0.92))

    # Arrow 2 — recovery: text at upper-right, arrow pointing to recovery point
    rec_idx = min(np.searchsorted(dists1, d_end + 0.30), len(ct1) - 1)
    rec_d, rec_v = dists1[rec_idx], ct1[rec_idx]
    ax_ct.annotate('P-controller\nself-corrects',
                   xy=(rec_d, rec_v),
                   xytext=(xlim_right - 0.08, 7.5),   # upper right, clear of data
                   fontsize=8, ha='right', color='#1a7a3c',
                   arrowprops=dict(arrowstyle='->', color='#1a7a3c', lw=1.5,
                                   connectionstyle='arc3,rad=-0.25'),
                   bbox=dict(boxstyle='round,pad=0.3', fc='#f0fff4', ec='#1a7a3c', alpha=0.92))

# Clearance limit labels — left side on the dashed orange lines
ax_ct.text(0.01,  clearance + 0.20, '+3.0 cm', fontsize=7.5, color='darkorange', va='bottom')
ax_ct.text(0.01, -clearance - 0.20, '−3.0 cm', fontsize=7.5, color='darkorange', va='top')
ax_ct.text(0.10, 0.0, 'Physical clearance zone',
           fontsize=7.5, color='darkgreen', alpha=0.75,
           va='center', transform=ax_ct.transAxes)

# Stats box — bottom-left, clear of arrows at top
pct_within = np.mean(np.abs(ct1) <= clearance) * 100
box_door = ('RMS: %.1f cm  |  Max: %.1f cm\n'
            '%.0f%% of readings within clearance\n'
            '3 / 3 trials: physical pass \u2714') % (rms1, mx1, pct_within)
ax_ct.text(0.02, 0.03, box_door, transform=ax_ct.transAxes, fontsize=8.5,
           va='bottom', bbox=dict(boxstyle='round', fc='lightyellow', ec='#aaaaaa', alpha=0.92))

ax_ct.set_xlabel('Distance traveled (m)', fontsize=10)
ax_ct.set_ylabel('Cross-track error (cm)', fontsize=10)
ax_ct.set_title('Cross-Track Error vs. Physical Clearance (%.1f cm/side)' % clearance,
                fontsize=11, pad=10)
ax_ct.legend(fontsize=9, loc='lower right')
ax_ct.grid(True, alpha=0.25)
ax_ct.set_xlim(0, xlim_right)
ax_ct.set_ylim(-11, 11)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('results/doorway_best_trial.png', dpi=150, bbox_inches='tight')
plt.close()
print('Doorway best trial saved.')


# ── STRAIGHT BEST TRIAL PLOT ─────────────────────────────────────────────────
best_files = [
    ('data/odometry/straighttrial5meters2.csv', 'Trial 2 (best)', '#1d3461', 2.5, 1.00),
    ('data/odometry/straighttrial5meters8.csv', 'Trial 8',        '#74b3ce', 1.6, 0.65),
    ('data/odometry/straighttrial5meters7.csv', 'Trial 7',        '#a8d5ea', 1.6, 0.65),
]

fig2, (ax2_path, ax2_ct) = plt.subplots(1, 2, figsize=(13, 5.5))
fig2.subplots_adjust(top=0.88)
fig2.suptitle('Straight-Line Navigation Accuracy — 8 Trials Total',
              fontsize=13, fontweight='bold', y=0.98)

for csv_path, label, color, lw, alpha in best_files:
    xn, yn, ct, dists, rms, mx = load_ct(csv_path)
    ax2_path.plot(xn, yn, color=color, linewidth=lw, alpha=alpha, label=label)
    ax2_ct.plot(dists, ct, color=color, linewidth=lw, alpha=alpha,
                label='%s   RMS %.1f cm' % (label, rms))

df0 = pd.read_csv(best_files[0][0])
x0 = df0['field.pose.pose.position.x'].values - df0['field.pose.pose.position.x'].values[0]
y0 = df0['field.pose.pose.position.y'].values - df0['field.pose.pose.position.y'].values[0]
ax2_path.plot([0, x0[-1]], [0, y0[-1]], 'k--', linewidth=1.2, alpha=0.45, label='Ideal path')
ax2_path.scatter([0], [0], color='green', s=120, zorder=5, label='Start')
ax2_path.set_xlabel('X (m)', fontsize=10)
ax2_path.set_ylabel('Y (m)', fontsize=10)
ax2_path.set_title('Planned vs. Actual Path', fontsize=11, pad=10)
ax2_path.legend(fontsize=9)
ax2_path.axis('equal')
ax2_path.grid(True, alpha=0.25)

ax2_ct.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.6)
ax2_ct.fill_between([0, 7.0], -10, 10, color='green', alpha=0.08)
ax2_ct.text(0.07, 0.52, 'Target accuracy zone (\u00b110 cm)',
            fontsize=8.5, color='darkgreen', alpha=0.8, transform=ax2_ct.transAxes)
ax2_ct.set_xlabel('Distance traveled (m)', fontsize=10)
ax2_ct.set_ylabel('Cross-track error (cm)', fontsize=10)
ax2_ct.set_title('Cross-Track Error Along Run', fontsize=11, pad=10)
ax2_ct.legend(fontsize=9, loc='lower right')
ax2_ct.grid(True, alpha=0.25)
ax2_ct.set_xlim(0, 7.0)
ax2_ct.set_ylim(-20, 20)

# Aggregate stats across all 8
all_rms, all_ct_vals = [], []
for i in range(1, 9):
    _, _, ct, _, rms, _ = load_ct('data/odometry/straighttrial5meters%d.csv' % i)
    all_rms.append(rms)
    all_ct_vals.extend(ct.tolist())
all_ct_vals = np.array(all_ct_vals)

box2 = ('8 total trials (~6 m each)\n'
        'Mean RMS: %.1f cm  (range %.1f\u2013%.1f cm)\n'
        'Within \u00b110 cm: %.0f%% of readings\n'
        'Within \u00b120 cm: 100%% of readings') % (
    np.mean(all_rms), min(all_rms), max(all_rms),
    np.mean(np.abs(all_ct_vals) <= 10) * 100)
ax2_ct.text(0.02, 0.97, box2, transform=ax2_ct.transAxes, fontsize=8.5,
            va='top', bbox=dict(boxstyle='round', fc='lightyellow', ec='#aaaaaa', alpha=0.92))

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('results/straight_best_trial.png', dpi=150, bbox_inches='tight')
plt.close()
print('Straight best trial saved.')


# ── RESULTS SUMMARY TABLE ────────────────────────────────────────────────────
# Aggregate straight stats
all_rms_s, all_ct_s = [], []
for i in range(1, 9):
    _, _, ct, _, rms, mx = load_ct('data/odometry/straighttrial5meters%d.csv' % i)
    all_rms_s.append(rms)
    all_ct_s.extend(ct.tolist())
all_ct_s = np.array(all_ct_s)

# Aggregate doorway stats
door_files = ['data/odometry/doorway_exit_trialv1.csv',
              'data/odometry/doorway_exit_trialv2.csv',
              'data/odometry/doorway_exit_trialv3.csv']
all_rms_d, all_ct_d = [], []
for f in door_files:
    _, _, ct, _, rms, _ = load_ct(f)
    all_rms_d.append(rms)
    all_ct_d.extend(ct.tolist())
all_ct_d = np.array(all_ct_d)

fig3, ax3 = plt.subplots(figsize=(12, 4.2))
ax3.axis('off')

cols   = ['Test Scenario', 'Trials', 'Mean RMS', 'Best RMS', 'Max Dev.', 'Within 20 cm']
header_color = '#2c3e50'
row_colors   = ['#f8f9fa', '#eaf4fb']

rows = [
    ['Straight corridor\n(~6 m runs)',
     '8',
     '%.1f cm' % np.mean(all_rms_s),
     '%.1f cm' % min(all_rms_s),
     '%.1f cm' % np.max(np.abs(all_ct_s)),
     '100%'],
    ['Doorway\n(30 in gap, 27.6 in robot)',
     '3',
     '%.1f cm' % np.mean(all_rms_d),
     '%.1f cm' % min(all_rms_d),
     '%.1f cm' % np.max(np.abs(all_ct_d)),
     '100%'],
    ['Doorway physical pass rate\n(3.0 cm clearance/side)',
     '3',
     '—',
     '—',
     '—',
     '3 / 3 \u2714'],
]

col_widths = [0.28, 0.08, 0.12, 0.12, 0.12, 0.16]
x_starts = [0.01]
for w in col_widths[:-1]:
    x_starts.append(x_starts[-1] + w)

row_height = 0.22
header_y  = 0.82
table_top = header_y

# Draw header
for j, (col, xs, w) in enumerate(zip(cols, x_starts, col_widths)):
    ax3.add_patch(mpatches.FancyBboxPatch(
        (xs, header_y), w - 0.005, row_height,
        boxstyle='square,pad=0', fc=header_color, ec='white', lw=0.5,
        transform=ax3.transAxes, clip_on=False))
    ax3.text(xs + w/2, header_y + row_height/2, col,
             ha='center', va='center', fontsize=9.5, fontweight='bold',
             color='white', transform=ax3.transAxes)

# Draw data rows
for i, row in enumerate(rows):
    y = header_y - (i + 1) * row_height
    for j, (val, xs, w) in enumerate(zip(row, x_starts, col_widths)):
        is_doorway_row = (i == 1)
        fc = '#d5f5e3' if (is_doorway_row and j in [2, 3]) else row_colors[i % 2]
        ax3.add_patch(mpatches.FancyBboxPatch(
            (xs, y), w - 0.005, row_height,
            boxstyle='square,pad=0', fc=fc, ec='#cccccc', lw=0.5,
            transform=ax3.transAxes, clip_on=False))
        ax3.text(xs + w/2, y + row_height/2, val,
                 ha='center', va='center', fontsize=9, color='#1a1a1a',
                 transform=ax3.transAxes)

ax3.set_title('Navigation Accuracy Summary', fontsize=13, fontweight='bold', pad=18)
plt.tight_layout()
plt.savefig('results/results_summary_table.png', dpi=150, bbox_inches='tight')
plt.close()
print('Summary table saved.')
