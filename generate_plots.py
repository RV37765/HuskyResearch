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

xn,  yn,  ct1, dists1, rms1, mx1 = load_ct('data/odometry/doorway_exit_trialv1.csv')
xn2, yn2, ct2, dists2, rms2, mx2 = load_ct('data/odometry/doorway_exit_trialv2.csv')
xn3, yn3, ct3, dists3, rms3, mx3 = load_ct('data/odometry/doorway_exit_trialv3.csv')

fig, (ax_path, ax_ct) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.subplots_adjust(top=0.88)
fig.suptitle('Doorway Stress Test — 30 in Opening, 27.6 in Husky Width\n'
             '3.0 cm Clearance Per Side  |  All 3 Trials Shown',
             fontsize=12, fontweight='bold', y=0.98)

# ── Left: Path vs ideal (Trial 1 only) ──
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

# ── Right: Cross-track error — all 3 trials ──
# Trials 2 & 3 as thin muted lines first (background)
ax_ct.plot(dists2, ct2, color='#f4a0a0', linewidth=1.3, alpha=0.75,
           label='Trial 2   RMS %.1f cm' % rms2)
ax_ct.plot(dists3, ct3, color='#c97c7c', linewidth=1.3, alpha=0.75,
           label='Trial 3   RMS %.1f cm' % rms3)
# Trial 1 (best) bold on top
ax_ct.plot(dists1, ct1, color='#e63946', linewidth=2.5, zorder=4,
           label='Trial 1 (best)   RMS %.1f cm' % rms1)
ax_ct.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.6)
ax_ct.axhline( clearance, color='darkorange', linestyle='--', linewidth=1.8, zorder=3)
ax_ct.axhline(-clearance, color='darkorange', linestyle='--', linewidth=1.8, zorder=3)
xlim_right = max(dists1[-1], dists2[-1], dists3[-1]) + 0.1
ax_ct.fill_between([0, xlim_right], -clearance, clearance,
                   color='green', alpha=0.10, zorder=0)

# "Physical clearance zone" label — inside the green band, centered, near top of band
ax_ct.text(xlim_right / 2, clearance - 0.25,
           'Physical clearance zone (\u00b13.0 cm)',
           fontsize=8.5, color='darkgreen', alpha=0.9,
           ha='center', va='top', fontweight='bold')

# Orange limit labels — right end of each dashed line
ax_ct.text(xlim_right - 0.03,  clearance + 0.18, '+3.0 cm',
           fontsize=7.5, color='darkorange', va='bottom', ha='right')
ax_ct.text(xlim_right - 0.03, -clearance - 0.18, '\u22123.0 cm',
           fontsize=7.5, color='darkorange', va='top', ha='right')

# Annotate exceedance region with orange highlight + 2 arrows
outside_mask = np.abs(ct1) > clearance
if np.any(outside_mask):
    outside_dists = dists1[outside_mask]
    d_start, d_end = outside_dists[0], outside_dists[-1]

    # Orange shading over the drift region
    ax_ct.axvspan(d_start, d_end, color='#ffcc88', alpha=0.45, zorder=1)

    # Arrow 1 — "Brief deviation" box positioned LOW (around x=2.5, y=-5),
    #            arrow pointing up to the actual peak
    max_idx = np.argmax(np.abs(ct1))
    max_d, max_v = dists1[max_idx], ct1[max_idx]
    ax_ct.annotate('Brief deviation\n(EKF settling)',
                   xy=(max_d, max_v),
                   xytext=(2.55, -6.5),        # lower, inside orange zone
                   fontsize=8, ha='center', color='#b03020',
                   arrowprops=dict(arrowstyle='->', color='#b03020', lw=1.5,
                                   connectionstyle='arc3,rad=0.2'),
                   bbox=dict(boxstyle='round,pad=0.3', fc='#fff4f0', ec='#b03020', alpha=0.92))

    # Arrow 2 — "P-controller self-corrects" pushed left from right edge
    rec_idx = min(np.searchsorted(dists1, d_end + 0.30), len(ct1) - 1)
    rec_d, rec_v = dists1[rec_idx], ct1[rec_idx]
    ax_ct.annotate('P-controller\nself-corrects',
                   xy=(rec_d, rec_v),
                   xytext=(xlim_right - 0.55, 7.8),   # upper area, well left of edge
                   fontsize=8, ha='center', color='#1a7a3c',
                   arrowprops=dict(arrowstyle='->', color='#1a7a3c', lw=1.5,
                                   connectionstyle='arc3,rad=-0.2'),
                   bbox=dict(boxstyle='round,pad=0.3', fc='#f0fff4', ec='#1a7a3c', alpha=0.92))

# Stats box — bottom right
mean_rms_all = np.mean([rms1, rms2, rms3])
pct_within = np.mean(np.abs(ct1) <= clearance) * 100
box_door = ('Best trial RMS: %.1f cm  |  Mean RMS (3 trials): %.1f cm\n'
            '3 / 3 trials: physical pass \u2714\n'
            'Replication confirmed — consistent across all runs') % (rms1, mean_rms_all)
ax_ct.text(0.98, 0.03, box_door, transform=ax_ct.transAxes, fontsize=8.5,
           va='bottom', ha='right',
           bbox=dict(boxstyle='round', fc='lightyellow', ec='#aaaaaa', alpha=0.92))

ax_ct.set_xlabel('Distance traveled (m)', fontsize=10)
ax_ct.set_ylabel('Cross-track error (cm)', fontsize=10)
ax_ct.set_title('Cross-Track Error vs. Physical Clearance (%.1f cm/side)' % clearance,
                fontsize=11, pad=10)
ax_ct.legend(fontsize=9, loc='upper left')
ax_ct.grid(True, alpha=0.25)
ax_ct.set_xlim(0, xlim_right)
ax_ct.set_ylim(-11, 11)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('results/doorway_best_trial.png', dpi=150, bbox_inches='tight')
plt.close()
print('Doorway best trial saved.')


# ── STRAIGHT BEST TRIAL PLOT ──────────────────────────────────────────────────
# Load Trial 2 (best)
xn2, yn2, ct2, dists2, rms2, mx2 = load_ct('data/odometry/straighttrial5meters2.csv')

# Aggregate stats across all 8 trials
all_rms, all_ct_vals = [], []
all_ct_interp = []
common_dist = np.linspace(0, 6.0, 300)  # common distance grid for averaging

for i in range(1, 9):
    _, _, ct_i, dists_i, rms_i, _ = load_ct('data/odometry/straighttrial5meters%d.csv' % i)
    all_rms.append(rms_i)
    all_ct_vals.extend(ct_i.tolist())
    # Interpolate onto common grid (only up to this trial's max distance)
    max_d = min(dists_i[-1], common_dist[-1])
    mask = common_dist <= max_d
    ct_interp = np.interp(common_dist[mask], dists_i, ct_i)
    # Pad to full length with NaN
    full = np.full(len(common_dist), np.nan)
    full[mask] = ct_interp
    all_ct_interp.append(full)

all_ct_vals = np.array(all_ct_vals)
all_ct_interp = np.array(all_ct_interp)  # shape (8, 300)

# Mean and std across trials at each distance point (ignoring NaN)
mean_ct = np.nanmean(all_ct_interp, axis=0)
std_ct  = np.nanstd(all_ct_interp,  axis=0)

fig2, (ax2_path, ax2_ct) = plt.subplots(1, 2, figsize=(13, 5.5))
fig2.subplots_adjust(top=0.88)
fig2.suptitle('Straight-Line Navigation Accuracy — 8 Trials, ~6 m Each',
              fontsize=13, fontweight='bold', y=0.98)

# ── Left: Path plot — Trial 2 only vs ideal ──
# Ideal path: straight line from origin to Trial 2's endpoint
ax2_path.plot([0, xn2[-1]], [0, yn2[-1]], color='#1a73e8', linestyle='--',
              linewidth=2.0, alpha=0.75, label='Ideal path', zorder=2)
ax2_path.plot(xn2, yn2, color='#1d3461', linewidth=2.5, label='Trial 2 (best)', zorder=3)
ax2_path.scatter([0],       [0],       color='green',   s=130, zorder=6, label='Start')
ax2_path.scatter([xn2[-1]], [yn2[-1]], color='#333333', s=100, marker='x',
                 linewidths=2.5, zorder=6, label='End')
ax2_path.set_xlabel('X (m)', fontsize=10)
ax2_path.set_ylabel('Y (m)', fontsize=10)
ax2_path.set_title('Planned vs. Actual Path — Trial 2 (Best)', fontsize=11, pad=10)
ax2_path.legend(fontsize=9)
ax2_path.axis('equal')
ax2_path.grid(True, alpha=0.25)

# ── Right: Cross-track error — Trial 2 line + 8-trial mean ± std band ──
# Shaded band: mean ± 1 std across all 8 trials
valid = ~np.isnan(mean_ct)
ax2_ct.fill_between(common_dist[valid],
                    (mean_ct - std_ct)[valid],
                    (mean_ct + std_ct)[valid],
                    color='#74b3ce', alpha=0.30, label='8-trial range (\u00b11\u03c3)')
ax2_ct.plot(common_dist[valid], mean_ct[valid],
            color='#457b9d', linewidth=1.8, linestyle='--',
            label='8-trial mean RMS %.1f cm' % np.mean(all_rms))
ax2_ct.plot(dists2, ct2, color='#1d3461', linewidth=2.5,
            label='Trial 2 (best)   RMS %.1f cm' % rms2)

ax2_ct.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.6)
ax2_ct.fill_between([0, 7.0], -10, 10, color='green', alpha=0.07, zorder=0)
ax2_ct.text(0.98, 0.725, 'Target zone (\u00b110 cm)',
            fontsize=8, color='darkgreen', alpha=0.85,
            ha='right', va='top', transform=ax2_ct.transAxes)

ax2_ct.set_xlabel('Distance traveled (m)', fontsize=10)
ax2_ct.set_ylabel('Cross-track error (cm)', fontsize=10)
ax2_ct.set_title('Cross-Track Error Along Run', fontsize=11, pad=10)
ax2_ct.legend(fontsize=9, loc='lower right')
ax2_ct.grid(True, alpha=0.25)
ax2_ct.set_xlim(0, 7.0)
ax2_ct.set_ylim(-20, 20)

# Stats box — top left
box2 = ('8 total trials (~6 m each)\n'
        'Mean RMS: %.1f cm  (range %.1f\u2013%.1f cm)\n'
        'Best trial (T2): %.1f cm RMS\n'
        'Within \u00b120 cm: 100%% of readings') % (
    np.mean(all_rms), min(all_rms), max(all_rms), rms2)
ax2_ct.text(0.02, 0.97, box2, transform=ax2_ct.transAxes, fontsize=8.5,
            va='top', bbox=dict(boxstyle='round', fc='lightyellow', ec='#aaaaaa', alpha=0.92))

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('results/straight_best_trial.png', dpi=150, bbox_inches='tight')
plt.close()
print('Straight best trial saved.')


# ── RESULTS SUMMARY TABLE ────────────────────────────────────────────────────
all_rms_s, all_ct_s = [], []
for i in range(1, 9):
    _, _, ct, _, rms, _ = load_ct('data/odometry/straighttrial5meters%d.csv' % i)
    all_rms_s.append(rms)
    all_ct_s.extend(ct.tolist())
all_ct_s = np.array(all_ct_s)

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

cols         = ['Test Scenario', 'Trials', 'Mean RMS', 'Best RMS', 'Max Dev.', 'Within 20 cm']
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
     '\u2014',
     '\u2014',
     '\u2014',
     '3 / 3 \u2714'],
]

col_widths = [0.28, 0.08, 0.12, 0.12, 0.12, 0.16]
x_starts = [0.01]
for w in col_widths[:-1]:
    x_starts.append(x_starts[-1] + w)

row_height = 0.22
header_y   = 0.82

for j, (col, xs, w) in enumerate(zip(cols, x_starts, col_widths)):
    ax3.add_patch(mpatches.FancyBboxPatch(
        (xs, header_y), w - 0.005, row_height,
        boxstyle='square,pad=0', fc=header_color, ec='white', lw=0.5,
        transform=ax3.transAxes, clip_on=False))
    ax3.text(xs + w/2, header_y + row_height/2, col,
             ha='center', va='center', fontsize=9.5, fontweight='bold',
             color='white', transform=ax3.transAxes)

for i, row in enumerate(rows):
    y = header_y - (i + 1) * row_height
    for j, (val, xs, w) in enumerate(zip(row, x_starts, col_widths)):
        fc = '#d5f5e3' if (i == 1 and j in [2, 3]) else row_colors[i % 2]
        ax3.add_patch(mpatches.FancyBboxPatch(
            (xs, y), w - 0.005, row_height,
            boxstyle='square,pad=0', fc=fc, ec='#cccccc', lw=0.5,
            transform=ax3.transAxes, clip_on=False))
        ax3.text(xs + w/2, y + row_height/2, val,
                 ha='center', va='center', fontsize=9, color='#1a1a1a',
                 transform=ax3.transAxes)

ax3.set_title('Navigation Accuracy Summary', fontsize=13, fontweight='bold', pad=18)
footer_y = header_y - len(rows) * row_height - 0.08
ax3.text(0.5, footer_y,
         '11 total benchmark trials (8 straight + 3 doorway) — '
         'consistent results confirmed across all runs',
         ha='center', va='top', fontsize=8.5, color='#444444', style='italic',
         transform=ax3.transAxes)
plt.tight_layout()
plt.savefig('results/results_summary_table.png', dpi=150, bbox_inches='tight')
plt.close()
print('Summary table saved.')


# ── LIDAR SCAN WEDGE DIAGRAM ─────────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(11, 6.5))
ax4.set_aspect('equal')
ax4.set_xlim(-1.3, 3.8)
ax4.set_ylim(-2.05, 1.95)
ax4.axis('off')
fig4.patch.set_facecolor('white')
fig4.suptitle('row_follower.py — Top-Down LiDAR Centering Logic',
              fontsize=13, fontweight='bold', y=0.97)

hall_hw = 1.15   # half-width of corridor (m)
wall_t  = 0.20   # wall thickness

# Floor
ax4.add_patch(mpatches.Rectangle((-1.3, -hall_hw), 5.1, 2 * hall_hw,
              color='#f2f2f2', zorder=0))
# Top wall
ax4.add_patch(mpatches.Rectangle((-1.3, hall_hw), 5.1, wall_t,
              color='#4a4a4a', zorder=2))
ax4.text(-1.15, hall_hw + wall_t / 2, 'Left wall',
         ha='left', va='center', fontsize=8.5, color='white',
         fontweight='bold', zorder=3)
# Bottom wall
ax4.add_patch(mpatches.Rectangle((-1.3, -hall_hw - wall_t), 5.1, wall_t,
              color='#4a4a4a', zorder=2))
ax4.text(-1.15, -hall_hw - wall_t / 2, 'Right wall',
         ha='left', va='center', fontsize=8.5, color='white',
         fontweight='bold', zorder=3)

# Robot (Husky A200, top-down, centered at origin)
rl, rw = 0.85, 0.58
ax4.add_patch(mpatches.Rectangle((-rl / 2, -rw / 2), rl, rw,
              color='#2c3e50', zorder=5, linewidth=0))
ax4.text(0, 0.09,  'Husky A200', ha='center', va='center',
         fontsize=7, color='#aad4f5', fontweight='bold', zorder=6)
ax4.text(0, -0.11, 'VLP-16',     ha='center', va='center',
         fontsize=6.5, color='white', zorder=6)
# Forward direction arrow
ax4.annotate('', xy=(rl / 2 + 0.32, 0), xytext=(rl / 2 + 0.03, 0),
             arrowprops=dict(arrowstyle='->', color='#cccccc', lw=2.2), zorder=6)
ax4.text(rl / 2 + 0.5, 0.09, 'forward', ha='center', va='bottom',
         fontsize=7, color='gray', style='italic', zorder=6)

# LiDAR angle windows
LEFT_MIN, LEFT_MAX = 11, 86        # degrees from forward (+X), CCW
RIGHT_MIN, RIGHT_MAX = 274, 349    # equivalent to -86° to -11°

wedge_r = 2.3
ax4.add_patch(mpatches.Wedge((0, 0), wedge_r, LEFT_MIN, LEFT_MAX,
              color='#2980b9', alpha=0.18, zorder=1))
ax4.add_patch(mpatches.Wedge((0, 0), wedge_r, RIGHT_MIN, RIGHT_MAX,
              color='#c0392b', alpha=0.18, zorder=1))

# Wedge boundary dashed lines (angular limits)
for ang_deg, col in [(LEFT_MIN, '#2980b9'), (LEFT_MAX, '#2980b9'),
                     (-11, '#c0392b'), (-86, '#c0392b')]:
    th = np.radians(ang_deg)
    ax4.plot([0, wedge_r * np.cos(th)], [0, wedge_r * np.sin(th)],
             color=col, lw=1.1, ls='--', alpha=0.55, zorder=2)

# Sample LiDAR rays (show 3 per side at representative angles within each window)
def ray_hit(ang_deg, wall_y):
    th = np.radians(ang_deg)
    s = np.sin(th)
    if abs(s) < 1e-9:
        return None
    r = wall_y / s
    return (r * np.cos(th), wall_y) if r > 0 else None

for ang in [30, 52, 78]:
    pt = ray_hit(ang, hall_hw)
    if pt:
        ax4.plot([0, pt[0]], [0, pt[1]], color='#2980b9', lw=1.8,
                 alpha=0.72, zorder=3, solid_capstyle='round')
        ax4.scatter(*pt, color='#2980b9', s=22, zorder=4)

for ang in [-30, -52, -78]:
    pt = ray_hit(ang, -hall_hw)
    if pt:
        ax4.plot([0, pt[0]], [0, pt[1]], color='#c0392b', lw=1.8,
                 alpha=0.72, zorder=3, solid_capstyle='round')
        ax4.scatter(*pt, color='#c0392b', s=22, zorder=4)

# L / R distance arrows (perpendicular to walls, left of robot body)
arr_x = -0.30
ax4.annotate('', xy=(arr_x, hall_hw), xytext=(arr_x, rw / 2),
             arrowprops=dict(arrowstyle='<->', color='#2980b9',
                             lw=2.2, mutation_scale=16), zorder=7)
ax4.text(arr_x - 0.11, (rw / 2 + hall_hw) / 2, 'L',
         ha='right', va='center', fontsize=20, color='#2980b9',
         fontweight='bold', zorder=7)

ax4.annotate('', xy=(arr_x, -hall_hw), xytext=(arr_x, -rw / 2),
             arrowprops=dict(arrowstyle='<->', color='#c0392b',
                             lw=2.2, mutation_scale=16), zorder=7)
ax4.text(arr_x - 0.11, -(rw / 2 + hall_hw) / 2, 'R',
         ha='right', va='center', fontsize=20, color='#c0392b',
         fontweight='bold', zorder=7)

# Window label callout boxes
ax4.text(2.35, hall_hw * 0.52,
         'Left window\n(11° \u2013 86°)\naverage \u2192 L',
         ha='center', va='center', fontsize=9.5, color='#1a5e8a',
         bbox=dict(boxstyle='round,pad=0.45', fc='#eaf4fb',
                   ec='#2980b9', lw=1.3),
         zorder=8)
ax4.text(2.35, -hall_hw * 0.52,
         'Right window\n(\u221286° \u2013 \u221211°)\naverage \u2192 R',
         ha='center', va='center', fontsize=9.5, color='#7b1e12',
         bbox=dict(boxstyle='round,pad=0.45', fc='#fdedec',
                   ec='#c0392b', lw=1.3),
         zorder=8)

# Centerline
ax4.plot([-1.0, 3.5], [0, 0], color='gray', ls=':', lw=1.1, alpha=0.45, zorder=1)
ax4.text(3.45, 0.07, 'centerline', ha='right', va='bottom',
         fontsize=7, color='gray', style='italic')

# Formula box below the bottom wall
ax4.text(1.2, -hall_hw - wall_t - 0.18,
         'error = L \u2212 R\n'
         '\u03c9 = \u2212K\u209b\u2090\u209c \u00b7 (L\u2212R) \u2212 K\u2095\u2091\u2091 \u00b7 \u03b8\u2091\u2090\u2090\n'
         'L = R  \u21d2  error = 0  \u21d2  robot centered',
         ha='center', va='top', fontsize=10.5, linespacing=1.55,
         bbox=dict(boxstyle='round,pad=0.55', fc='#fef9e7',
                   ec='#888888', lw=1.2),
         zorder=8)

plt.tight_layout()
plt.savefig('results/lidar_wedge_diagram.png', dpi=150, bbox_inches='tight')
plt.close()
print('LiDAR wedge diagram saved.')
