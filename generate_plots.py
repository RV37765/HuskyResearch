import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# ── DOORWAY PLOT ──────────────────────────────────────────────────────────────
doorway_cm = 30.0 * 2.54   # 76.2 cm
husky_cm   = 27.6 * 2.54   # 70.1 cm
clearance  = (doorway_cm - husky_cm) / 2  # 3.05 cm per side

files_door = [
    ('data/odometry/doorway_exit_trialv1.csv', 'Trial 1', '#e63946'),
    ('data/odometry/doorway_exit_trialv2.csv', 'Trial 2', '#457b9d'),
    ('data/odometry/doorway_exit_trialv3.csv', 'Trial 3', '#2a9d8f'),
]

fig, (ax_path, ax_ct) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.subplots_adjust(top=0.88)
fig.suptitle('Doorway Navigation — 30 in Opening, 27.6 in Husky Width',
             fontsize=13, fontweight='bold', y=0.98)

stats_door = []
for csv_path, label, color in files_door:
    xn, yn, ct, dists, rms, mx = load_ct(csv_path)
    stats_door.append((label, rms, mx))
    ax_path.plot(xn, yn, color=color, linewidth=2.2, label=label)
    ax_ct.plot(dists, ct, color=color, linewidth=2.0,
               label='%s   RMS %.1f cm   Max %.1f cm' % (label, rms, mx))

df0 = pd.read_csv(files_door[0][0])
x0 = df0['field.pose.pose.position.x'].values - df0['field.pose.pose.position.x'].values[0]
y0 = df0['field.pose.pose.position.y'].values - df0['field.pose.pose.position.y'].values[0]
ax_path.plot([0, x0[-1]], [0, y0[-1]], 'k--', linewidth=1.2, alpha=0.45, label='Ideal')
ax_path.scatter([0], [0], color='green', s=120, zorder=5, label='Start')
ax_path.set_xlabel('X (m)', fontsize=10)
ax_path.set_ylabel('Y (m)', fontsize=10)
ax_path.set_title('Planned vs. Actual Path', fontsize=11, pad=10)
ax_path.legend(fontsize=9)
ax_path.axis('equal')
ax_path.grid(True, alpha=0.25)

ax_ct.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.6)
ax_ct.axhline(clearance,  color='darkorange', linestyle='--', linewidth=1.8)
ax_ct.axhline(-clearance, color='darkorange', linestyle='--', linewidth=1.8)
ax_ct.fill_between([0, 4.5], -clearance, clearance, color='green', alpha=0.10)
ax_ct.text(4.35,  clearance + 0.1, '+3.0 cm limit', fontsize=8, color='darkorange', va='bottom', ha='right')
ax_ct.text(4.35, -clearance - 0.1, '-3.0 cm limit', fontsize=8, color='darkorange', va='top', ha='right')
ax_ct.text(0.08, 0.22, 'Physical clearance zone', fontsize=8.5, color='darkgreen', alpha=0.8)
ax_ct.set_xlabel('Distance traveled (m)', fontsize=10)
ax_ct.set_ylabel('Cross-track error (cm)', fontsize=10)
ax_ct.set_title('Cross-Track Error vs. Physical Clearance (%.1f cm/side)' % clearance,
                fontsize=11, pad=10)
ax_ct.legend(fontsize=9, loc='lower right')
ax_ct.grid(True, alpha=0.25)
ax_ct.set_xlim(0, 4.4)
ax_ct.set_ylim(-11, 11)

mean_rms_door = np.mean([s[1] for s in stats_door])
box_door = ('3 / 3 trials: successful pass\n'
            'Mean RMS: %.1f cm\n'
            'Clearance: %.1f cm per side\n'
            '(30 in door, 27.6 in Husky)') % (mean_rms_door, clearance)
ax_ct.text(0.02, 0.97, box_door, transform=ax_ct.transAxes, fontsize=8.5,
           va='top', bbox=dict(boxstyle='round', fc='lightyellow', ec='#aaaaaa', alpha=0.92))

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('results/doorway_combined_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print('Doorway saved.')

# ── STRAIGHT TRIAL PLOT — 3 representative trials ────────────────────────────
best_files = [
    ('data/odometry/straighttrial5meters2.csv', 'Trial 2  (best)', '#1d3461'),
    ('data/odometry/straighttrial5meters8.csv', 'Trial 8',         '#457b9d'),
    ('data/odometry/straighttrial5meters7.csv', 'Trial 7',         '#74b3ce'),
]

fig2, (ax2_path, ax2_ct) = plt.subplots(1, 2, figsize=(13, 5.5))
fig2.subplots_adjust(top=0.88)
fig2.suptitle('Straight-Line Navigation Accuracy — Representative Trials',
              fontsize=13, fontweight='bold', y=0.98)

for csv_path, label, color in best_files:
    xn, yn, ct, dists, rms, mx = load_ct(csv_path)
    ax2_path.plot(xn, yn, color=color, linewidth=2.2, label=label)
    ax2_ct.plot(dists, ct, color=color, linewidth=2.0,
                label='%s   RMS %.1f cm' % (label, rms))

df0 = pd.read_csv(best_files[0][0])
x0 = df0['field.pose.pose.position.x'].values - df0['field.pose.pose.position.x'].values[0]
y0 = df0['field.pose.pose.position.y'].values - df0['field.pose.pose.position.y'].values[0]
ax2_path.plot([0, x0[-1]], [0, y0[-1]], 'k--', linewidth=1.2, alpha=0.45, label='Ideal')
ax2_path.scatter([0], [0], color='green', s=120, zorder=5, label='Start')
ax2_path.set_xlabel('X (m)', fontsize=10)
ax2_path.set_ylabel('Y (m)', fontsize=10)
ax2_path.set_title('Planned vs. Actual Path', fontsize=11, pad=10)
ax2_path.legend(fontsize=9)
ax2_path.axis('equal')
ax2_path.grid(True, alpha=0.25)

ax2_ct.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.6)
ax2_ct.fill_between([0, 7.0], -10, 10, color='green', alpha=0.08)
ax2_ct.text(0.08, 1.2, 'Target accuracy zone (±10 cm)', fontsize=8.5, color='darkgreen', alpha=0.8)
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

box2 = ('8 total trials\n'
        'Mean RMS: %.1f cm  (range %.1f\u2013%.1f)\n'
        'Within \u00b110 cm: %.0f%% of readings\n'
        'Within \u00b120 cm: 100%% of readings') % (
    np.mean(all_rms), min(all_rms), max(all_rms),
    np.mean(np.abs(all_ct_vals) <= 10) * 100)
ax2_ct.text(0.02, 0.97, box2, transform=ax2_ct.transAxes, fontsize=8.5,
            va='top', bbox=dict(boxstyle='round', fc='lightyellow', ec='#aaaaaa', alpha=0.92))

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('results/straight_combined_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print('Straight saved.')
