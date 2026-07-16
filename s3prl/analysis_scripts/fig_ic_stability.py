"""
fig_ic_stability.py — IC Stability Spotlight Figure (Chapter 5)

Highlights how Correlation-Permutation (CP) preserves IC accuracy at λ=0.8
while Task Arithmetic (TA) collapses for L_KD models.

Output: result/analysis/chapter5/ic_stability_spotlight.pdf
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 9, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
configs = ['2L L$_{KD}$', '2L L$_{CL}$', '3L L$_{KD}$', '3L L$_{CL}$', 'Wide L$_{CL}$']
configs_short = ['2L_LKD', '2L_LCL', '3L_LKD', '3L_LCL', 'Wide_LCL']

# IC accuracy values
ta_09  = [93.56, 93.97, 93.70, 94.54, 94.67]
ta_08  = [69.29, 92.05, 89.24, 93.57, 92.53]
cp_09  = [93.92, 93.99, 92.88, 94.91, 92.41]
cp_08  = [90.30, 90.43, 87.64, 93.20, 88.77]

# Ensemble values (None where not available)
ens    = [85.55, None, 86.13, None, 85.21]

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
# TA: blue family  (light=0.9, dark=0.8)
# CP: red/orange family (light=0.9, dark=0.8)
ta_09_color = '#6baed6'   # light blue
ta_08_color = '#08519c'   # dark blue
cp_09_color = '#fc8d59'   # light orange-red
cp_08_color = '#b30000'   # dark red

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
n_configs = len(configs)
n_bars = 4          # TA@0.9, TA@0.8, CP@0.9, CP@0.8
group_width = 0.75  # total width for each config group
bar_width = group_width / n_bars
gap = 1.0           # spacing between config groups

x_centers = np.arange(n_configs) * (group_width + gap)

# Bar x-positions within each group
offsets = np.array([0, 1, 2, 3]) * bar_width - (n_bars - 1) * bar_width / 2

fig, ax = plt.subplots(figsize=(12, 5))

# ---------------------------------------------------------------------------
# Draw bars
# ---------------------------------------------------------------------------
bars_ta09 = ax.bar(x_centers + offsets[0], ta_09, bar_width,
                   color=ta_09_color, label='TA λ=0.9', zorder=3,
                   edgecolor='white', linewidth=0.4)
bars_ta08 = ax.bar(x_centers + offsets[1], ta_08, bar_width,
                   color=ta_08_color, label='TA λ=0.8', zorder=3,
                   edgecolor='white', linewidth=0.4)
bars_cp09 = ax.bar(x_centers + offsets[2], cp_09, bar_width,
                   color=cp_09_color, label='CP λ=0.9', zorder=3,
                   edgecolor='white', linewidth=0.4)
bars_cp08 = ax.bar(x_centers + offsets[3], cp_08, bar_width,
                   color=cp_08_color, label='CP λ=0.8', zorder=3,
                   edgecolor='white', linewidth=0.4)

# ---------------------------------------------------------------------------
# Ensemble dashed lines (per-config horizontal spans)
# ---------------------------------------------------------------------------
span_half = group_width / 2 + bar_width * 0.2   # slightly wider than bars
for i, ens_val in enumerate(ens):
    if ens_val is not None:
        x_left  = x_centers[i] - span_half
        x_right = x_centers[i] + span_half
        ax.hlines(ens_val, x_left, x_right,
                  colors='dimgray', linestyles='--', linewidth=1.4, zorder=4)

# Add a single ensemble legend entry
ens_line = matplotlib.lines.Line2D([0], [0], color='dimgray', linestyle='--',
                                    linewidth=1.4, label='Ensemble distill.')

# ---------------------------------------------------------------------------
# Annotate the 2L L_KD TA@0.8 bar (collapse highlight)
# ---------------------------------------------------------------------------
# Bar position for 2L_LKD TA@0.8 (index 0, offset 1)
collapse_x = x_centers[0] + offsets[1]
collapse_y = ta_08[0]   # 69.29

# Arrow from annotation text above the bar
ax.annotate('69.3%',
            xy=(collapse_x, collapse_y + 0.4),   # arrow tip just above bar top
            xytext=(collapse_x - 0.25, collapse_y + 10.5),
            fontsize=9, color=ta_08_color, fontweight='bold',
            ha='center',
            arrowprops=dict(arrowstyle='->', color=ta_08_color,
                            lw=1.5, connectionstyle='arc3,rad=0.15'))

# ---------------------------------------------------------------------------
# Axes formatting
# ---------------------------------------------------------------------------
ax.set_ylabel('IC Accuracy (%)', fontsize=12)
ax.set_ylim(65, 100)
ax.set_xlim(x_centers[0] - group_width * 0.75,
            x_centers[-1] + group_width * 0.75)

ax.set_xticks(x_centers)
ax.set_xticklabels(configs, fontsize=10)

ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
ax.grid(axis='y', which='major', linestyle=':', linewidth=0.6,
        color='gray', alpha=0.5, zorder=0)


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
ta_09_patch = mpatches.Patch(color=ta_09_color, label='TA λ=0.9')
ta_08_patch = mpatches.Patch(color=ta_08_color, label='TA λ=0.8')
cp_09_patch = mpatches.Patch(color=cp_09_color, label='CP λ=0.9')
cp_08_patch = mpatches.Patch(color=cp_08_color, label='CP λ=0.8')

ax.legend(handles=[ta_09_patch, ta_08_patch, cp_09_patch, cp_08_patch, ens_line],
          loc='upper right', framealpha=0.9, ncol=3, fontsize=9)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.join(os.path.dirname(__file__), '..', 'result', 'analysis', 'chapter5')
out_path = os.path.join(out_dir, 'ic_stability_spotlight.pdf')
os.makedirs(out_dir, exist_ok=True)
plt.savefig(out_path)
print(f"Saved: {os.path.abspath(out_path)}")

# Also save PNG preview
png_path = out_path.replace('.pdf', '.png')
plt.savefig(png_path)
print(f"Saved: {os.path.abspath(png_path)}")
