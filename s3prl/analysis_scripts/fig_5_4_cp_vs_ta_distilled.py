"""
Figure 5.4 — CP vs TA Comparison (Distilled Models)
Average scores using 9-task SUPERB formula, two interpolation weights λ=0.9 and λ=0.8.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# ── Thesis-verified Average Scores (9-task SUPERB formula) ──────────────────
#
# configs: 2L LKD, 2L LCL, Wide LCL, 3L LKD, 3L LCL
# methods: TA, CP, Ens (Ens only where available)
#
# None = method not available for that config

DATA = {
    # (config, method): (avg_lambda09, avg_lambda08)
    # Values match Table 5.4 after VocID baseline correction (2026-04-11).
    ('2L LKD',  'TA'):  (936.7, 899.2),
    ('2L LKD',  'CP'):  (938.2, 915.4),
    ('2L LKD',  'Ens'): (927.7, 927.7),
    ('2L LCL',  'TA'):  (935.3, 929.1),
    ('2L LCL',  'CP'):  (940.9, 928.9),
    ('Wide',    'TA'):  (930.8, 923.4),
    ('Wide',    'CP'):  (927.6, 910.6),
    ('Wide',    'Ens'): (915.5, 915.5),
    ('3L LKD',  'TA'):  (935.9, 924.3),
    ('3L LKD',  'CP'):  (941.4, 921.8),
    ('3L LKD',  'Ens'): (933.5, 933.5),
    ('3L LCL',  'TA'):  (941.1, 936.4),
    ('3L LCL',  'CP'):  (939.2, 940.4),
}

CONFIGS = ['2L LKD', '2L LCL', 'Wide', '3L LKD', '3L LCL']
CONFIG_LABELS = ['2L LKD', '2L LCL', 'Wide LCL', '3L LKD', '3L LCL']
METHODS = ['TA', 'CP', 'Ens']

COLOR_TA  = '#1f77b4'  # blue
COLOR_CP  = '#ff7f0e'  # orange
COLOR_ENS = '#2ca02c'  # green
COLORS = {'TA': COLOR_TA, 'CP': COLOR_CP, 'Ens': COLOR_ENS}

LAMBDA_IDX = {0.9: 0, 0.8: 1}
LAMBDA_LABELS = {0.9: '(a) λ = 0.9', 0.8: '(b) λ = 0.8'}

BAR_WIDTH = 0.22
Y_MIN, Y_MAX = 880, 950


def get_scores(lam_idx):
    """Return dict: config -> {method: score} for a given lambda index."""
    result = {}
    for cfg in CONFIGS:
        result[cfg] = {}
        for method in METHODS:
            key = (cfg, method)
            if key in DATA:
                result[cfg][method] = DATA[key][lam_idx]
    return result


def plot_subplot(ax, scores, title):
    n_configs = len(CONFIGS)

    # Determine bar positions per config
    # Configs with Ens: 3 bars centred; without: 2 bars centred
    for i, cfg in enumerate(CONFIGS):
        available = [m for m in METHODS if m in scores[cfg]]
        n_bars = len(available)
        total_width = n_bars * BAR_WIDTH
        offsets = np.linspace(-total_width / 2 + BAR_WIDTH / 2,
                               total_width / 2 - BAR_WIDTH / 2, n_bars)
        for j, method in enumerate(available):
            val = scores[cfg][method]
            bar = ax.bar(i + offsets[j], val - Y_MIN, BAR_WIDTH,
                         color=COLORS[method], label=method if i == 0 else '_nolegend_',
                         edgecolor='white', linewidth=0.5)
            # Annotate bar top
            ax.text(i + offsets[j], val - Y_MIN + 0.5, f'{val:.0f}',
                    ha='center', va='bottom', fontsize=8, rotation=90,
                    fontweight='bold')

    ax.set_title(title, fontweight='bold')
    ax.set_xticks(range(n_configs))
    ax.set_xticklabels(CONFIG_LABELS, rotation=15, ha='right')
    ax.set_ylim(0, Y_MAX - Y_MIN + 15)
    ax.set_yticks(np.arange(0, Y_MAX - Y_MIN + 10, 10))
    ax.set_yticklabels([f'{v + Y_MIN:.0f}' for v in np.arange(0, Y_MAX - Y_MIN + 10, 10)])
    ax.set_ylabel('Average Score')
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ── Build figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, lam in zip(axes, [0.9, 0.8]):
    scores = get_scores(LAMBDA_IDX[lam])
    plot_subplot(ax, scores, LAMBDA_LABELS[lam])

# Shared legend
handles = [
    plt.Rectangle((0, 0), 1, 1, color=COLOR_TA,  label='Task Arithmetic (TA)'),
    plt.Rectangle((0, 0), 1, 1, color=COLOR_CP,  label='Correlation Permutation (CP)'),
    plt.Rectangle((0, 0), 1, 1, color=COLOR_ENS, label='Ensemble'),
]
fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.08),
           frameon=True, edgecolor='gray')

# Only left subplot needs y-label; right shares axis (sharey=True)
axes[1].set_ylabel('')

plt.tight_layout(rect=[0, 0.05, 1, 1])

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = os.path.join(os.environ.get('S3PRL_RESULT_ROOT', './result'), 'analysis/chapter5/ch5_cp_vs_ta_distilled.pdf')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path)
print(f'Saved: {out_path}')
