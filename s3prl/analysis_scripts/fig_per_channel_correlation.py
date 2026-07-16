"""
Per-Channel Correlation Before vs After Permutation (pretrained HuBERT + MERT)

Loads the npz files produced by `run_correlation_heatmap_experiment` and
plots, for each selected merge node, the sorted per-channel absolute
cross-correlation between HuBERT channel i and its matched MERT channel,
both before (identity matching) and after (Hungarian matching) permutation.

Output: result/analysis/chapter5/per_channel_corr_before_after.pdf
"""

import os
import glob
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.labelsize': 12, 'axes.titlesize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

RESULT_ROOT = os.environ.get('S3PRL_RESULT_ROOT', './result')

NPZ_DIR = (
    f'{RESULT_ROOT}/merged_pretrain_upstream/permutation-covariance/'
    'ch5_correlation_heatmaps_merge_cnn_True_use_ties_False_quantile_0.8_'
    'maintain_hubert_behavior_True/correlation_heatmaps_recomputed_no_deepcopy'
)

# Node layout: (node_id, human-readable title)
NODES = [
    (4,   'CNN Layer 1 (512 ch)'),
    (107, 'Transformer Layer 5 (768 ch)'),
    (191, 'Transformer Layer 11 (768 ch)'),
]

OUT_PATH = f'{RESULT_ROOT}/analysis/chapter5/per_channel_corr_before_after.pdf'

COLOR_BEFORE = '#888888'  # grey
COLOR_AFTER  = '#d62728'  # red


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def find_npz(node_id):
    pattern = os.path.join(NPZ_DIR, f'correlation_matrices_node_{node_id}_*.npz')
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f'No npz found for node {node_id} in {NPZ_DIR}')
    return matches[0]


def extract_matched_abs_corr(npz_path):
    """
    Return (before_sorted_desc, after_sorted_desc) — each a 1D array of length D.

    We recompute the Hungarian matching from C_AB_before directly, which is
    cleaner and independent of the stored P_B convention. The matched
    correlation for HuBERT channel a is |C_AB_before[a, col_ind[a]]| where
    col_ind comes from linear_sum_assignment on |C_AB_before| (maximise).

    before[a] = |C_AB_before[a, a]|                (identity pairing)
    after[a]  = |C_AB_before[a, col_ind[a]]|       (Hungarian-matched B channel)
    """
    data = np.load(npz_path, allow_pickle=True)
    C_before = data['C_AB_before']
    absC = np.abs(C_before)

    before = np.abs(np.diag(C_before))

    row_ind, col_ind = linear_sum_assignment(absC, maximize=True)
    # row_ind is 0..D-1 for a square matrix
    after = absC[row_ind, col_ind]

    # Sort descending so the curves compare distribution-wise
    before_sorted = np.sort(before)[::-1]
    after_sorted  = np.sort(after)[::-1]

    return before_sorted, after_sorted


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------

def main():
    n_nodes = len(NODES)
    fig, axes = plt.subplots(1, n_nodes, figsize=(4.2 * n_nodes, 3.6), sharey=True)
    if n_nodes == 1:
        axes = [axes]

    for idx, (ax, (node_id, title)) in enumerate(zip(axes, NODES)):
        npz_path = find_npz(node_id)
        before_sorted, after_sorted = extract_matched_abs_corr(npz_path)
        D = len(before_sorted)

        x = np.arange(D)
        ax.plot(x, before_sorted, color=COLOR_BEFORE, linewidth=1.6,
                label='Before (identity pairing)')
        ax.plot(x, after_sorted, color=COLOR_AFTER, linewidth=1.6,
                label='After (Hungarian)')
        ax.fill_between(x, before_sorted, after_sorted,
                        where=(after_sorted >= before_sorted),
                        color=COLOR_AFTER, alpha=0.15, linewidth=0)

        # Means as inline text (computed from the same data)
        mean_before = before_sorted.mean()
        mean_after  = after_sorted.mean()
        # Panel 1 (CNN) keeps bottom-right; panels 2 and 3 use top-right
        # because the decaying curves leave that corner empty.
        if idx == 0:
            box_x, box_y, va = 0.98, 0.02, 'bottom'
        else:
            box_x, box_y, va = 0.98, 0.98, 'top'
        ax.text(box_x, box_y,
                f'mean $|\\rho|$:\nbefore {mean_before*100:.1f}%\nafter  {mean_after*100:.1f}%',
                transform=ax.transAxes, ha='right', va=va,
                fontsize=9, family='monospace',
                bbox=dict(facecolor='white', edgecolor='lightgray',
                          boxstyle='round,pad=0.3', alpha=0.9))

        ax.set_title(title)
        ax.set_xlabel('Channel rank (sorted by $|\\rho|$, descending)')
        ax.set_xlim(0, D - 1)
        ax.set_ylim(0, 1.0)
        ax.yaxis.grid(True, linestyle=':', alpha=0.5)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('Absolute cross-correlation $|\\rho|$')

    # Shared legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 1.02), frameon=True, edgecolor='gray')

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH)
    png_path = OUT_PATH.replace('.pdf', '.png')
    fig.savefig(png_path)
    print(f'Saved: {OUT_PATH}')
    print(f'Saved: {png_path}')


if __name__ == '__main__':
    main()
