#!/usr/bin/env python3
# =============================================================================
#  Chapters 4 & 5 statistical testing  —  STEP 3  (RUN THIS ON NSCC)
# =============================================================================
#
#  Implements the 207 tests defined in reports/stat_tests/README_ch4_ch5.md:
#
#    C4-1  TA vs Ensemble Distillation        3 pairs × 9 tasks =  27 tests
#    C4-2  L_KD vs L_CL at key λ             6 pairs × 9 tasks =  54 tests
#    C4-3  Pre-training data variants         2 pairs × 9 tasks =  18 tests
#    C5-1  CP vs TA on distilled models      10 pairs × 9 tasks =  90 tests
#    C5-2  CP different-init vs shared-init   2 pairs × 9 tasks =  18 tests
#                                                        Total = 207 tests
#
#    * Classification tasks → McNemar's test
#        - continuity-corrected χ² when discordant n ≥ 25
#        - exact binomial when discordant n < 25
#    * ASR → paired bootstrap (10 000 resamples, seed 1234)
#    * Multi-fold tasks (ER 5-fold, ESC-50 5-fold) → samples pooled across folds
#    * Multiple-testing control → Benjamini-Hochberg (FDR q = 0.05)
#                                  within each priority group
#
#  Degrades gracefully: tests with missing prediction files are reported as
#  NOT TESTED with the reason and expected location.
#
#  Run:
#      cd <repo>/s3prl
#      python reports/stat_tests/run_stat_tests_ch4_ch5.py
#
#  Outputs:
#      reports/stat_tests/RESULTS_ch4_ch5.md   (main table + BH summary)
#      reports/stat_tests/RESULTS_ch4_ch5.tsv  (machine-readable)
# =============================================================================
import os
import sys
import csv
import glob
import math
import random
from collections import defaultdict

import numpy as np

try:
    from scipy.stats import chi2 as _chi2, binomtest as _binomtest
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
STAGE = os.path.join(PROJECT_ROOT, "reports", "stat_tests", "predictions_ch4_ch5")

SEARCH_ROOTS = [
    os.path.join(STAGE, "nscc_local"),
    os.path.join(STAGE, "ntu"),
    os.path.join(PROJECT_ROOT, "result", "downstream"),
]

BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_SEED = 1234
BH_Q = 0.05

# ============================================================================
#  Model registry
# ============================================================================
MODELS = {
    # ---- Chapter 4 endpoints (single-domain students) ----
    "D1":  "distilhubert-ls960-own",
    "D2":  "distill_only_mert-init-weight-from-hubert_base-models-simple-avg-pool-for-teacher-train-libri-960",
    "D3":  "barlow_no_distort_old",
    "D4":  "mert_barlow_2layers_old_style",
    "D7":  "barlow_no_distort_old_3layers",
    "D8":  "mert_barlow_2layers_old_style-3layers",
    "D3w": "hubert_barlow_wide",
    "D4w": "mert_barlow_wide",
    "D5w": "hubert_l1_wide",
    "D6w": "mert_l1_wide",
    # ---- Baselines ----
    "ENS-2L": "mert-and-hubert-init-hubert-distillation-without-feat-translation",
    "ENS-3L": "mert-and-hubert-init-hubert-distillation-without-feat-translation-3layers",
    "ENS-W":  "multi_teacher_l1_wide",
    # ---- Teachers ----
    "HuBERT": "hubert_base",
    "MERT":   "mert_v0_public",
    # ---- Task Arithmetic (merged) ----
    # 2L L_KD
    "TA-2L-KD-0.9": "task_vector_dhubert_ls_960_weight_0.9_and_mert_ls_960_weight_0.1_both_init_hubert_both_same_seed",
    "TA-2L-KD-0.8": "task_vector_dhubert_ls_960_weight_0.8_and_mert_ls_960_weight_0.2_both_init_hubert_both_same_seed-new-models",
    "TA-2L-KD-0.5": "task_vector_dhubert_ls_960_weight_0.5_and_mert_ls_960_weight_0.5_both_init_hubert_both_same_seed-new-models",
    "TA-2L-KD-0.1": "task_vector_dhubert_ls_960_weight_0.1_and_mert_ls_960_weight_0.9_both_init_hubert_both_same_seed-new-models",
    # 2L L_CL
    "TA-2L-CL-0.9": "task_vector_barlow_speech_weight_0.9_and_barlow_music_weight_0.1_both_init_hubert",
    "TA-2L-CL-0.8": "task_vector_barlow_speech_weight_0.8_and_barlow_music_weight_0.2_both_init_hubert",
    "TA-2L-CL-0.5": "task_vector_barlow_speech_weight_0.5_and_barlow_music_weight_0.5_both_init_hubert",
    "TA-2L-CL-0.1": "task_vector_barlow_speech_weight_0.1_and_barlow_music_weight_0.9_both_init_hubert",
    # 3L L_KD
    "TA-3L-KD-0.9": "task_vector_dhubert_3layers_ls_960_weight_0.9_and_mert_3layers_ls_960_weight_0.1_both_init_hubert_both_same_seed",
    "TA-3L-KD-0.8": "task_vector_dhubert_3layers_ls_960_weight_0.8_and_mert_3layers_ls_960_weight_0.2_both_init_hubert_both_same_seed",
    "TA-3L-KD-0.5": "task_vector_dhubert_3layers_ls_960_weight_0.5_and_mert_3layers_ls_960_weight_0.5_both_init_hubert_both_same_seed",
    "TA-3L-KD-0.1": "task_vector_dhubert_3layers_ls_960_weight_0.1_and_mert_3layers_ls_960_weight_0.9_both_init_hubert_both_same_seed",
    # 3L L_CL
    "TA-3L-CL-0.9": "task_vector_barlow3l_hubert_weight_0.9_mert_weight_0.1",
    "TA-3L-CL-0.8": "task_vector_barlow3l_hubert_weight_0.8_mert_weight_0.2",
    "TA-3L-CL-0.5": "task_vector_barlow3l_hubert_weight_0.5_mert_weight_0.5",
    "TA-3L-CL-0.1": "task_vector_barlow3l_hubert_weight_0.1_mert_weight_0.9",
    # Wide L_KD
    "TA-W-KD-0.9": "task_vector_wide_l1_hubert_weight_0.9_mert_weight_0.1",
    "TA-W-KD-0.8": "task_vector_wide_l1_hubert_weight_0.8_mert_weight_0.2",
    "TA-W-KD-0.5": "task_vector_wide_l1_hubert_weight_0.5_mert_weight_0.5",
    "TA-W-KD-0.1": "task_vector_wide_l1_hubert_weight_0.1_mert_weight_0.9",
    # Wide L_CL
    "TA-W-CL-0.9": "task_vector_wide_barlow_hubert_weight_0.9_mert_weight_0.1",
    "TA-W-CL-0.8": "task_vector_wide_barlow_hubert_weight_0.8_mert_weight_0.2",
    "TA-W-CL-0.5": "task_vector_wide_barlow_hubert_weight_0.5_mert_weight_0.5",
    "TA-W-CL-0.1": "task_vector_wide_barlow_hubert_weight_0.1_mert_weight_0.9",
    # Pre-training data variants
    "TA-MS-0.9":  "task_vector_distilhubert_music4all_ls960_2layers_weight_0.9_and_distilmert_music4all_ls960_2layers_weight_0.1",
    "TA-MSA-0.9": "task_vector_dhubert_ls_960_audioset_music4all_weight_0.9_and_mert_ls_960_audioset_music4all_weight_0.1_both_init_hubert_both_same_seed",
    # ---- Chapter 5: Correlation-Permutation (distilled, shared init, fnn+attn fixed) ----
    "CP-2L-KD-0.9": "perm_2L_LKD_G1_fnn_attn_fixed_0.9_0.1",
    "CP-2L-KD-0.8": "perm_2L_LKD_G1_fnn_attn_fixed_0.8_0.2",
    "CP-2L-CL-0.9": "perm_2L_LCL_G1_fnn_attn_fixed_0.9_0.1",
    "CP-2L-CL-0.8": "perm_2L_LCL_G1_fnn_attn_fixed_0.8_0.2",
    "CP-3L-KD-0.9": "perm_3L_LKD_G1_fnn_attn_fixed_0.9_0.1",
    "CP-3L-KD-0.8": "perm_3L_LKD_G1_fnn_attn_fixed_0.8_0.2",
    "CP-3L-CL-0.9": "perm_3L_LCL_G1_fnn_attn_fixed_0.9_0.1",
    "CP-3L-CL-0.8": "perm_3L_LCL_G1_fnn_attn_fixed_0.8_0.2",
    "CP-W-CL-0.9":  "perm_2L_wide_LCL_G1_fnn_attn_fixed_0.9_0.1",
    "CP-W-CL-0.8":  "perm_2L_wide_LCL_G1_fnn_attn_fixed_0.8_0.2",
    # ---- Chapter 5: CP different init ----
    "CP-MI-0.9": "perm_2L_LKD_MI_G1_fnn_attn_fixed_0.9_0.1",
    "CP-MI-0.8": "perm_2L_LKD_MI_G1_fnn_attn_fixed_0.8_0.2",
}

# ============================================================================
#  Task definitions
# ============================================================================
# Each task has: kind (cls/asr), list of subdirectory patterns.
# For folded tasks, multiple directory patterns are tried to handle naming
# differences between NTU and NSCC.

TASK_DEFS = {
    "ASR": dict(kind="asr", dirs=["asr_paper_method"]),
    "KS":  dict(kind="cls", dirs=["speech_commands_paper_method"]),
    "IC":  dict(kind="cls_csv", dirs=["fluent_commands_paper_method"]),
    "ER":  dict(kind="cls_fold", n_folds=5,
                dir_patterns=[
                    "emotion_fold{fold}_paper_method",
                    "emotion_paper_method_fold{fold}",
                ]),
    "SingerID": dict(kind="cls", dirs=["vocalset_singer_id_paper_method"]),
    "VocID":    dict(kind="cls", dirs=["vocalset_technique_id_paper_method"]),
    "InstCls":  dict(kind="cls", dirs=["instrument_nsynth_paper_method"]),
    "GenreID":  dict(kind="cls", dirs=["genre_gtzan_paper_method"]),
    "ESC-50":   dict(kind="cls_fold", n_folds=5,
                     dir_patterns=[
                         "aec_esc50_paper_method_fold{fold}",
                     ]),
}

TASK_ORDER = ["ASR", "KS", "IC", "ER", "SingerID", "VocID", "InstCls",
              "GenreID", "ESC-50"]

# ============================================================================
#  Test pairs (23 pairs → 207 tests at 9 tasks each)
# ============================================================================
TEST_PAIRS = [
    # C4-1: TA vs Ensemble Distillation (Table 4.1)
    ("C4-1a", "C4-1", "TA-2L-KD-0.9 vs ENS-2L",  "TA-2L-KD-0.9", "ENS-2L"),
    ("C4-1b", "C4-1", "TA-3L-CL-0.9 vs ENS-3L",  "TA-3L-CL-0.9", "ENS-3L"),
    ("C4-1c", "C4-1", "TA-W-CL-0.9 vs ENS-W",    "TA-W-CL-0.9",  "ENS-W"),
    # C4-2: L_KD vs L_CL at key λ (Tables 4.2, 4.3, 4.6)
    ("C4-2a", "C4-2", "2L KD-0.8 vs CL-0.8",  "TA-2L-KD-0.8", "TA-2L-CL-0.8"),
    ("C4-2b", "C4-2", "2L KD-0.5 vs CL-0.5",  "TA-2L-KD-0.5", "TA-2L-CL-0.5"),
    ("C4-2c", "C4-2", "3L KD-0.8 vs CL-0.8",  "TA-3L-KD-0.8", "TA-3L-CL-0.8"),
    ("C4-2d", "C4-2", "3L KD-0.5 vs CL-0.5",  "TA-3L-KD-0.5", "TA-3L-CL-0.5"),
    ("C4-2e", "C4-2", "W KD-0.8 vs CL-0.8",   "TA-W-KD-0.8",  "TA-W-CL-0.8"),
    ("C4-2f", "C4-2", "W KD-0.5 vs CL-0.5",   "TA-W-KD-0.5",  "TA-W-CL-0.5"),
    # C4-3: Pre-training data (Table 4.4)
    ("C4-3a", "C4-3", "S vs M+S (0.9)",   "TA-2L-KD-0.9", "TA-MS-0.9"),
    ("C4-3b", "C4-3", "S vs M+S+A (0.9)", "TA-2L-KD-0.9", "TA-MSA-0.9"),
    # C5-1: CP vs TA on distilled models (Table 5.3)
    ("C5-1a", "C5-1", "TA vs CP 2L-KD-0.9", "TA-2L-KD-0.9", "CP-2L-KD-0.9"),
    ("C5-1b", "C5-1", "TA vs CP 2L-KD-0.8", "TA-2L-KD-0.8", "CP-2L-KD-0.8"),
    ("C5-1c", "C5-1", "TA vs CP 2L-CL-0.9", "TA-2L-CL-0.9", "CP-2L-CL-0.9"),
    ("C5-1d", "C5-1", "TA vs CP 2L-CL-0.8", "TA-2L-CL-0.8", "CP-2L-CL-0.8"),
    ("C5-1e", "C5-1", "TA vs CP 3L-KD-0.9", "TA-3L-KD-0.9", "CP-3L-KD-0.9"),
    ("C5-1f", "C5-1", "TA vs CP 3L-KD-0.8", "TA-3L-KD-0.8", "CP-3L-KD-0.8"),
    ("C5-1g", "C5-1", "TA vs CP 3L-CL-0.9", "TA-3L-CL-0.9", "CP-3L-CL-0.9"),
    ("C5-1h", "C5-1", "TA vs CP 3L-CL-0.8", "TA-3L-CL-0.8", "CP-3L-CL-0.8"),
    ("C5-1i", "C5-1", "TA vs CP W-CL-0.9",  "TA-W-CL-0.9",  "CP-W-CL-0.9"),
    ("C5-1j", "C5-1", "TA vs CP W-CL-0.8",  "TA-W-CL-0.8",  "CP-W-CL-0.8"),
    # C5-2: Different-init vs shared-init (Table 5.4)
    ("C5-2a", "C5-2", "shared vs diff init 0.9", "CP-2L-KD-0.9", "CP-MI-0.9"),
    ("C5-2b", "C5-2", "shared vs diff init 0.8", "CP-2L-KD-0.8", "CP-MI-0.8"),
]


# ============================================================================
#  File resolution
# ============================================================================
def _model_bases(model_dir):
    """Return all existing '<root>/<model_dir>' paths across SEARCH_ROOTS."""
    if not model_dir:
        return []
    bases = []
    for root in SEARCH_ROOTS:
        cand = os.path.join(root, model_dir)
        if os.path.isdir(cand):
            bases.append(cand)
        for h in sorted(glob.glob(os.path.join(root, "*", model_dir))):
            if os.path.isdir(h):
                bases.append(h)
    return bases


def _find_dir(model_dir):
    b = _model_bases(model_dir)
    return b[0] if b else None


def _split_rank(fname):
    b = os.path.basename(fname)
    if b.startswith("train"):
        return None
    if b.startswith("test"):
        return 0
    if b.startswith("valid"):
        return 1
    if b.startswith("dev"):
        return 2
    return 3


def _cls_files_txt(model_dir, task_dir):
    """Find (predict_txt, truth_txt) for a standard classification task."""
    for base in _model_bases(model_dir):
        td = os.path.join(base, task_dir)
        cands = []
        for p in glob.glob(os.path.join(td, "*_predict.txt")):
            t = p[:-len("_predict.txt")] + "_truth.txt"
            r = _split_rank(p)
            if r is not None and os.path.isfile(t):
                cands.append((r, p, t))
        if cands:
            cands.sort()
            return (cands[0][1], cands[0][2])
    return None


def _cls_files_csv(model_dir, task_dir):
    """Find (predict_csv, truth_csv) for IC which uses CSV format."""
    for base in _model_bases(model_dir):
        td = os.path.join(base, task_dir)
        cands = []
        for ext in ["csv", "txt"]:
            for p in glob.glob(os.path.join(td, f"*_predict.{ext}")):
                t = p[:-len(f"_predict.{ext}")] + f"_truth.{ext}"
                r = _split_rank(p)
                if r is not None and os.path.isfile(t):
                    cands.append((r, p, t, ext))
        if cands:
            cands.sort()
            return (cands[0][1], cands[0][2], cands[0][3])
    return None


def _fold_files(model_dir, dir_patterns, fold):
    """Find prediction files for a specific fold, trying all naming patterns."""
    for pattern in dir_patterns:
        task_dir = pattern.format(fold=fold)
        result = _cls_files_txt(model_dir, task_dir)
        if result is not None:
            return result
    return None


def _asr_files(model_dir, task_dir):
    for base in _model_bases(model_dir):
        td = os.path.join(base, task_dir)
        hyp = os.path.join(td, "test-clean-noLM-hyp.ark")
        ref = os.path.join(td, "test-clean-noLM-ref.ark")
        if os.path.isfile(hyp) and os.path.isfile(ref):
            return (hyp, ref)
        hyp2 = os.path.join(td, "dev-clean-noLM-hyp.ark")
        ref2 = os.path.join(td, "dev-clean-noLM-ref.ark")
        if os.path.isfile(hyp2) and os.path.isfile(ref2):
            return (hyp2, ref2)
    return None


# ============================================================================
#  Parsers
# ============================================================================
def _parse_kv(path):
    """'<id> <label...>' → {id: label}."""
    d = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) == 1:
                d[parts[0]] = ""
            else:
                d[parts[0]] = parts[1].strip()
    return d


def _parse_csv(path):
    """Parse CSV prediction/truth file: first field = id, rest = label."""
    d = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            uid = row[0].strip()
            label = ",".join(f.strip() for f in row[1:])
            d[uid] = label
    return d


def correctness(predict_path, truth_path, fmt="txt"):
    if fmt == "csv":
        pred = _parse_csv(predict_path)
        truth = _parse_csv(truth_path)
    else:
        pred = _parse_kv(predict_path)
        truth = _parse_kv(truth_path)
    ids = sorted(set(pred) & set(truth))
    return {i: int(pred[i] == truth[i]) for i in ids}, len(pred), len(truth)


def _word_edit_distance(ref, hyp):
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def asr_per_utt(hyp_path, ref_path):
    hyp = _parse_kv(hyp_path)
    ref = _parse_kv(ref_path)
    out = {}
    for u in set(hyp) & set(ref):
        r = ref[u].split()
        h = hyp[u].split()
        out[u] = (_word_edit_distance(r, h), len(r))
    return out


# ============================================================================
#  Statistical tests
# ============================================================================
def mcnemar_test(corr_a, corr_b):
    ids = sorted(set(corr_a) & set(corr_b))
    n = len(ids)
    a_arr = np.fromiter((corr_a[i] for i in ids), dtype=np.int8, count=n)
    b_arr = np.fromiter((corr_b[i] for i in ids), dtype=np.int8, count=n)
    n10 = int(np.sum((a_arr == 1) & (b_arr == 0)))
    n01 = int(np.sum((a_arr == 0) & (b_arr == 1)))
    disc = n10 + n01
    acc_a = float(a_arr.mean()) if n else float("nan")
    acc_b = float(b_arr.mean()) if n else float("nan")
    if disc == 0:
        stat, p, kind = 0.0, 1.0, "exact"
    elif disc < 25:
        kind = "exact"
        k = min(n10, n01)
        if _HAVE_SCIPY:
            p = float(_binomtest(k, disc, 0.5).pvalue)
        else:
            p = min(1.0, 2.0 * sum(math.comb(disc, j) for j in range(k + 1))
                    * (0.5 ** disc))
        stat = float(k)
    else:
        kind = "chi2_cc"
        stat = (abs(n10 - n01) - 1.0) ** 2 / disc
        if _HAVE_SCIPY:
            p = float(_chi2.sf(stat, 1))
        else:
            p = math.erfc(math.sqrt(stat / 2.0))
    if n:
        cohen_h = float(2 * math.asin(math.sqrt(max(acc_a, 0.0)))
                        - 2 * math.asin(math.sqrt(max(acc_b, 0.0))))
    else:
        cohen_h = float("nan")
    odds_ratio = (n10 + 0.5) / (n01 + 0.5)
    if n:
        d = (n10 - n01) / n
        se = math.sqrt(max((n10 + n01) - (n10 - n01) ** 2 / n, 0.0)) / n
        ci_lo, ci_hi = 100 * (d - 1.96 * se), 100 * (d + 1.96 * se)
    else:
        ci_lo = ci_hi = float("nan")
    return dict(n=n, acc_a=acc_a, acc_b=acc_b, n10=n10, n01=n01,
                disc=disc, stat=stat, p=p, kind=kind,
                metric_a=100 * acc_a, metric_b=100 * acc_b,
                delta=100 * (acc_a - acc_b),
                cohen_h=cohen_h, odds_ratio=odds_ratio,
                ci_lo=ci_lo, ci_hi=ci_hi)


def bootstrap_asr(per_utt_a, per_utt_b, iters=BOOTSTRAP_ITERS,
                  seed=BOOTSTRAP_SEED):
    utts = sorted(set(per_utt_a) & set(per_utt_b))
    n = len(utts)
    eA = np.array([per_utt_a[u][0] for u in utts], dtype=np.float64)
    rA = np.array([per_utt_a[u][1] for u in utts], dtype=np.float64)
    eB = np.array([per_utt_b[u][0] for u in utts], dtype=np.float64)
    rB = np.array([per_utt_b[u][1] for u in utts], dtype=np.float64)
    wer_a = eA.sum() / rA.sum()
    wer_b = eB.sum() / rB.sum()
    d0 = wer_a - wer_b
    rng = np.random.default_rng(seed)
    deltas = np.empty(iters, dtype=np.float64)
    for k in range(iters):
        idx = rng.integers(0, n, n)
        wa = eA[idx].sum() / rA[idx].sum()
        wb = eB[idx].sum() / rB[idx].sum()
        deltas[k] = wa - wb
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2.0 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
    p = float(min(1.0, p))
    return dict(n=n, metric_a=100 * wer_a, metric_b=100 * wer_b,
                delta=100 * d0, ci_lo=100 * lo, ci_hi=100 * hi,
                stat=float("nan"), p=p, kind="paired_bootstrap")


# ============================================================================
#  Benjamini-Hochberg
# ============================================================================
def benjamini_hochberg(pvals, q=BH_Q):
    m = len(pvals)
    if m == 0:
        return [], []
    order = sorted(range(m), key=lambda i: pvals[i])
    qadj = [0.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = pvals[i] * m / (rank + 1)
        prev = min(prev, val)
        qadj[i] = min(1.0, prev)
    reject = [qadj[i] <= q for i in range(m)]
    return reject, qadj


# ============================================================================
#  Per-task test runner
# ============================================================================
def _run_one_task(model_a_dir, model_b_dir, task_name, tdef):
    """Run a single task test. Returns (status, result_dict_or_reason)."""
    kind = tdef["kind"]

    if kind == "cls":
        task_dir = tdef["dirs"][0]
        fa = _cls_files_txt(model_a_dir, task_dir)
        fb = _cls_files_txt(model_b_dir, task_dir)
        if fa is None:
            return "NOT TESTED", f"predict files missing for A in {task_dir}"
        if fb is None:
            return "NOT TESTED", f"predict files missing for B in {task_dir}"
        ca, _, _ = correctness(fa[0], fa[1])
        cb, _, _ = correctness(fb[0], fb[1])
        return "OK", mcnemar_test(ca, cb)

    elif kind == "cls_csv":
        task_dir = tdef["dirs"][0]
        fa = _cls_files_csv(model_a_dir, task_dir)
        fb = _cls_files_csv(model_b_dir, task_dir)
        if fa is None:
            return "NOT TESTED", f"predict files missing for A in {task_dir}"
        if fb is None:
            return "NOT TESTED", f"predict files missing for B in {task_dir}"
        ca, _, _ = correctness(fa[0], fa[1], fmt=fa[2])
        cb, _, _ = correctness(fb[0], fb[1], fmt=fb[2])
        return "OK", mcnemar_test(ca, cb)

    elif kind == "cls_fold":
        corr_a, corr_b = {}, {}
        missing = []
        for fold in range(1, tdef["n_folds"] + 1):
            fa = _fold_files(model_a_dir, tdef["dir_patterns"], fold)
            fb = _fold_files(model_b_dir, tdef["dir_patterns"], fold)
            if fa is None:
                missing.append(f"A/fold{fold}")
                continue
            if fb is None:
                missing.append(f"B/fold{fold}")
                continue
            ca, _, _ = correctness(fa[0], fa[1])
            cb, _, _ = correctness(fb[0], fb[1])
            pref = f"fold{fold}/"
            corr_a.update({pref + k: v for k, v in ca.items()})
            corr_b.update({pref + k: v for k, v in cb.items()})
        if not corr_a or not corr_b:
            return "NOT TESTED", f"fold files missing: {', '.join(missing)}"
        if missing:
            pass  # partial folds — proceed with what we have
        return "OK", mcnemar_test(corr_a, corr_b)

    elif kind == "asr":
        task_dir = tdef["dirs"][0]
        fa = _asr_files(model_a_dir, task_dir)
        fb = _asr_files(model_b_dir, task_dir)
        if fa is None:
            return "NOT TESTED", f".ark files missing for A in {task_dir}"
        if fb is None:
            return "NOT TESTED", f".ark files missing for B in {task_dir}"
        pa = asr_per_utt(*fa)
        pb = asr_per_utt(*fb)
        return "OK", bootstrap_asr(pa, pb)

    return "NOT TESTED", f"unknown task kind: {kind}"


# ============================================================================
#  Driver
# ============================================================================
def run():
    results = []
    for pair_id, group, label, code_a, code_b in TEST_PAIRS:
        dir_a = MODELS[code_a]
        dir_b = MODELS[code_b]
        loc_a = _find_dir(dir_a)
        loc_b = _find_dir(dir_b)

        for task_name in TASK_ORDER:
            rec = dict(
                id=f"{pair_id}/{task_name}",
                prio=group,
                label=label,
                task=task_name,
                test="bootstrap" if task_name == "ASR" else "mcnemar",
                A=code_a, B=code_b,
                A_dir=dir_a, B_dir=dir_b,
            )

            if not loc_a:
                rec.update(status="NOT TESTED",
                           reason=f"model A ({code_a}) dir not found: {dir_a}")
                results.append(rec)
                continue
            if not loc_b:
                rec.update(status="NOT TESTED",
                           reason=f"model B ({code_b}) dir not found: {dir_b}")
                results.append(rec)
                continue

            tdef = TASK_DEFS[task_name]
            try:
                status, payload = _run_one_task(dir_a, dir_b, task_name, tdef)
                if status == "OK":
                    rec.update(status="OK", **payload)
                else:
                    rec.update(status="NOT TESTED", reason=payload)
            except Exception as e:
                rec.update(status="ERROR", reason=repr(e))
            results.append(rec)

    # ---- Benjamini-Hochberg within each priority group --------------------
    by_group = defaultdict(list)
    for i, r in enumerate(results):
        if r.get("status") == "OK":
            by_group[r["prio"]].append(i)
    for prio, idxs in by_group.items():
        rej, qadj = benjamini_hochberg([results[i]["p"] for i in idxs])
        for j, i in enumerate(idxs):
            results[i]["bh_p"] = qadj[j]
            results[i]["sig"] = rej[j]

    _write_outputs(results)
    return results


def _fmt(x, nd=2):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) \
        else f"{x:.{nd}f}"


def _write_outputs(results):
    out_md = os.path.join(PROJECT_ROOT, "reports", "stat_tests",
                          "RESULTS_ch4_ch5.md")
    out_tsv = os.path.join(PROJECT_ROOT, "reports", "stat_tests",
                           "RESULTS_ch4_ch5.tsv")
    ok = [r for r in results if r.get("status") == "OK"]
    nt = [r for r in results if r.get("status") != "OK"]

    lines = []
    lines.append("# Chapters 4 & 5 — Statistical Test Results\n")
    lines.append(f"- Tests defined: **{len(results)}**  |  "
                 f"computed: **{len(ok)}**  |  "
                 f"not tested: **{len(nt)}**")
    lines.append(f"- ASR bootstrap = {BOOTSTRAP_ITERS} iters  |  "
                 f"BH FDR q = {BH_Q} (within each priority group)")
    lines.append(f"- scipy available: {_HAVE_SCIPY}\n")

    # ---- 1. Computed tests ------------------------------------------------
    lines.append("## 1. Computed tests\n")
    lines.append("D = A-B (Acc% for classification, WER% for ASR). "
                 "95% CI is on D. Effect size: Cohen's h (classification) "
                 "or bootstrap CI (ASR). "
                 "dag = significant at BH-FDR q={:.2f}.\n".format(BH_Q))
    lines.append("| ID | Group | Comparison | Task | Test | "
                 "Metric A | Metric B | D (A-B) | 95% CI | Effect | "
                 "n | raw p | BH p | Sig |")
    lines.append("|----|-------|-----------|------|------|"
                 "---------|---------|--------|--------|--------|"
                 "---|-------|------|-----|")
    for r in sorted(ok, key=lambda x: (x["prio"], x["id"])):
        unit = "WER%" if r["test"] == "bootstrap" else "Acc%"
        sig = "**dag**" if r.get("sig") else "no"
        ci = (f"[{_fmt(r.get('ci_lo'))}, {_fmt(r.get('ci_hi'))}]"
              if r.get("ci_lo") is not None else "")
        if r["test"] == "bootstrap":
            eff = f"DWER {_fmt(r['delta'])} pp"
        else:
            eff = (f"h={_fmt(r.get('cohen_h'),3)}, "
                   f"OR={_fmt(r.get('odds_ratio'),2)}")
        lines.append(
            f"| {r['id']} | {r['prio']} | {r['label']} | {r['task']} | "
            f"{r['kind']} | "
            f"{_fmt(r['metric_a'])} {unit} | {_fmt(r['metric_b'])} {unit} | "
            f"{_fmt(r['delta'])} | {ci} | {eff} | "
            f"{r['n']} | {_fmt(r['p'],4)} | "
            f"{_fmt(r.get('bh_p'),4)} | {sig} |")

    # ---- Per-group BH summary ---------------------------------------------
    lines.append("\n### Per-group Benjamini-Hochberg summary\n")
    by_g = defaultdict(list)
    for r in ok:
        by_g[r["prio"]].append(r)
    for g in sorted(by_g):
        nsig = sum(1 for r in by_g[g] if r.get("sig"))
        ntot = len(by_g[g])
        lines.append(f"- **{g}**: {nsig}/{ntot} significant after BH "
                     f"(q={BH_Q})")

    # ---- Per-group per-pair summary (significant tasks for each pair) ------
    lines.append("\n### Per-pair significant tasks\n")
    for pair_id, group, label, code_a, code_b in TEST_PAIRS:
        pair_ok = [r for r in ok
                   if r["id"].startswith(pair_id + "/")]
        if not pair_ok:
            lines.append(f"- **{pair_id}** ({label}): no tests computed")
            continue
        sig_tasks = [r["task"] for r in pair_ok if r.get("sig")]
        nonsig_tasks = [r["task"] for r in pair_ok if not r.get("sig")]
        n_total = len(pair_ok)
        if sig_tasks:
            lines.append(f"- **{pair_id}** ({label}): "
                         f"{len(sig_tasks)}/{n_total} sig — "
                         f"{', '.join(sig_tasks)}")
        else:
            lines.append(f"- **{pair_id}** ({label}): "
                         f"0/{n_total} significant")

    # ---- 2. NOT TESTED ----------------------------------------------------
    lines.append("\n## 2. NOT TESTED (missing predictions)\n")
    # Summarize by pair to avoid 100+ row table
    nt_by_pair = defaultdict(list)
    for r in nt:
        pid = r["id"].rsplit("/", 1)[0]
        nt_by_pair[pid].append(r)
    lines.append("| Pair | Group | Comparison | Missing tasks | Reason sample |")
    lines.append("|------|-------|-----------|---------------|---------------|")
    for pid in sorted(nt_by_pair):
        recs = nt_by_pair[pid]
        tasks = ", ".join(r["task"] for r in recs)
        reason = recs[0].get("reason", "")[:80]
        lines.append(f"| {pid} | {recs[0]['prio']} | {recs[0]['label']} | "
                     f"{tasks} | {reason} |")

    # ---- 3. Interpretation ------------------------------------------------
    lines.append("\n## 3. Interpretation\n")
    if not ok:
        lines.append("_No tests computed yet — populate the prediction tree "
                     "and re-run._\n")
    else:
        for g in sorted(by_g):
            lines.append(f"\n### {g}\n")
            for r in sorted(by_g[g], key=lambda x: x["id"]):
                if r["test"] == "bootstrap":
                    crosses0 = (r.get("ci_lo", 0) <= 0 <= r.get("ci_hi", 0))
                    small = abs(r["delta"]) < 0.3
                    if r.get("sig") and not crosses0 and not small:
                        conf = "**significant** WER difference"
                    elif small:
                        conf = "**negligible** |DWER| < 0.3 pp"
                    else:
                        conf = "not significant"
                    lines.append(
                        f"- {r['id']}: WER {_fmt(r['metric_a'])} vs "
                        f"{_fmt(r['metric_b'])} (D {_fmt(r['delta'])} pp, "
                        f"CI [{_fmt(r.get('ci_lo'))}, {_fmt(r.get('ci_hi'))}]) "
                        f"— {conf}")
                else:
                    h = abs(r.get("cohen_h") or 0.0)
                    mag = ("negligible" if h < 0.2 else "small" if h < 0.5
                           else "medium" if h < 0.8 else "large")
                    if r.get("sig"):
                        conf = f"**significant** ({mag} effect)"
                    elif h < 0.2:
                        conf = f"not significant, {mag} effect"
                    else:
                        conf = f"not significant but {mag} effect (trend)"
                    lines.append(
                        f"- {r['id']}: {_fmt(r['metric_a'])}% vs "
                        f"{_fmt(r['metric_b'])}% (D {_fmt(r['delta'])} pp, "
                        f"h={_fmt(r.get('cohen_h'),3)}) — {conf}")

    lines.append(
        "\n**Threats to validity.** (i) Single pre-training seed: between-seed "
        "variance is unobserved. (ii) ER and ESC-50 pool the 5 CV folds; the "
        "pooled McNemar treats folds as one test set and ignores fold-level "
        "correlation (mildly anti-conservative). (iii) McNemar conditions on "
        "discordant pairs; with few discordant samples, power is low.\n")

    # ---- 4. Model locations -----------------------------------------------
    lines.append("\n## 4. Model locations\n")
    lines.append("| Code | Directory | Found at |")
    lines.append("|------|-----------|----------|")
    seen = set()
    for code in sorted(MODELS):
        d = MODELS[code]
        if d in seen:
            continue
        seen.add(d)
        loc = _find_dir(d)
        lines.append(f"| {code} | `{d[:60]}{'...' if len(d)>60 else ''}` | "
                     f"{loc or '**NOT FOUND**'} |")

    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    cols = ["id", "prio", "label", "task", "test", "status",
            "metric_a", "metric_b", "delta", "ci_lo", "ci_hi",
            "cohen_h", "odds_ratio", "n10", "n01", "disc",
            "n", "stat", "p", "bh_p", "sig",
            "kind", "A", "B", "A_dir", "B_dir", "reason"]
    with open(out_tsv, "w", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in results:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    print("\n".join(lines))
    print(f"\n[written] {out_md}")
    print(f"[written] {out_tsv}")


if __name__ == "__main__":
    run()
