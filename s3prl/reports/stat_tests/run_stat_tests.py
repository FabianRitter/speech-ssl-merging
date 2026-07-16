#!/usr/bin/env python3
# =============================================================================
#  Chapter 3 statistical testing  —  STEP 3  (RUN THIS ON NSCC)
# =============================================================================
#
#  Implements the 31 tests defined in reports/statistical_testing_plan.md:
#
#    * Classification tasks (IC, KS, ER, ESC-50, VocID)  -> McNemar's test
#        - continuity-corrected chi^2 when discordant n >= 25
#        - exact binomial when discordant n < 25
#    * ASR                                               -> paired bootstrap
#        - per-utterance word-level edit distance, 10 000 resamples
#    * Multi-fold tasks (ER = emotion 5-fold,
#                        ESC-50 = aec_esc50 5-fold)      -> samples pooled
#                                                           across folds
#    * Multiple-testing control -> Benjamini-Hochberg (FDR q = 0.05)
#                                  applied within each priority group
#
#  It degrades gracefully: every test whose checkpoint / prediction files are
#  missing is reported as NOT TESTED with the reason and the expected location,
#  so the output also doubles as the "what is still missing" report.
#
#  Run:
#      cd <repo>/s3prl
#      python reports/stat_tests/run_stat_tests.py
#
#  Outputs:
#      reports/stat_tests/RESULTS.md      (main table + BH summary)
#      reports/stat_tests/RESULTS.tsv     (machine-readable)
# =============================================================================
import os
import sys
import glob
import math
import json
import random
from collections import defaultdict

import numpy as np

try:
    from scipy.stats import chi2 as _chi2, binomtest as _binomtest
    _HAVE_SCIPY = True
except Exception:                                       # pragma: no cover
    _HAVE_SCIPY = False

PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
STAGE = os.path.join(PROJECT_ROOT, "reports", "stat_tests", "predictions")

# Roots searched, in priority order, for "<model_dir>/<task_subpath>".
SEARCH_ROOTS = [
    os.path.join(STAGE, "nscc_local"),
    os.path.join(STAGE, "ntu", "downstream-thesis-pending-exps"),
    os.path.join(STAGE, "ntu", "downstream"),
    os.path.join(PROJECT_ROOT, "result", "downstream"),
]

BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_SEED = 1234
BH_Q = 0.05
NOISE_COND = "chime"                 # the "noisy" condition used in the thesis

# ----------------------------------------------------------------------------
#  Model registry  (code -> directory name).  None = checkpoint not available.
#  M5/M6/M7 dir names can be overridden via env vars once located on NTU.
# ----------------------------------------------------------------------------
MODELS = {
    "M1": dict(  # L_KD, Setup 2, HuBERT, fixed
        name="DistilHuBERT L_KD (Setup2, HuBERT, fixed)",
        dir="setup2_2-dis_t_nocont_teacher_100hourslibri_l1-2026"),
    "M2": dict(  # L_CL full, Setup 2, HuBERT, fixed
        name="DistilHuBERT L_CL full (Setup2, HuBERT, fixed)",
        dir="setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_"
            "with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct"),
    "M3": dict(  # CC-only, Setup 2, HuBERT, fixed
        name="DistilHuBERT CC-only (Setup2, HuBERT, fixed)",
        dir="barlow_old_setup2_2-dis_nocont_teacher_100hourslibri-2026-"
            "noselfcorr"),
    "M4": dict(  # Heuristic weights -- NOT FOUND anywhere
        name="DistilHuBERT L_CL heuristic weights (Setup2, HuBERT)",
        dir=os.environ.get("M4_DIR", "")),
    "M5": dict(  # L_KD, Setup 2, HuBERT+, fixed
        name="DistilHuBERT L_KD (Setup2, HuBERT+, fixed)",
        dir=os.environ.get("M5_DIR", "")),
    "M6": dict(  # L_CL, Setup 2, HuBERT+, fixed
        name="DistilHuBERT L_CL (Setup2, HuBERT+, fixed)",
        dir=os.environ.get("M6_DIR", "")),
    "M7": dict(  # L_CL, Setup 2, HuBERT, SAME noise teacher+student
        name="DistilHuBERT L_CL same-noise (Setup2, HuBERT)",
        dir=os.environ.get(
            "M7_DIR",
            "barlow_old_setup2_2-dis_nocont_teacher_100hours-"
            "same-noise-teacher-and-student")),
}

# Task -> (kind, directory(ies)).  For folded tasks the dir holds {fold}.
TASKS = {
    "IC":     dict(kind="cls", dirs=["fluent_commands_paper_method"]),
    "KS":     dict(kind="cls", dirs=["speech_commands_paper_method"]),
    "VocID":  dict(kind="cls", dirs=["vocalset_technique_id_paper_method"]),
    "ER":     dict(kind="cls",
                   dirs=[f"emotion_paper_method_fold{i}" for i in range(1, 6)]),
    "ESC-50": dict(kind="cls",
                   dirs=[f"aec_esc50_paper_method_fold{i}" for i in range(1, 6)]),
    "ASR":    dict(kind="asr", dirs=["asr_paper_method"]),
}

# ----------------------------------------------------------------------------
#  The 31 tests.  (id, priority, label, modelA, modelB, task, condition, test)
#  condition in {clean, noisy}.  test in {mcnemar, bootstrap, probe}.
# ----------------------------------------------------------------------------
def _expand(prefix, prio, label, a, b, task, conds, test):
    return [dict(id=f"{prefix}", prio=prio, label=label, A=a, B=b,
                 task=task, cond=c, test=test) for c in conds]

TESTS = []
# ---- Priority 1 : core contribution  L_KD vs L_CL (M1 vs M2) ---------------
TESTS += _expand("P1-1", "P1", "L_KD vs L_CL", "M1", "M2", "IC",
                 ["clean", "noisy"], "mcnemar")
TESTS += _expand("P1-2", "P1", "L_KD vs L_CL", "M1", "M2", "KS",
                 ["clean", "noisy"], "mcnemar")
TESTS += _expand("P1-3", "P1", "L_KD vs L_CL", "M1", "M2", "ER",
                 ["clean", "noisy"], "mcnemar")
TESTS += _expand("P1-4", "P1", "L_KD vs L_CL", "M1", "M2", "ASR",
                 ["clean", "noisy"], "bootstrap")
# ---- Priority 2 : teacher-agnostic  L_KD vs L_CL (HuBERT+, M5 vs M6) -------
TESTS += _expand("P2-1", "P2", "L_KD vs L_CL (HuBERT+)", "M5", "M6", "IC",
                 ["clean", "noisy"], "mcnemar")
TESTS += _expand("P2-2", "P2", "L_KD vs L_CL (HuBERT+)", "M5", "M6", "ER",
                 ["clean", "noisy"], "mcnemar")
# ---- Priority 3 : self-correlation ablation  CC-only vs full (M3 vs M2) ----
TESTS += _expand("P3-1", "P3", "CC-only vs full L_CL", "M3", "M2", "IC",
                 ["noisy"], "mcnemar")
TESTS += _expand("P3-2", "P3", "CC-only vs full L_CL", "M3", "M2", "ER",
                 ["noisy"], "mcnemar")
TESTS += _expand("P3-3", "P3", "CC-only vs full L_CL", "M3", "M2", "ASR",
                 ["noisy"], "bootstrap")
TESTS += _expand("P3-4", "P3", "CC-only vs full L_CL", "M3", "M2", "KS",
                 ["noisy"], "mcnemar")
# ---- Priority 4 : heuristic weighting  Fixed vs Heuristic (M2 vs M4) -------
TESTS += _expand("P4-1", "P4", "Fixed vs Heuristic", "M2", "M4", "IC",
                 ["clean", "noisy"], "mcnemar")
TESTS += _expand("P4-2", "P4", "Fixed vs Heuristic", "M2", "M4", "KS",
                 ["clean", "noisy"], "mcnemar")
TESTS += _expand("P4-3", "P4", "Fixed vs Heuristic", "M2", "M4", "ASR",
                 ["clean", "noisy"], "bootstrap")
# ---- Priority 5 : cross-domain transfer  L_KD vs L_CL (M1 vs M2) ----------
TESTS += _expand("P5-1", "P5", "L_KD vs L_CL", "M1", "M2", "ESC-50",
                 ["clean", "noisy"], "mcnemar")
TESTS += _expand("P5-2", "P5", "L_KD vs L_CL", "M1", "M2", "VocID",
                 ["clean", "noisy"], "mcnemar")
# ---- Priority 6 : mechanistic analysis ------------------------------------
# P6-1/2/3 operate on LR-768 noise-probe per-sample predictions, which are
# NOT saved as artefacts -> reported NOT TESTED (needs probe re-run).
TESTS += [dict(id="P6-1", prio="P6", label="L_KD vs L_CL (LR-768 probe)",
               A="M1", B="M2", task="PROBE", cond="probe", test="probe")]
TESTS += [dict(id="P6-2", prio="P6", label="CC-only vs full (LR-768 probe)",
               A="M3", B="M2", task="PROBE", cond="probe", test="probe")]
TESTS += [dict(id="P6-3", prio="P6",
               label="same-noise vs indep (LR-768 probe)",
               A="M7", B="M2", task="PROBE", cond="probe", test="probe")]
# P6-4 : same-noise vs independent-noise L_CL on the IC downstream task
TESTS += _expand("P6-4", "P6", "same-noise vs indep (IC downstream)",
                 "M7", "M2", "IC", ["clean", "noisy"], "mcnemar")

assert len(TESTS) == 31, f"expected 31 tests, built {len(TESTS)}"


# ----------------------------------------------------------------------------
#  File resolution
# ----------------------------------------------------------------------------
def _model_bases(model_dir):
    """Return ALL existing '<root>/<model_dir>' across SEARCH_ROOTS, in
    priority order.  The same model's tasks may be split across roots
    (e.g. music tasks local on NSCC, speech/ER on the NTU upload), so the
    per-(model,task) lookup must consider every base, not just the first."""
    if not model_dir:
        return []
    bases = []
    for root in SEARCH_ROOTS:
        cand = os.path.join(root, model_dir)
        if os.path.isdir(cand):
            bases.append(cand)
        # tolerate the upload landing one level deeper
        for h in sorted(glob.glob(os.path.join(root, "*", model_dir))):
            if os.path.isdir(h):
                bases.append(h)
    return bases


def _find_dir(model_dir):
    """First base that exists (used for the model-location report)."""
    b = _model_bases(model_dir)
    return b[0] if b else None


def _split_rank(fname):
    """Rank prediction files by evaluation split. Lower = preferred.
    test_* (incl. test_fold3_*) > valid_* (genre) > dev_*; never 'train_*'."""
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


def _cls_files(model_dir, task_dir, cond):
    """Find (predict, truth). Handles generic `test_predict.txt`, genre's
    `valid_predict.txt`, and emotion's fold-suffixed
    `test_fold{N}_predict.txt` uniformly by globbing and ranking."""
    for base in _model_bases(model_dir):
        td = os.path.join(base, task_dir)
        if cond == "noisy":
            td = os.path.join(td, "evaluation", NOISE_COND)
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


def _asr_files(model_dir, task_dir, cond):
    for base in _model_bases(model_dir):
        td = os.path.join(base, task_dir)
        if cond == "noisy":
            td = os.path.join(td, "evaluation", NOISE_COND)
        hyp = os.path.join(td, "test-clean-noLM-hyp.ark")
        ref = os.path.join(td, "test-clean-noLM-ref.ark")
        if os.path.isfile(hyp) and os.path.isfile(ref):
            return (hyp, ref)
    return None


# ----------------------------------------------------------------------------
#  Parsers
# ----------------------------------------------------------------------------
def _parse_kv(path):
    """'<id> <label...>' -> {id: label}.  id = 1st token, label = remainder."""
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


def correctness(predict_path, truth_path):
    """Return dict id -> 0/1 correctness."""
    pred = _parse_kv(predict_path)
    truth = _parse_kv(truth_path)
    ids = sorted(set(pred) & set(truth))
    return {i: int(pred[i] == truth[i]) for i in ids}, len(pred), len(truth)


def _word_edit_distance(ref, hyp):
    """Levenshtein on token lists -> total errors (S+D+I)."""
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
    """Return {utt: (errors, ref_word_count)}."""
    hyp = _parse_kv(hyp_path)
    ref = _parse_kv(ref_path)
    out = {}
    for u in set(hyp) & set(ref):
        r = ref[u].split()
        h = hyp[u].split()
        out[u] = (_word_edit_distance(r, h), len(r))
    return out


# ----------------------------------------------------------------------------
#  Statistical tests
# ----------------------------------------------------------------------------
def mcnemar_test(corr_a, corr_b):
    """McNemar on paired per-sample correctness dicts."""
    ids = sorted(set(corr_a) & set(corr_b))
    n = len(ids)
    a_arr = np.fromiter((corr_a[i] for i in ids), dtype=np.int8, count=n)
    b_arr = np.fromiter((corr_b[i] for i in ids), dtype=np.int8, count=n)
    n10 = int(np.sum((a_arr == 1) & (b_arr == 0)))   # A right, B wrong
    n01 = int(np.sum((a_arr == 0) & (b_arr == 1)))   # A wrong, B right
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
    # ---- effect sizes (a p-value without an effect size is not reportable)
    # Cohen's h on the two marginal accuracies (small .2 / medium .5 / large .8)
    if n:
        cohen_h = float(2 * math.asin(math.sqrt(max(acc_a, 0.0)))
                        - 2 * math.asin(math.sqrt(max(acc_b, 0.0))))
    else:
        cohen_h = float("nan")
    # Paired odds ratio from the discordant cells (Haldane-corrected)
    odds_ratio = (n10 + 0.5) / (n01 + 0.5)
    # 95% CI on the paired accuracy difference d = (n10 - n01)/N
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
    """Paired bootstrap over utterances (Koehn 2004) on corpus WER."""
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


# ----------------------------------------------------------------------------
#  Benjamini-Hochberg
# ----------------------------------------------------------------------------
def benjamini_hochberg(pvals, q=BH_Q):
    """Return (reject[bool], qadj[float]) aligned with input order."""
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


# ----------------------------------------------------------------------------
#  Driver
# ----------------------------------------------------------------------------
def run():
    results = []
    for t in TESTS:
        rec = dict(t)
        ma, mb = MODELS[t["A"]], MODELS[t["B"]]
        rec["A_dir"] = ma["dir"] or "(unknown)"
        rec["B_dir"] = mb["dir"] or "(unknown)"
        rec["A_loc"] = _find_dir(ma["dir"]) if ma["dir"] else None
        rec["B_loc"] = _find_dir(mb["dir"]) if mb["dir"] else None

        if t["test"] == "probe":
            rec.update(status="NOT TESTED", reason=(
                "LR-768 noise-probe per-sample predictions are not saved as "
                "artefacts; requires re-running the probe analysis."))
            results.append(rec)
            continue

        if not ma["dir"]:
            rec.update(status="NOT TESTED", reason=(
                f"checkpoint for model {t['A']} "
                f"({ma['name']}) not available anywhere."))
            results.append(rec)
            continue
        if not mb["dir"]:
            rec.update(status="NOT TESTED", reason=(
                f"checkpoint for model {t['B']} "
                f"({mb['name']}) not available anywhere."))
            results.append(rec)
            continue

        tinfo = TASKS[t["task"]]
        try:
            if tinfo["kind"] == "cls":
                corr_a, corr_b = {}, {}
                missing = []
                for d in tinfo["dirs"]:                     # folds (or single)
                    fa = _cls_files(ma["dir"], d, t["cond"])
                    fb = _cls_files(mb["dir"], d, t["cond"])
                    if fa is None:
                        missing.append((t["A"], d))
                        continue
                    if fb is None:
                        missing.append((t["B"], d))
                        continue
                    ca, _, _ = correctness(*fa)
                    cb, _, _ = correctness(*fb)
                    pref = d + "/"
                    corr_a.update({pref + k: v for k, v in ca.items()})
                    corr_b.update({pref + k: v for k, v in cb.items()})
                if not corr_a or not corr_b:
                    who = ", ".join(f"{m}:{d}" for m, d in missing) or "all"
                    rec.update(status="NOT TESTED", reason=(
                        f"prediction files missing ({t['cond']}): {who}"))
                    results.append(rec)
                    continue
                r = mcnemar_test(corr_a, corr_b)
                rec.update(status="OK", **r)
            else:                                            # ASR
                fa = _asr_files(ma["dir"], tinfo["dirs"][0], t["cond"])
                fb = _asr_files(mb["dir"], tinfo["dirs"][0], t["cond"])
                if fa is None or fb is None:
                    miss = t["A"] if fa is None else t["B"]
                    rec.update(status="NOT TESTED", reason=(
                        f".ark files missing for {miss} ({t['cond']})"))
                    results.append(rec)
                    continue
                pa = asr_per_utt(*fa)
                pb = asr_per_utt(*fb)
                r = bootstrap_asr(pa, pb)
                rec.update(status="OK", **r)
        except Exception as e:                               # pragma: no cover
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
    out_md = os.path.join(PROJECT_ROOT, "reports", "stat_tests", "RESULTS.md")
    out_tsv = os.path.join(PROJECT_ROOT, "reports", "stat_tests", "RESULTS.tsv")
    ok = [r for r in results if r.get("status") == "OK"]
    nt = [r for r in results if r.get("status") != "OK"]

    lines = []
    lines.append("# Chapter 3 — Statistical Test Results\n")
    lines.append(f"- Tests defined: **{len(results)}**  |  "
                 f"computed: **{len(ok)}**  |  "
                 f"not tested: **{len(nt)}**")
    lines.append(f"- Noisy condition = `{NOISE_COND}`  |  "
                 f"ASR bootstrap = {BOOTSTRAP_ITERS} iters  |  "
                 f"BH FDR q = {BH_Q} (within each priority group)")
    lines.append(f"- scipy available: {_HAVE_SCIPY}\n")

    lines.append("## 1. Computed tests\n")
    lines.append("Δ = A−B (Acc% for classification, WER% for ASR). "
                 "95% CI is on Δ. Effect size: Cohen's h (classification "
                 "marginal accuracies; |h|≈0.2 small, 0.5 medium, 0.8 large) "
                 "with paired discordant odds ratio b/c; for ASR the bootstrap "
                 "CI on the WER difference is the effect size. "
                 "† = significant at BH-FDR q={:.2f}.\n".format(BH_Q))
    lines.append("| ID | Prio | Comparison | Task | Cond | Test | "
                 "Metric A | Metric B | Δ (A−B) | 95% CI Δ | Effect size | "
                 "n | raw p | BH p | Sig | Model A dir | Model B dir |")
    lines.append("|----|------|-----------|------|------|------|"
                 "---------|---------|--------|----------|-------------|"
                 "---|-------|------|-----|-------------|-------------|")
    for r in sorted(ok, key=lambda x: (x["prio"], x["id"], x["cond"])):
        unit = "WER%" if r["test"] == "bootstrap" else "Acc%"
        sig = "**YES †**" if r.get("sig") else "no"
        ci = (f"[{_fmt(r.get('ci_lo'))}, {_fmt(r.get('ci_hi'))}]"
              if r.get("ci_lo") is not None else "")
        if r["test"] == "bootstrap":
            eff = f"ΔWER {_fmt(r['delta'])} pp"
        else:
            eff = (f"h={_fmt(r.get('cohen_h'),3)}, "
                   f"OR={_fmt(r.get('odds_ratio'),2)}")
        lines.append(
            f"| {r['id']} | {r['prio']} | {r['label']} | {r['task']} | "
            f"{r['cond']} | {r['kind']} | "
            f"{_fmt(r['metric_a'])} ({unit}) | {_fmt(r['metric_b'])} ({unit}) | "
            f"{_fmt(r['delta'])} | {ci} | {eff} | "
            f"{r['n']} | {_fmt(r['p'],4)} | "
            f"{_fmt(r.get('bh_p'),4)} | {sig} | "
            f"`{r['A_dir']}` | `{r['B_dir']}` |")

    lines.append("\n### Per-group Benjamini-Hochberg summary\n")
    by_g = defaultdict(list)
    for r in ok:
        by_g[r["prio"]].append(r)
    for g in sorted(by_g):
        nsig = sum(1 for r in by_g[g] if r.get("sig"))
        lines.append(f"- **{g}**: {nsig}/{len(by_g[g])} significant "
                     f"after BH (q={BH_Q})")

    lines.append("\n## 2. NOT TESTED  (missing checkpoints / artefacts)\n")
    lines.append("| ID | Prio | Comparison | Task | Cond | Reason | "
                 "Model A | Model B |")
    lines.append("|----|------|-----------|------|------|--------|"
                 "---------|---------|")
    for r in sorted(nt, key=lambda x: (x["prio"], x["id"], x.get("cond", ""))):
        lines.append(
            f"| {r['id']} | {r['prio']} | {r['label']} | {r['task']} | "
            f"{r.get('cond','')} | {r.get('reason','')} | "
            f"{MODELS[r['A']]['name']} (`{r['A_dir']}`) | "
            f"{MODELS[r['B']]['name']} (`{r['B_dir']}`) |")

    lines.append("\n## 3. Model checkpoint locations\n")
    lines.append("| Code | Role | Directory | Resolved location |")
    lines.append("|------|------|-----------|-------------------|")
    for code, m in MODELS.items():
        loc = _find_dir(m["dir"]) if m["dir"] else None
        lines.append(f"| {code} | {m['name']} | "
                     f"`{m['dir'] or '(none / unknown)'}` | "
                     f"{loc or '**NOT FOUND** — needs upload from NTU'} |")

    # ---- 4. Interpretation (calibrated confidence + threats to validity) --
    lines.append("\n## 4. Interpretation\n")
    if not ok:
        lines.append("_No tests computed yet — populate the prediction tree "
                     "and re-run._\n")
    else:
        for g in sorted(by_g):
            lines.append(f"\n**{g} — {by_g[g][0]['label']}**\n")
            for r in sorted(by_g[g], key=lambda x: (x["id"], x["cond"])):
                if r["test"] == "bootstrap":
                    crosses0 = (r.get("ci_lo", 0) <= 0 <= r.get("ci_hi", 0))
                    small = abs(r["delta"]) < 0.3        # domain heuristic
                    if r.get("sig") and not crosses0 and not small:
                        conf = ("**high confidence** of a real WER difference "
                                "(BH-significant, 95% CI excludes 0)")
                    elif small:
                        conf = ("**speculative**: |ΔWER| < 0.3 pp is below the "
                                "level reliably distinguishable with one seed")
                    else:
                        conf = ("**moderate/none**: 95% CI on ΔWER includes 0 "
                                "— no reliable difference")
                    lines.append(
                        f"- {r['id']} {r['task']}/{r['cond']}: "
                        f"WER {_fmt(r['metric_a'])} vs {_fmt(r['metric_b'])} "
                        f"(Δ {_fmt(r['delta'])} pp, 95% CI "
                        f"[{_fmt(r.get('ci_lo'))}, {_fmt(r.get('ci_hi'))}]) — "
                        f"{conf}.")
                else:
                    h = abs(r.get("cohen_h") or 0.0)
                    mag = ("negligible" if h < 0.2 else "small" if h < 0.5
                           else "medium" if h < 0.8 else "large")
                    if r.get("sig"):
                        conf = (f"**statistically reliable** (BH-significant) "
                                f"but {mag} in magnitude: Δ={_fmt(r['delta'])}"
                                f" pp accuracy, |h|={h:.3f}, "
                                f"OR={_fmt(r.get('odds_ratio'),2)} — the "
                                f"paired disagreement is consistent in "
                                f"direction; report the effect size, not just "
                                f"the p-value")
                    elif h < 0.2:
                        conf = (f"**speculative→null**: not significant and "
                                f"effect is {mag} (|h|={h:.3f}); treat as "
                                f"'no measurable difference', not 'worse'")
                    else:
                        conf = (f"**moderate**: not BH-significant though "
                                f"effect is {mag}; report as a trend")
                    lines.append(
                        f"- {r['id']} {r['task']}/{r['cond']}: "
                        f"acc {_fmt(r['metric_a'])}% vs {_fmt(r['metric_b'])}% "
                        f"(Δ {_fmt(r['delta'])} pp, 95% CI "
                        f"[{_fmt(r.get('ci_lo'))}, {_fmt(r.get('ci_hi'))}]) — "
                        f"{conf}.")
    lines.append(
        "\n**Threats to validity.** (i) Single pre-training seed: between-seed "
        "variance is unobserved, so a non-significant per-sample result does "
        "not prove equivalence at the training-procedure level — it bounds the "
        "per-sample evidence only. (ii) ER and ESC-50 pool the 5 CV folds; the "
        "pooled McNemar treats folds as one test set and ignores fold-level "
        "correlation (mildly anti-conservative). (iii) The 'noisy' condition is "
        f"a single noise type (`{NOISE_COND}`); robustness claims generalise "
        "only to that perturbation. (iv) McNemar conditions on the discordant "
        "pairs; with few discordant samples the exact binomial is used and "
        "power is low by construction.\n")

    lines.append("\n## 5. Methodology (for the thesis Experimental Setup)\n")
    lines.append(
        "All Chapter-3 results come from single pre-training seeds, so tests "
        "operate on per-sample predictions from a single checkpoint. "
        "Classification tasks (IC, KS, ER, ESC-50, VocID) use **McNemar's "
        "test** on the paired per-sample correct/incorrect outcomes "
        "(continuity-corrected χ² for ≥25 discordant pairs, exact binomial "
        "otherwise), reported with **Cohen's h** and the paired discordant "
        "odds ratio as effect sizes plus a 95% CI on the paired accuracy "
        "difference. ER and ESC-50 pool the per-sample outcomes across all 5 "
        "cross-validation folds. "
        "**ASR** uses a **paired bootstrap** "
        f"({BOOTSTRAP_ITERS} resamples, seed {BOOTSTRAP_SEED}) over utterances "
        "on the corpus-level WER (Koehn, 2004), reporting a 95% CI on the WER "
        "difference. The MAPSSWE test (NIST SCTK `sc_stats`) — the other "
        "option listed in the testing plan — was deliberately **not** used: "
        "it requires the SCTK toolchain and segment-aligned CTM input, whereas "
        "only `.ark` hypothesis/reference text is available; and the paired "
        "bootstrap is non-parametric and, by resampling whole utterances, "
        "correctly preserves the length-weighting of the corpus-level WER "
        "ratio (the exact bias that a per-utterance t-test would introduce). "
        "Both are sanctioned by the plan (§2.1); the bootstrap is the more "
        "robust and assumption-light choice given the available artefacts. "
        f"Multiple testing is controlled with Benjamini-Hochberg FDR "
        f"(q={BH_Q}) applied within each priority group (the plan's "
        "recommended family-wise scheme). † denotes significance after BH "
        "correction. Bootstrap RNG is seeded for reproducibility; "
        f"libraries: numpy {np.__version__}, "
        f"scipy {'present' if _HAVE_SCIPY else 'absent (pure-python fallback)'}"
        ".\n")

    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    cols = ["id", "prio", "label", "task", "cond", "test", "status",
            "metric_a", "metric_b", "delta", "ci_lo", "ci_hi",
            "cohen_h", "odds_ratio", "n10", "n01", "disc",
            "n", "stat", "p", "bh_p", "sig",
            "kind", "A_dir", "B_dir", "reason"]
    with open(out_tsv, "w", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in results:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    print("\n".join(lines))
    print(f"\n[written] {out_md}")
    print(f"[written] {out_tsv}")


if __name__ == "__main__":
    run()
