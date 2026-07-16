# Chapter 3 — Statistical Test Results

- Tests defined: **31**  |  computed: **15**  |  not tested: **16**
- Noisy condition = `chime`  |  ASR bootstrap = 10000 iters  |  BH FDR q = 0.05 (within each priority group)
- scipy available: True

## 1. Computed tests

Δ = A−B (Acc% for classification, WER% for ASR). 95% CI is on Δ. Effect size: Cohen's h (classification marginal accuracies; |h|≈0.2 small, 0.5 medium, 0.8 large) with paired discordant odds ratio b/c; for ASR the bootstrap CI on the WER difference is the effect size. † = significant at BH-FDR q=0.05.

| ID | Prio | Comparison | Task | Cond | Test | Metric A | Metric B | Δ (A−B) | 95% CI Δ | Effect size | n | raw p | BH p | Sig | Model A dir | Model B dir |
|----|------|-----------|------|------|------|---------|---------|--------|----------|-------------|---|-------|------|-----|-------------|-------------|
| P1-1 | P1 | L_KD vs L_CL | IC | clean | chi2_cc | 93.70 (Acc%) | 96.23 (Acc%) | -2.53 | [-3.32, -1.74] | h=-0.117, OR=0.42 | 3793 | 0.0000 | 0.0000 | **YES †** | `setup2_2-dis_t_nocont_teacher_100hourslibri_l1-2026` | `setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct` |
| P1-1 | P1 | L_KD vs L_CL | IC | noisy | chi2_cc | 89.67 (Acc%) | 93.62 (Acc%) | -3.95 | [-5.00, -2.91] | h=-0.144, OR=0.47 | 3793 | 0.0000 | 0.0000 | **YES †** | `setup2_2-dis_t_nocont_teacher_100hourslibri_l1-2026` | `setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct` |
| P1-2 | P1 | L_KD vs L_CL | KS | clean | chi2_cc | 95.49 (Acc%) | 96.14 (Acc%) | -0.65 | [-1.15, -0.15] | h=-0.032, OR=0.52 | 3081 | 0.0158 | 0.0158 | **YES †** | `setup2_2-dis_t_nocont_teacher_100hourslibri_l1-2026` | `setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct` |
| P1-2 | P1 | L_KD vs L_CL | KS | noisy | chi2_cc | 93.77 (Acc%) | 95.07 (Acc%) | -1.30 | [-1.93, -0.66] | h=-0.057, OR=0.43 | 3081 | 0.0001 | 0.0001 | **YES †** | `setup2_2-dis_t_nocont_teacher_100hourslibri_l1-2026` | `setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct` |
| P1-3 | P1 | L_KD vs L_CL | ER | clean | chi2_cc | 60.68 (Acc%) | 63.64 (Acc%) | -2.97 | [-4.12, -1.81] | h=-0.061, OR=0.73 | 5531 | 0.0000 | 0.0000 | **YES †** | `setup2_2-dis_t_nocont_teacher_100hourslibri_l1-2026` | `setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct` |
| P1-3 | P1 | L_KD vs L_CL | ER | noisy | chi2_cc | 58.90 (Acc%) | 62.74 (Acc%) | -3.83 | [-5.04, -2.62] | h=-0.079, OR=0.69 | 5531 | 0.0000 | 0.0000 | **YES †** | `setup2_2-dis_t_nocont_teacher_100hourslibri_l1-2026` | `setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct` |
| P3-1 | P3 | CC-only vs full L_CL | IC | noisy | chi2_cc | 90.83 (Acc%) | 93.62 (Acc%) | -2.79 | [-3.71, -1.87] | h=-0.105, OR=0.50 | 3793 | 0.0000 | 0.0000 | **YES †** | `barlow_old_setup2_2-dis_nocont_teacher_100hourslibri-2026-noselfcorr` | `setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct` |
| P3-2 | P3 | CC-only vs full L_CL | ER | noisy | chi2_cc | 60.30 (Acc%) | 62.74 (Acc%) | -2.44 | [-3.50, -1.38] | h=-0.050, OR=0.74 | 5531 | 0.0000 | 0.0000 | **YES †** | `barlow_old_setup2_2-dis_nocont_teacher_100hourslibri-2026-noselfcorr` | `setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct` |
| P3-4 | P3 | CC-only vs full L_CL | KS | noisy | chi2_cc | 94.61 (Acc%) | 95.07 (Acc%) | -0.45 | [-1.05, 0.14] | h=-0.021, OR=0.73 | 3081 | 0.1658 | 0.1658 | no | `barlow_old_setup2_2-dis_nocont_teacher_100hourslibri-2026-noselfcorr` | `setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct` |
| P5-1 | P5 | L_KD vs L_CL | ESC-50 | clean | chi2_cc | 67.30 (Acc%) | 70.30 (Acc%) | -3.00 | [-4.98, -1.02] | h=-0.065, OR=0.75 | 2000 | 0.0036 | 0.0071 | **YES †** | `setup2_2-dis_t_nocont_teacher_100hourslibri_l1-2026` | `setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct` |
| P5-1 | P5 | L_KD vs L_CL | ESC-50 | noisy | chi2_cc | 59.40 (Acc%) | 64.35 (Acc%) | -4.95 | [-7.02, -2.88] | h=-0.102, OR=0.64 | 2000 | 0.0000 | 0.0000 | **YES †** | `setup2_2-dis_t_nocont_teacher_100hourslibri_l1-2026` | `setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct` |
| P5-2 | P5 | L_KD vs L_CL | VocID | clean | chi2_cc | 59.79 (Acc%) | 62.90 (Acc%) | -3.11 | [-5.67, -0.55] | h=-0.064, OR=0.77 | 1415 | 0.0201 | 0.0268 | **YES †** | `setup2_2-dis_t_nocont_teacher_100hourslibri_l1-2026` | `setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct` |
| P5-2 | P5 | L_KD vs L_CL | VocID | noisy | chi2_cc | 56.68 (Acc%) | 56.33 (Acc%) | 0.35 | [-2.26, 2.97] | h=0.007, OR=1.03 | 1415 | 0.8323 | 0.8323 | no | `setup2_2-dis_t_nocont_teacher_100hourslibri_l1-2026` | `setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct` |
| P6-4 | P6 | same-noise vs indep (IC downstream) | IC | clean | chi2_cc | 95.41 (Acc%) | 96.23 (Acc%) | -0.82 | [-1.54, -0.09] | h=-0.041, OR=0.73 | 3793 | 0.0326 | 0.0326 | **YES †** | `barlow_old_setup2_2-dis_nocont_teacher_100hours-same-noise-teacher-and-student` | `setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct` |
| P6-4 | P6 | same-noise vs indep (IC downstream) | IC | noisy | chi2_cc | 79.75 (Acc%) | 93.62 (Acc%) | -13.87 | [-15.26, -12.47] | h=-0.423, OR=0.21 | 3793 | 0.0000 | 0.0000 | **YES †** | `barlow_old_setup2_2-dis_nocont_teacher_100hours-same-noise-teacher-and-student` | `setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct` |

### Per-group Benjamini-Hochberg summary

- **P1**: 6/6 significant after BH (q=0.05)
- **P3**: 2/3 significant after BH (q=0.05)
- **P5**: 3/4 significant after BH (q=0.05)
- **P6**: 2/2 significant after BH (q=0.05)

## 2. NOT TESTED  (missing checkpoints / artefacts)

| ID | Prio | Comparison | Task | Cond | Reason | Model A | Model B |
|----|------|-----------|------|------|--------|---------|---------|
| P1-4 | P1 | L_KD vs L_CL | ASR | clean | .ark files missing for M1 (clean) | DistilHuBERT L_KD (Setup2, HuBERT, fixed) (`setup2_2-dis_t_nocont_teacher_100hourslibri_l1-2026`) | DistilHuBERT L_CL full (Setup2, HuBERT, fixed) (`setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct`) |
| P1-4 | P1 | L_KD vs L_CL | ASR | noisy | .ark files missing for M1 (noisy) | DistilHuBERT L_KD (Setup2, HuBERT, fixed) (`setup2_2-dis_t_nocont_teacher_100hourslibri_l1-2026`) | DistilHuBERT L_CL full (Setup2, HuBERT, fixed) (`setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct`) |
| P2-1 | P2 | L_KD vs L_CL (HuBERT+) | IC | clean | checkpoint for model M5 (DistilHuBERT L_KD (Setup2, HuBERT+, fixed)) not available anywhere. | DistilHuBERT L_KD (Setup2, HuBERT+, fixed) (`(unknown)`) | DistilHuBERT L_CL (Setup2, HuBERT+, fixed) (`(unknown)`) |
| P2-1 | P2 | L_KD vs L_CL (HuBERT+) | IC | noisy | checkpoint for model M5 (DistilHuBERT L_KD (Setup2, HuBERT+, fixed)) not available anywhere. | DistilHuBERT L_KD (Setup2, HuBERT+, fixed) (`(unknown)`) | DistilHuBERT L_CL (Setup2, HuBERT+, fixed) (`(unknown)`) |
| P2-2 | P2 | L_KD vs L_CL (HuBERT+) | ER | clean | checkpoint for model M5 (DistilHuBERT L_KD (Setup2, HuBERT+, fixed)) not available anywhere. | DistilHuBERT L_KD (Setup2, HuBERT+, fixed) (`(unknown)`) | DistilHuBERT L_CL (Setup2, HuBERT+, fixed) (`(unknown)`) |
| P2-2 | P2 | L_KD vs L_CL (HuBERT+) | ER | noisy | checkpoint for model M5 (DistilHuBERT L_KD (Setup2, HuBERT+, fixed)) not available anywhere. | DistilHuBERT L_KD (Setup2, HuBERT+, fixed) (`(unknown)`) | DistilHuBERT L_CL (Setup2, HuBERT+, fixed) (`(unknown)`) |
| P3-3 | P3 | CC-only vs full L_CL | ASR | noisy | .ark files missing for M2 (noisy) | DistilHuBERT CC-only (Setup2, HuBERT, fixed) (`barlow_old_setup2_2-dis_nocont_teacher_100hourslibri-2026-noselfcorr`) | DistilHuBERT L_CL full (Setup2, HuBERT, fixed) (`setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct`) |
| P4-1 | P4 | Fixed vs Heuristic | IC | clean | checkpoint for model M4 (DistilHuBERT L_CL heuristic weights (Setup2, HuBERT)) not available anywhere. | DistilHuBERT L_CL full (Setup2, HuBERT, fixed) (`setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct`) | DistilHuBERT L_CL heuristic weights (Setup2, HuBERT) (`(unknown)`) |
| P4-1 | P4 | Fixed vs Heuristic | IC | noisy | checkpoint for model M4 (DistilHuBERT L_CL heuristic weights (Setup2, HuBERT)) not available anywhere. | DistilHuBERT L_CL full (Setup2, HuBERT, fixed) (`setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct`) | DistilHuBERT L_CL heuristic weights (Setup2, HuBERT) (`(unknown)`) |
| P4-2 | P4 | Fixed vs Heuristic | KS | clean | checkpoint for model M4 (DistilHuBERT L_CL heuristic weights (Setup2, HuBERT)) not available anywhere. | DistilHuBERT L_CL full (Setup2, HuBERT, fixed) (`setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct`) | DistilHuBERT L_CL heuristic weights (Setup2, HuBERT) (`(unknown)`) |
| P4-2 | P4 | Fixed vs Heuristic | KS | noisy | checkpoint for model M4 (DistilHuBERT L_CL heuristic weights (Setup2, HuBERT)) not available anywhere. | DistilHuBERT L_CL full (Setup2, HuBERT, fixed) (`setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct`) | DistilHuBERT L_CL heuristic weights (Setup2, HuBERT) (`(unknown)`) |
| P4-3 | P4 | Fixed vs Heuristic | ASR | clean | checkpoint for model M4 (DistilHuBERT L_CL heuristic weights (Setup2, HuBERT)) not available anywhere. | DistilHuBERT L_CL full (Setup2, HuBERT, fixed) (`setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct`) | DistilHuBERT L_CL heuristic weights (Setup2, HuBERT) (`(unknown)`) |
| P4-3 | P4 | Fixed vs Heuristic | ASR | noisy | checkpoint for model M4 (DistilHuBERT L_CL heuristic weights (Setup2, HuBERT)) not available anywhere. | DistilHuBERT L_CL full (Setup2, HuBERT, fixed) (`setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct`) | DistilHuBERT L_CL heuristic weights (Setup2, HuBERT) (`(unknown)`) |
| P6-1 | P6 | L_KD vs L_CL (LR-768 probe) | PROBE | probe | LR-768 noise-probe per-sample predictions are not saved as artefacts; requires re-running the probe analysis. | DistilHuBERT L_KD (Setup2, HuBERT, fixed) (`setup2_2-dis_t_nocont_teacher_100hourslibri_l1-2026`) | DistilHuBERT L_CL full (Setup2, HuBERT, fixed) (`setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct`) |
| P6-2 | P6 | CC-only vs full (LR-768 probe) | PROBE | probe | LR-768 noise-probe per-sample predictions are not saved as artefacts; requires re-running the probe analysis. | DistilHuBERT CC-only (Setup2, HuBERT, fixed) (`barlow_old_setup2_2-dis_nocont_teacher_100hourslibri-2026-noselfcorr`) | DistilHuBERT L_CL full (Setup2, HuBERT, fixed) (`setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct`) |
| P6-3 | P6 | same-noise vs indep (LR-768 probe) | PROBE | probe | LR-768 noise-probe per-sample predictions are not saved as artefacts; requires re-running the probe analysis. | DistilHuBERT L_CL same-noise (Setup2, HuBERT) (`barlow_old_setup2_2-dis_nocont_teacher_100hours-same-noise-teacher-and-student`) | DistilHuBERT L_CL full (Setup2, HuBERT, fixed) (`setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct`) |

## 3. Model checkpoint locations

| Code | Role | Directory | Resolved location |
|------|------|-----------|-------------------|
| M1 | DistilHuBERT L_KD (Setup2, HuBERT, fixed) | `setup2_2-dis_t_nocont_teacher_100hourslibri_l1-2026` | s3prl/reports/stat_tests/predictions/nscc_local/setup2_2-dis_t_nocont_teacher_100hourslibri_l1-2026 |
| M2 | DistilHuBERT L_CL full (Setup2, HuBERT, fixed) | `setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct` | s3prl/reports/stat_tests/predictions/nscc_local/setup2_2-dis_loss_barlow_frame_avg_pooling_150_batch_size_24_with_cosine_with_cross_corr_5e-5_self_corr_5e-6-960hrs-correct |
| M3 | DistilHuBERT CC-only (Setup2, HuBERT, fixed) | `barlow_old_setup2_2-dis_nocont_teacher_100hourslibri-2026-noselfcorr` | s3prl/reports/stat_tests/predictions/nscc_local/barlow_old_setup2_2-dis_nocont_teacher_100hourslibri-2026-noselfcorr |
| M4 | DistilHuBERT L_CL heuristic weights (Setup2, HuBERT) | `(none / unknown)` | **NOT FOUND** — needs upload from NTU |
| M5 | DistilHuBERT L_KD (Setup2, HuBERT+, fixed) | `(none / unknown)` | **NOT FOUND** — needs upload from NTU |
| M6 | DistilHuBERT L_CL (Setup2, HuBERT+, fixed) | `(none / unknown)` | **NOT FOUND** — needs upload from NTU |
| M7 | DistilHuBERT L_CL same-noise (Setup2, HuBERT) | `barlow_old_setup2_2-dis_nocont_teacher_100hours-same-noise-teacher-and-student` | s3prl/reports/stat_tests/predictions/ntu/downstream-thesis-pending-exps/barlow_old_setup2_2-dis_nocont_teacher_100hours-same-noise-teacher-and-student |

## 4. Interpretation


**P1 — L_KD vs L_CL**

- P1-1 IC/clean: acc 93.70% vs 96.23% (Δ -2.53 pp, 95% CI [-3.32, -1.74]) — **statistically reliable** (BH-significant) but negligible in magnitude: Δ=-2.53 pp accuracy, |h|=0.117, OR=0.42 — the paired disagreement is consistent in direction; report the effect size, not just the p-value.
- P1-1 IC/noisy: acc 89.67% vs 93.62% (Δ -3.95 pp, 95% CI [-5.00, -2.91]) — **statistically reliable** (BH-significant) but negligible in magnitude: Δ=-3.95 pp accuracy, |h|=0.144, OR=0.47 — the paired disagreement is consistent in direction; report the effect size, not just the p-value.
- P1-2 KS/clean: acc 95.49% vs 96.14% (Δ -0.65 pp, 95% CI [-1.15, -0.15]) — **statistically reliable** (BH-significant) but negligible in magnitude: Δ=-0.65 pp accuracy, |h|=0.032, OR=0.52 — the paired disagreement is consistent in direction; report the effect size, not just the p-value.
- P1-2 KS/noisy: acc 93.77% vs 95.07% (Δ -1.30 pp, 95% CI [-1.93, -0.66]) — **statistically reliable** (BH-significant) but negligible in magnitude: Δ=-1.30 pp accuracy, |h|=0.057, OR=0.43 — the paired disagreement is consistent in direction; report the effect size, not just the p-value.
- P1-3 ER/clean: acc 60.68% vs 63.64% (Δ -2.97 pp, 95% CI [-4.12, -1.81]) — **statistically reliable** (BH-significant) but negligible in magnitude: Δ=-2.97 pp accuracy, |h|=0.061, OR=0.73 — the paired disagreement is consistent in direction; report the effect size, not just the p-value.
- P1-3 ER/noisy: acc 58.90% vs 62.74% (Δ -3.83 pp, 95% CI [-5.04, -2.62]) — **statistically reliable** (BH-significant) but negligible in magnitude: Δ=-3.83 pp accuracy, |h|=0.079, OR=0.69 — the paired disagreement is consistent in direction; report the effect size, not just the p-value.

**P3 — CC-only vs full L_CL**

- P3-1 IC/noisy: acc 90.83% vs 93.62% (Δ -2.79 pp, 95% CI [-3.71, -1.87]) — **statistically reliable** (BH-significant) but negligible in magnitude: Δ=-2.79 pp accuracy, |h|=0.105, OR=0.50 — the paired disagreement is consistent in direction; report the effect size, not just the p-value.
- P3-2 ER/noisy: acc 60.30% vs 62.74% (Δ -2.44 pp, 95% CI [-3.50, -1.38]) — **statistically reliable** (BH-significant) but negligible in magnitude: Δ=-2.44 pp accuracy, |h|=0.050, OR=0.74 — the paired disagreement is consistent in direction; report the effect size, not just the p-value.
- P3-4 KS/noisy: acc 94.61% vs 95.07% (Δ -0.45 pp, 95% CI [-1.05, 0.14]) — **speculative→null**: not significant and effect is negligible (|h|=0.021); treat as 'no measurable difference', not 'worse'.

**P5 — L_KD vs L_CL**

- P5-1 ESC-50/clean: acc 67.30% vs 70.30% (Δ -3.00 pp, 95% CI [-4.98, -1.02]) — **statistically reliable** (BH-significant) but negligible in magnitude: Δ=-3.00 pp accuracy, |h|=0.065, OR=0.75 — the paired disagreement is consistent in direction; report the effect size, not just the p-value.
- P5-1 ESC-50/noisy: acc 59.40% vs 64.35% (Δ -4.95 pp, 95% CI [-7.02, -2.88]) — **statistically reliable** (BH-significant) but negligible in magnitude: Δ=-4.95 pp accuracy, |h|=0.102, OR=0.64 — the paired disagreement is consistent in direction; report the effect size, not just the p-value.
- P5-2 VocID/clean: acc 59.79% vs 62.90% (Δ -3.11 pp, 95% CI [-5.67, -0.55]) — **statistically reliable** (BH-significant) but negligible in magnitude: Δ=-3.11 pp accuracy, |h|=0.064, OR=0.77 — the paired disagreement is consistent in direction; report the effect size, not just the p-value.
- P5-2 VocID/noisy: acc 56.68% vs 56.33% (Δ 0.35 pp, 95% CI [-2.26, 2.97]) — **speculative→null**: not significant and effect is negligible (|h|=0.007); treat as 'no measurable difference', not 'worse'.

**P6 — same-noise vs indep (IC downstream)**

- P6-4 IC/clean: acc 95.41% vs 96.23% (Δ -0.82 pp, 95% CI [-1.54, -0.09]) — **statistically reliable** (BH-significant) but negligible in magnitude: Δ=-0.82 pp accuracy, |h|=0.041, OR=0.73 — the paired disagreement is consistent in direction; report the effect size, not just the p-value.
- P6-4 IC/noisy: acc 79.75% vs 93.62% (Δ -13.87 pp, 95% CI [-15.26, -12.47]) — **statistically reliable** (BH-significant) but small in magnitude: Δ=-13.87 pp accuracy, |h|=0.423, OR=0.21 — the paired disagreement is consistent in direction; report the effect size, not just the p-value.

**Missing ASR tests — inference from evaluated tasks.** The three ASR comparisons (P1-4 clean/noisy, P3-3 noisy) could not be computed because per-utterance `.ark` prediction files were not produced during the original evaluation campaign, and re-running ASR training (~16 h per model) was not feasible within the thesis timeline. We note the following trends from the evaluated tasks to inform expectations:

- *P1-4 (L_KD vs L_CL, ASR clean/noisy)*: all six evaluated P1 tests (IC, KS, ER × clean/noisy) are BH-significant in favour of L_CL, with the gap widening under noise in every case. We expect the ASR comparison to follow the same direction.
- *P3-3 (CC-only vs full L_CL, ASR noisy)*: P3 shows mixed evidence — IC noisy and ER noisy are significant in favour of the full L_CL, but KS noisy is not (Δ=−0.45 pp, p=0.17). The ASR comparison therefore remains inconclusive based on the available evidence.

**Threats to validity.** (i) Single pre-training seed: between-seed variance is unobserved, so a non-significant per-sample result does not prove equivalence at the training-procedure level — it bounds the per-sample evidence only. (ii) ER and ESC-50 pool the 5 CV folds; the pooled McNemar treats folds as one test set and ignores fold-level correlation (mildly anti-conservative). (iii) The 'noisy' condition is a single noise type (`chime`); robustness claims generalise only to that perturbation. (iv) McNemar conditions on the discordant pairs; with few discordant samples the exact binomial is used and power is low by construction. (v) The ASR trend inference above is not a formal test — it is a qualitative observation based on the consistency of the evaluated tasks and should be reported as such.


## 5. Methodology (for the thesis Experimental Setup)

All Chapter-3 results come from single pre-training seeds, so tests operate on per-sample predictions from a single checkpoint. Classification tasks (IC, KS, ER, ESC-50, VocID) use **McNemar's test** on the paired per-sample correct/incorrect outcomes (continuity-corrected χ² for ≥25 discordant pairs, exact binomial otherwise), reported with **Cohen's h** and the paired discordant odds ratio as effect sizes plus a 95% CI on the paired accuracy difference. ER and ESC-50 pool the per-sample outcomes across all 5 cross-validation folds. **ASR** uses a **paired bootstrap** (10000 resamples, seed 1234) over utterances on the corpus-level WER (Koehn, 2004), reporting a 95% CI on the WER difference. The MAPSSWE test (NIST SCTK `sc_stats`) — the other option listed in the testing plan — was deliberately **not** used: it requires the SCTK toolchain and segment-aligned CTM input, whereas only `.ark` hypothesis/reference text is available; and the paired bootstrap is non-parametric and, by resampling whole utterances, correctly preserves the length-weighting of the corpus-level WER ratio (the exact bias that a per-utterance t-test would introduce). Both are sanctioned by the plan (§2.1); the bootstrap is the more robust and assumption-light choice given the available artefacts. Multiple testing is controlled with Benjamini-Hochberg FDR (q=0.05) applied within each priority group (the plan's recommended family-wise scheme). † denotes significance after BH correction. Bootstrap RNG is seeded for reproducibility; libraries: numpy 2.2.3, scipy present.

