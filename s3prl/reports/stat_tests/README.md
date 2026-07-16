# Distillation-robustness statistical tests

`run_stat_tests.py` implements the 31 significance tests for the
distillation-robustness comparisons (L_KD vs L_CL objectives, clean and noisy
evaluation). See `docs/statistical-testing.md` at the repo root for the
methodology and input format.

## Run

```bash
# from the s3prl/ directory; expects per-sample prediction dumps under
# reports/stat_tests/predictions/ (or set PROJECT_ROOT / M*_DIR overrides)
python reports/stat_tests/run_stat_tests.py
```

`run_stat_tests.py` is **safe to run any time** — every test whose files are
missing is listed under "NOT TESTED" with the reason and expected location, so
it doubles as a live gap report.

## Tests → models

| Code | Role | Directory | Where |
|------|------|-----------|-------|
| M1 | L_KD Setup2 HuBERT (fixed) | `setup2_2-dis_t_nocont_teacher_100hourslibri_l1-2026` | music tasks local on NSCC; speech/ESC-50 on **NTU** |
| M2 | L_CL full Setup2 HuBERT (fixed) | `setup2_2-dis_loss_barlow_..._5e-6-960hrs-correct` | music tasks local; speech/ESC-50/ER on **NTU** |
| M3 | CC-only Setup2 HuBERT (fixed) | `barlow_old_setup2_2-dis_nocont_teacher_100hourslibri-2026-noselfcorr` | ASR/KS/music local; IC/ER on **NTU** |
| M4 | L_CL heuristic weights | — | **MISSING everywhere** (the SNR/heuristic model) |
| M5 | L_KD Setup2 HuBERT+ (fixed) | unknown | **not located** — set `M5_DIR` once found on NTU |
| M6 | L_CL Setup2 HuBERT+ (fixed) | unknown | **not located** — set `M6_DIR` once found on NTU |
| M7 | L_CL same-noise Setup2 HuBERT | `barlow_old_setup2_2-dis_nocont_teacher_100hours-same-noise-teacher-and-student` | pretrain dir local; downstream on **NTU** |

| Prio | Tests | Comparison | Models | Feasible? |
|------|-------|-----------|--------|-----------|
| P1 | 8 | L_KD vs L_CL (IC,KS,ER,ASR × clean,noisy) | M1 vs M2 | ✅ after NTU upload |
| P2 | 4 | L_KD vs L_CL HuBERT+ (IC,ER × clean,noisy) | M5 vs M6 | ⚠️ needs M5/M6 located |
| P3 | 4 | CC-only vs full L_CL (IC,ER,ASR,KS noisy) | M3 vs M2 | ✅ after NTU upload |
| P4 | 6 | Fixed vs Heuristic (IC,KS,ASR × clean,noisy) | M2 vs M4 | ❌ M4 missing |
| P5 | 4 | L_KD vs L_CL (ESC-50,VocID × clean,noisy) | M1 vs M2 | ✅ after NTU upload |
| P6 | 5 | probes (3) + same-noise IC (2) | M1/M3/M7 vs M2 | ⚠️ probes need re-run; IC needs M7 |

## Outcome (2026-05-20, after NTU IC+KS evals)

**15 / 31 computed** (up from 7 on 2026-05-19). 13 significant, 2 true nulls.

| Test | Comparison | Δ (A−B) | BH p | Verdict |
|------|-----------|---------|------|---------|
| P1-1 IC clean | L_KD vs L_CL | −2.53 pp | <1e-4 | L_CL better ✓ |
| P1-1 IC noisy | L_KD vs L_CL | −3.95 pp | <1e-4 | L_CL better ✓ |
| P1-2 KS clean | L_KD vs L_CL | −0.65 pp | 0.016 | L_CL better ✓ |
| P1-2 KS noisy | L_KD vs L_CL | −1.30 pp | <1e-4 | L_CL better ✓ |
| P1-3 ER clean | L_KD vs L_CL | −2.97 pp | <1e-4 | L_CL better ✓ |
| P1-3 ER noisy | L_KD vs L_CL | −3.83 pp | <1e-4 | L_CL better ✓ |
| P3-1 IC noisy | CC-only vs full L_CL | −2.79 pp | <1e-4 | self-corr helps ✓ |
| P3-2 ER noisy | CC-only vs full L_CL | −2.44 pp | <1e-4 | self-corr helps ✓ |
| P3-4 KS noisy | CC-only vs full L_CL | −0.45 pp | 0.17 | no difference |
| P5-1 ESC-50 clean | L_KD vs L_CL | −3.00 pp | 0.007 | L_CL better ✓ |
| P5-1 ESC-50 noisy | L_KD vs L_CL | −4.95 pp | <1e-4 | L_CL better ✓ |
| P5-2 VocID clean | L_KD vs L_CL | −3.11 pp | 0.027 | L_CL better ✓ |
| P5-2 VocID noisy | L_KD vs L_CL | +0.35 pp | 0.83 | no difference |
| P6-4 IC clean | same-noise vs indep | −0.82 pp | 0.033 | indep better ✓ |
| P6-4 IC noisy | same-noise vs indep | **−13.87 pp** | <1e-4 | same-noise collapses ✓ |

### Per-group summary

- **P1 (L_KD vs L_CL)**: 6/6 significant — L_CL consistently better across IC, KS, ER
- **P3 (CC-only vs full)**: 2/3 significant — self-corr helps for IC and ER, not KS
- **P5 (L_KD vs L_CL, audio/music)**: 3/4 significant — VocID noisy is a true null
- **P6 (same-noise vs indep)**: 2/2 significant — **standout**: noisy IC Δ=−13.87 pp (|h|=0.42, medium effect)

## Tests still pending (3 — need ASR from NTU)

- **P1-4 ASR clean/noisy** (M1 vs M2) — ASR downstream training ~16h on NTU
- **P3-3 ASR noisy** (M3 vs M2) — needs M2 ASR noisy `.ark` files

## Tests structurally impossible (13)

- **P4 (6 tests)** — heuristic/SNR checkpoint (M4) does not exist anywhere
- **P2 (4 tests)** — HuBERT+ models (M5/M6) — checkpoints could not be located on any cluster or backup
- **P6-1/2/3 (3 tests)** — LR-768 noise-probe per-sample predictions never saved

Override unknown dirs without editing code:
`M5_DIR=... M6_DIR=... M7_DIR=... python reports/stat_tests/run_stat_tests.py`

## Methods

- Classification → McNemar (continuity-corrected χ², exact binomial when
  discordant < 25). ER / ESC-50 pool per-sample outcomes across the 5 folds.
- ASR → paired bootstrap (10 000) over utterances on corpus WER, 95 % CI.
- Multiple testing → Benjamini-Hochberg FDR q=0.05 within each priority group.
