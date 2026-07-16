# Chapter 3 — Statistical Testing Plan


## 1. Context

All Chapter 3 results come from **single pre-training seeds** with no repeated
runs.  No standard deviations or confidence intervals are currently reported in
any table.  This is standard practice for SSL pre-training (each run is
~200 k steps on LibriSpeech-100 h), but it means we cannot use tests that
require multiple independent training runs (e.g., paired t-test across seeds).

The tests below are chosen specifically for the **single-seed, single-run**
setting.  They operate on per-sample predictions from a single checkpoint.

---

## 2. Test Selection by Task Type

### 2.1 ASR (WER) — Paired Bootstrap Test

WER is a **corpus-level ratio** (total substitutions + deletions + insertions
divided by total reference words).  Longer utterances contribute more errors.
A standard per-utterance t-test would weight all utterances equally regardless
of length, biasing the result.

| Property | Detail |
|---|---|
| **Test** | Paired bootstrap resampling (Koehn 2004) or MAPSSWE (NIST SCTK `sc_stats`) |
| **Unit of resampling** | Utterance (sentence-segment) |
| **Iterations** | 10 000 bootstrap samples |
| **p-value** | Fraction of samples where the WER difference changes sign |
| **Tooling** | NIST SCTK `sclite` → `sc_stats`, or custom Python |
| **Input needed** | Per-utterance hypothesis and reference transcripts (`.trn` / `.ctm` format, or raw text + alignment) |

**Why not McNemar for ASR?**  McNemar operates on binary (correct/incorrect)
per-sample outcomes.  ASR produces per-word errors within variable-length
utterances — McNemar cannot capture this structure.

### 2.2 Classification Tasks (IC, KS, ER, ESC-50, music tasks) — McNemar's Test

All non-ASR downstream tasks report **accuracy** on a fixed test set.

| Property | Detail |
|---|---|
| **Test** | McNemar's test with continuity correction (or exact binomial for small discordant counts < 25) |
| **Unit** | Per-sample binary outcome: correct (1) or incorrect (0) |
| **Statistic** | χ² with 1 df on the 2×2 discordant table |
| **Tooling** | `statsmodels.stats.contingency_tables.mcnemar` or `scipy` manual |
| **Input needed** | Per-sample prediction vectors from both models on the same test set |

**Contingency table structure:**

```
                    Model B correct    Model B wrong
Model A correct        n₁₁                n₁₀
Model A wrong          n₀₁                n₀₀
```

McNemar tests whether n₁₀ ≠ n₀₁ (asymmetric disagreement).

### 2.3 Noise Probes

| Probe | Test | Rationale |
|---|---|---|
| PCA-32 RF (100 bootstrap seeds) | **Report 95 % confidence interval** from the 100 existing seeds; check for non-overlap between conditions | Already have distributional data |
| LR-768 (single fit) | **McNemar's test** on per-sample noise-type predictions | Same logic as classification tasks |

---

## 3. Essential Comparisons

The comparisons below are the **minimum set** needed to statistically validate
every key claim in the chapter.  They are grouped by priority.

### Priority 1 — Core contribution (L_KD vs L_CL)

These back the central claim that the correlation loss improves over the
standard distillation loss.

| ID | Comparison | Setup | Teacher | Tasks | Conditions | Test | Path Model 1 | Path Model 2 |
|----|------------|-------|---------|-------|------------|------|--------------|--------------|
| P1-1 | L_KD vs L_CL | Setup 2 | HuBERT | IC | Clean, Noisy | McNemar |  | 
| P1-2 | L_KD vs L_CL | Setup 2 | HuBERT | KS | Clean, Noisy | McNemar | |
| P1-3 | L_KD vs L_CL | Setup 2 | HuBERT | ER | Clean, Noisy | McNemar | |
| P1-4 | L_KD vs L_CL | Setup 2 | HuBERT | ASR | Clean, Noisy | Bootstrap |  |

**Total tests: 8** (4 tasks × 2 conditions).
Maps to: Table 1 Block 3.

### Priority 2 — Teacher-agnostic claim (L_KD vs L_CL with HuBERT+)

| ID | Comparison | Setup | Teacher | Tasks | Conditions | Test |  Path Model 1 | Path Model 2 |
|----|-----------|-------|---------|-------|------------|------|------|------|
| P2-1 | L_KD vs L_CL | Setup 2 | HuBERT+ | IC | Clean, Noisy | McNemar |
| P2-2 | L_KD vs L_CL | Setup 2 | HuBERT+ | ER | Clean, Noisy | McNemar |

**Total tests: 4**.
Maps to: Table 1 Block 5.

### Priority 3 — Self-correlation ablation (CC-only vs full L_CL)

Validates that the self-correlation term contributes beyond cross-correlation.

| ID | Comparison | Setup | Teacher | Tasks | Conditions | Test |
|----|-----------|-------|---------|-------|------------|------|
| P3-1 | CC-only vs full L_CL | Setup 2 | HuBERT | IC | Noisy | McNemar |
| P3-2 | CC-only vs full L_CL | Setup 2 | HuBERT | ER | Noisy | McNemar |
| P3-3 | CC-only vs full L_CL | Setup 2 | HuBERT | ASR | Noisy | Bootstrap |
| P3-4 | CC-only vs full L_CL | Setup 2 | HuBERT | KS | Noisy | McNemar |

**Total tests: 4** (noisy only — the ablation claim is about robustness).
Maps to: Table 5.

### Priority 4 — Heuristic weighting (Fixed vs Heuristic)

| ID | Comparison | Setup | Teacher | Tasks | Conditions | Test |
|----|-----------|-------|---------|-------|------------|------|
| P4-1 | Fixed vs Heuristic | Setup 2 | HuBERT | IC | Clean, Noisy | McNemar |
| P4-2 | Fixed vs Heuristic | Setup 2 | HuBERT | KS | Clean, Noisy | McNemar |
| P4-3 | Fixed vs Heuristic | Setup 2 | HuBERT | ASR | Clean, Noisy | Bootstrap |

**Total tests: 6**.
Maps to: Table 3.

### Priority 5 — Cross-domain transfer

| ID | Comparison | Setup | Teacher | Tasks | Conditions | Test |
|----|-----------|-------|---------|-------|------------|------|
| P5-1 | L_KD vs L_CL | Setup 2 | HuBERT | ESC-50 | Clean, Noisy | McNemar |
| P5-2 | L_KD vs L_CL | Setup 2 | HuBERT | VocID | Clean, Noisy | McNemar |

**Total tests: 4**.
Maps to: Table 2 Block 3.

### Priority 6 — Mechanistic analysis (noise probes)

| ID | Comparison | Probe | Test |
|----|-----------|-------|------|
| P6-1 | L_KD vs L_CL (Setup 2) | LR-768 | McNemar |
| P6-2 | CC-only vs full L_CL | LR-768 | McNemar |
| P6-3 | Same-noise vs independent-noise L_CL | LR-768 | McNemar |
| P6-4 | Same-noise vs independent-noise L_CL | IC downstream | McNemar (clean + noisy) |

**Total tests: 5**.
Maps to: Tables 6, 7, 8.

### Comparisons NOT needed

| Skipped | Reason |
|---------|--------|
| GenreID | Acknowledged as unreliable (GTZAN dataset issues); results are 0 % Δ — no claim is made |
| SingerID L_KD vs L_CL | Clean accuracy *decreases*; chapter frames it as a trade-off, not an improvement |
| InstCls | Acknowledged augmentation conflict; no improvement claim |
| Setup 1 comparisons | Chapter presents Setup 2 as the primary configuration; Setup 1 is secondary |
| Layer-wise CCA/MI (Table 4) | Descriptive analysis, not a performance comparison — no test needed |
| Per-dimension noise analysis (Table 9) | Descriptive/mechanistic; counts of dimensions > 30 % are not performance claims |

---

## 4. Grand Total

| Priority | # Tests | Claims validated |
|----------|---------|-----------------|
| P1 | 8 | Core contribution |
| P2 | 4 | Teacher-agnostic |
| P3 | 4 | Self-correlation ablation |
| P4 | 6 | Heuristic weighting |
| P5 | 4 | Cross-domain transfer |
| P6 | 5 | Mechanistic analysis |
| **Total** | **31** | |

With Bonferroni correction across all 31 tests: α = 0.05 / 31 ≈ 0.0016.
Alternatively, apply Benjamini-Hochberg FDR control at q = 0.05 (recommended —
less conservative, more appropriate for correlated tests within the same
experimental family).

**Recommended approach:** Apply Benjamini-Hochberg within each priority group
separately (family-wise), then report per-group corrected p-values.

---

## 5. Artifacts Needed Per Checkpoint

To run all 31 tests, the following must be extracted from each model checkpoint:

### 5.1 Classification tasks (IC, KS, ER, ESC-50, VocID)

For each model × task × condition (clean/noisy):

```
predictions_{model}_{task}_{condition}.npy   # shape: (N_test,) — predicted class index
labels_{task}.npy                             # shape: (N_test,) — ground truth class index
```

The per-sample correctness vector is: `(predictions == labels).astype(int)`

### 5.2 ASR

For each model × condition:

```
hypothesis_{model}_{condition}.trn   # one hypothesis per line, utterance-ID appended
reference.trn                        # matching reference transcripts
```

Or equivalently, per-utterance error counts:

```
per_utt_errors_{model}_{condition}.json
# [{utt_id: str, n_sub: int, n_del: int, n_ins: int, n_ref_words: int}, ...]
```

### 5.3 Noise probes

For each probe model:

```
noise_predictions_{model}_{probe_type}.npy   # shape: (N_probe,) — predicted noise type
noise_labels.npy                              # shape: (N_probe,) — true noise type
```

For the RF probe: the 100 per-seed accuracy values (to compute CIs).

---

## 6. Checkpoints Inventory

List of model checkpoints needed (verify you still have these):

| # | Model description | Config |
|---|-------------------|--------|
| 1 | DistilHuBERT + L_KD, Setup 2, HuBERT teacher | Fixed weights |
| 2 | DistilHuBERT + L_CL (full), Setup 2, HuBERT teacher | Fixed weights |
| 3 | DistilHuBERT + CC-only (λ_sc=0), Setup 2, HuBERT teacher | Fixed weights |
| 4 | DistilHuBERT + L_CL, Setup 2, HuBERT teacher | Heuristic weights |
| 5 | DistilHuBERT + L_KD, Setup 2, HuBERT+ teacher | Fixed weights |
| 6 | DistilHuBERT + L_CL, Setup 2, HuBERT+ teacher | Fixed weights |
| 7 | DistilHuBERT + L_CL, Setup 2, HuBERT teacher, **same noise** | Fixed weights |

**Total: 7 checkpoints** × re-inference on all relevant tasks.

---

## 7. Implementation Plan

### Step 1: Verify checkpoint availability
Confirm all 7 checkpoints above are accessible.  If any are missing, flag
immediately — re-training a single seed takes ~X GPU-hours and may block the
timeline.

### Step 2: Extract per-sample predictions
Run inference on each checkpoint for each relevant task × condition.  Save
per-sample predictions as `.npy` files.  For ASR, save per-utterance
hypothesis transcripts.

### Step 3: Run statistical tests
A single Python script that:
1. Loads prediction pairs
2. Runs McNemar (classification) or paired bootstrap (ASR)
3. Collects raw p-values
4. Applies Benjamini-Hochberg correction per priority group
5. Outputs a summary table with: comparison, task, condition, metric values,
   test statistic, raw p, corrected p, significant (yes/no)

### Step 4: Integrate into thesis
- Add a footnote or sentence per table noting statistical significance
  (e.g., "† denotes p < 0.05 after BH correction")
- Do NOT add p-values to every cell — use daggers/symbols for significant
  differences and state the test in the experimental setup section
- Add one paragraph to Section 3.4 (Experimental Setup) describing the
  statistical testing methodology

### Step 5: Handle non-significant results
If any Priority 1 comparison is not significant:
- Check if the absolute difference is small enough that non-significance is
  expected (e.g., ER clean: 60.73 vs 64.71 — likely significant; ER noisy Δ
  of 0.08 pp — may not be)
- Adjust claims in prose accordingly (e.g., "improves" → "shows a trend
  toward improvement")
- Non-significant Priority 4–6 results are less critical and can be noted
  as trends

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Missing checkpoints | Medium | High — cannot recompute | Check immediately |
| ER differences not significant (small Δ) | High | Low — ER is not a headline result | Acknowledge in prose |
| KS differences not significant (0.08 pp clean gap) | Medium | Low | Same |
| Heuristic vs Fixed not significant (small margins) | High | Medium — weakens Table 3 | Frame as "comparable or marginally better" |
| ASR bootstrap p > 0.05 on clean (1.63 WER gap) | Low | High | LibriSpeech test-clean has ~2600 utts; 1.63 WER gap should be detectable |

---

## 9. What This Report Does NOT Cover

- **Chapters 4 and 5**: Separate plans needed; different models and tasks.
- **Effect sizes**: Consider reporting Cohen's h (for McNemar) or bootstrap
  confidence intervals on WER differences alongside p-values.
- **Multiple-seed runs**: If the examiner asks "why single seed?", the
  standard defence is computational cost of SSL pre-training.  Having
  per-sample significance tests on the downstream evaluation partially
  compensates for this.


