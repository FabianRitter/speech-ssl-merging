# Statistical Testing

All results in both papers come from **single pretraining seeds** (each SSL run
is ~200k steps), so tests that require repeated training runs (e.g. t-tests
across seeds) are not applicable. Instead, significance is assessed from
**per-sample predictions of a single checkpoint pair** on the same test set.

## Methodology

| Task type | Test | Details |
|-----------|------|---------|
| Classification (KS, IC, ER, SID, SingID, VocID, InstCls, GenreID, ESC-50) | **McNemar's test** | per-sample binary correctness; χ² with continuity correction on the 2×2 discordant table, exact binomial when discordant count < 25 |
| ASR (WER) | **Paired bootstrap** (Koehn, 2004) | WER is a corpus-level ratio, so utterances are the resampling unit: 10,000 resamples (fixed seed 1234), p = fraction of resamples where the WER difference flips sign; a 95% CI on ΔWER is reported |
| Cross-validated tasks (ER, ESC-50) | pooled | per-sample outcomes pooled across the 5 folds before McNemar |
| Multiple testing | **Benjamini–Hochberg FDR** at q = 0.05 | applied within each pre-registered comparison group (family), not globally |

Why McNemar and not a t-test: accuracies on a shared test set are paired, and
McNemar uses exactly the asymmetric disagreement (model A right / model B wrong
vs the reverse). Why bootstrap and not McNemar for ASR: ASR errors are per-word
within variable-length utterances; a binary per-utterance correctness would
throw away the structure and weight utterances incorrectly.

The full test plan with the comparison families, priorities, and rationale is in
`reports/statistical_testing_plan.md` (distillation-paper comparisons); the
task-arithmetic and permutation-merging comparison pairs are defined directly
in `reports/stat_tests/run_stat_tests_ch4_ch5.py`.

## Inputs: prediction dumps

The tests consume the per-sample files that downstream evaluation writes into
each experiment directory (see
[evaluation.md](evaluation.md#where-results-land)):

```
<model_dir>/<task>_paper_method[/_foldN]/
    test_predict.txt          # "<sample_id> <predicted_label>"
    test_truth.txt            # "<sample_id> <true_label>"
    # GTZAN: valid_predict.txt / valid_truth.txt
    # ER:    test_fold{N}_predict.txt / _truth.txt
    # ASR:   test-clean-noLM-hyp.ark / test-clean-noLM-ref.ark
    evaluation/<noise_type>/  # same files for noisy-condition evaluation
```

Only these small text files are needed — no checkpoints, no TensorBoard events.

## Runners

Two self-contained scripts under `reports/stat_tests/`:

| Script | Covers | Output |
|--------|--------|--------|
| `run_stat_tests.py` | distillation-paper comparisons (L_KD vs L_CL, self-correlation ablation, cross-domain transfer, same-noise ablation) | `RESULTS.md`, `RESULTS.tsv` |
| `run_stat_tests_ch4_ch5.py` | merging-paper comparisons (TA vs ensemble distillation, L_KD vs L_CL at each λ, pretraining-data variants, permutation vs TA, shared vs different init) | `RESULTS_ch4_ch5.md`, `RESULTS_ch4_ch5.tsv` |

Run from the `s3prl/` directory:

```bash
python reports/stat_tests/run_stat_tests.py
python reports/stat_tests/run_stat_tests_ch4_ch5.py
```

Both scripts are **safe to run at any time and degrade gracefully**: every test
whose prediction files are missing is listed under "NOT TESTED" with the reason
and the expected file location, so the output doubles as a live gap report.
Re-run after adding prediction files.

## Pointing the runners at your predictions

The scripts search a small list of roots for `<model_dir>/<task_subpath>`:

1. a staging tree, `reports/stat_tests/predictions/` (or
   `predictions_ch4_ch5/`), for prediction files collected from other machines;
2. the local `result/downstream/` tree, for evaluations run in place.

Configuration is via environment variables — no code edits needed:

```bash
# repo root (defaults to the s3prl/ directory the script lives in)
PROJECT_ROOT=/path/to/repo/s3prl python reports/stat_tests/run_stat_tests.py

# override a model's directory name if yours differs from the registry
M5_DIR=my_model_dirname M6_DIR=... python reports/stat_tests/run_stat_tests.py
```

Each script contains a model registry at the top (`MODELS = {...}`) mapping
comparison codes to downstream-experiment directory names; edit or override it
to match your experiment names. The task → directory mapping
(`speech_commands_paper_method`, `emotion_..._fold{1..5}`, etc.) follows the
naming produced by `cluster_scripts/run_downstream_local.sh`.

If your evaluations ran on several machines, copy the prediction files (only
`*_predict.txt`, `*_truth.txt`, `*.ark`) into the staging tree preserving the
`<model_dir>/<task_dir>/` structure — e.g. with `rsync` or an `rclone` cloud
remote when the machines cannot reach each other directly.

## Output format

`RESULTS*.md` contains one row per test:

```
comparison | task | condition | metric A | metric B | Δ | statistic | raw p | BH-corrected p | significant?
```

plus a per-family BH summary. `RESULTS*.tsv` is the machine-readable
equivalent. Use the corrected p-values when adding significance daggers to
paper tables, and state the methodology once in the experimental-setup section
(test, correction, q level) rather than repeating p-values in every cell.

## Caveats

- The "noisy" condition compares predictions from
  `evaluation/<noise_type>/` dirs (CHiME additive noise by default;
  `NOISE_COND` constant in the scripts).
- BH is applied within each comparison family; tests within a family are
  correlated (same models, related tasks), which makes BH mildly
  anti-conservative — interpret borderline results accordingly.
- Single-seed testing validates *downstream* differences between two specific
  checkpoints; it does not quantify pretraining-seed variance. This is the
  standard trade-off for SSL-scale pretraining and is discussed in the papers.
