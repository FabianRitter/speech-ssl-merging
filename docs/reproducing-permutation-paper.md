# Reproducing the Permutation-Merging Paper

Step-by-step recipe for the correlation-based permutation-merging paper
(Ritter-Gutierrez et al., "A correlation-permutation approach for speech-music
encoders model merging," ASRU 2025,
[arXiv:2506.11403](https://arxiv.org/abs/2506.11403)). Read
[permutation-merging.md](permutation-merging.md) and
[evaluation.md](evaluation.md) first; this page only sequences them.

The paper has two experiment families:

- **A. Teacher merging** — merge the full pretrained HuBERT Base and MERT
  teachers (12 layers, 768-d) under different alignment configurations.
- **B. Distilled-student merging** — merge pairs of distilled students
  (the same endpoints as the task-arithmetic paper) with permutation alignment,
  and compare against task arithmetic at matched λ.

## 0. Environment

Same as the task-arithmetic paper:

```bash
conda create -n merge-ssl python=3.8 && conda activate merge-ssl
pip install torch torchaudio
pip install -e .
cd s3prl
```

## 1. Data preparation

- Downstream datasets: as in [evaluation.md](evaluation.md#dataset-paths).
- **Calibration data** for the permutation step: 5000 LibriSpeech
  train-clean-100 utterances + 5000 Music4All clips. Set
  `merging_utils/data_config.yaml`:

  ```yaml
  datarc:
      train: ['combined-shuffled']
      libri_root: $DATA_ROOT          # must contain LibriSpeech/ and music4all/
      bucket_file: ./data/len_for_bucket
  ```

  The `combined-shuffled` bucket file (shipped under `data/len_for_bucket/`)
  indexes exactly this speech+music mixture; use the shuffled variant to avoid
  ordering bias in batch-level correlation estimates. Speech-only calibration
  (`train-clean-100-5000-samples`) is used in the calibration-data ablation.

## 2. Checkpoints

- Teachers download automatically (torch.hub / HuggingFace Hub) on first use.
- Distilled endpoints: reuse the students from the task-arithmetic
  reproduction (step 3 of
  [reproducing-task-arithmetic-paper.md](reproducing-task-arithmetic-paper.md)),
  or download from the HuggingFace Hub release (link will appear in the README).
- The **different-init** ablation additionally needs a 2L L_KD pair whose music
  student is *not* HuBERT-initialized (independent init); train it by disabling
  `init_teacher_conv_layers`/`init_teacher_encoder_layers` for the music run.

## 3. Experiment family A — teacher merging

One command per row of the alignment-configuration table. All use Hungarian
matching (`match_tensors_permute`, the default) and interpolation weights
0.9/0.1 unless the paper says otherwise; the proposed configuration is
**CNN + ff+attn**.

```bash
# Proposed: CNN + ff+attn
python prepare_hubert_mert_merging_permutation_style.py \
    -n teacher_cnn_ffattn --model_mode teacher \
    --merge_type ff+attn --merge_cnn --experiment main

# Ablations (change only the flagged parts):
#   CNN only:            --merge_type none  --merge_cnn
#   CNN + ff_only:       --merge_type ff_only --merge_cnn
#   CNN + all:           --merge_type all   --merge_cnn
#   ff+attn only:        --merge_type ff+attn            (no --merge_cnn)
#   no permutation:      --merging_algorithm match_tensors_identity --merge_type ff+attn --merge_cnn
```

The naive-average baseline (0.5/0.5 without alignment) and weighted no-perm
baseline (0.9/0.1) come from the identity-matching runs at the corresponding
`--interpolation_weights`.

Optional variants reported in the paper:

- **TIES-adaptation**: add `--use_ties --quantile 0.8` (keeps HuBERT as the
  reference via `--maintain_hubert_behavior`).
- **ZipIt!**: `--merging_algorithm match_tensors_zipit --zipit_a 0.3 --zipit_b 0.125`.

## 4. Experiment family B — distilled-student merging

For each endpoint pair (2L L_KD, 2L L_CL, 3L L_KD, 3L L_CL, wide L_CL, and the
different-init 2L L_KD pair), run:

```bash
python prepare_hubert_mert_merging_permutation_style.py \
    -n perm_2L_LKD \
    --model_mode distilled \
    --model1_ckpt result/pretrain/SPEECH_STUDENT/states-200000.ckpt \
    --model2_ckpt result/pretrain/MUSIC_STUDENT/states-200000.ckpt \
    --merge_type ff+attn --merge_cnn \
    --merging_algorithm match_tensors_permute \
    --interpolation_weights 0.9 0.8 \
    --experiment main
```

Model 1 is the reference (stays unpermuted); model 2 is aligned toward it. The
paper's headline comparisons use λ = 0.9 and λ = 0.8.

Method variants (same command, different flags):

| Variant | Flags |
|---------|-------|
| Single-pass Hungarian (main) | as above |
| Multi-pass alignment | `--num_passes 2` (or more) |
| Sinkhorn on collected features | `--merging_algorithm match_tensors_sinkhorn_cc` |
| Sinkhorn end-to-end | `--merging_algorithm match_tensors_sinkhorn_cc --sinkhorn_e2e` |
| Speech-only calibration ablation | set `train: ['train-clean-100-5000-samples']` in `merging_utils/data_config.yaml` |

The matched task-arithmetic baselines at the same λ come from
[reproducing-task-arithmetic-paper.md](reproducing-task-arithmetic-paper.md)
step 4.

## 5. Evaluate

Evaluate every merged checkpoint on the task set, exactly as for any distilled
model:

```bash
bash cluster_scripts/run_downstream_local.sh \
    --checkpoint "result/merged_pretrain_upstream/permutation-covariance/<dir>/merged_..._interp_0.9_0.1.ckpt" \
    --gpus 0,1,2,3,4,5,6,7,0 \
    --tasks asr,ks,ic,sid,er,singid,vocid,instcls,pitchid,genreid,aec_esc50 \
    --upstream multi_distiller_local --stage train
```

A SLURM batch template for sweeping several merged checkpoints × tasks is in
`cluster_scripts/` (permutation sbatch template) — edit the checkpoint list and
partition header for your site.

## 6. Statistical tests

Groups C5-1 (permutation vs task arithmetic, per pair and λ) and C5-2
(shared-init vs different-init) are implemented in
`reports/stat_tests/run_stat_tests_ch4_ch5.py`; the teacher-merging table's
McNemar daggers follow the same McNemar + BH procedure with the proposed
configuration as the reference model.

```bash
python reports/stat_tests/run_stat_tests_ch4_ch5.py
# -> reports/stat_tests/RESULTS_ch4_ch5.md / .tsv
```

Stage prediction files / adjust the model registry as described in
[statistical-testing.md](statistical-testing.md).

## 7. Figures (`analysis_scripts/`)

| Script | Figure | Inputs |
|--------|--------|--------|
| `analysis_scripts/run_fig52_heatmaps_RTX.sbatch` | correlation-heatmap figure + `.npz` correlation dumps (runs the main script with `--run_correlation_heatmap_viz --merge_type ff+attn --merge_cnn --correlation_viz_nodes ...`) | teachers + calibration data |
| `analysis_scripts/fig_per_channel_correlation.py` | per-channel cross-correlation before vs after permutation (sorted, per merge node) | the `.npz` dumps from the heatmap run — update `NPZ_DIR` at the top |
| `analysis_scripts/exp_8A_alignment_cost.py` | per-node alignment-cost bar charts, shared vs divergent init | alignment costs logged by the merging runs (values embedded in the script; regenerate from your logs) |
| `analysis_scripts/fig_5_4_cp_vs_ta_distilled.py` | permutation vs task-arithmetic average-score comparison at λ = 0.9 / 0.8 | aggregate scores from step 5 (values embedded; update from your results) |
| `analysis_scripts/fig_ic_stability.py` | IC-accuracy stability spotlight (permutation preserves IC at λ = 0.8 where interpolation collapses) | IC accuracies from step 5 |

Permutation-intensity analysis (how strongly each layer is reordered) is
produced by the main script itself with `--make_permutation_analysis`.

Note: the plotting scripts embed the paper's aggregated numbers as literals for
layout reproducibility. To plot **your** reproduction, replace the data blocks
at the top of each script with the metrics from your step-5 result directories.
