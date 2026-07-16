# Correlation-Based Permutation Merging

Plain interpolation assumes the two models' weight spaces are aligned — roughly
true only when both share the same initialization. Permutation merging removes
that assumption: it first **aligns the internal representations** of model B to
model A using permutations computed from feature correlations on calibration
data, then interpolates the aligned weights.

The approach draws on ZipIt! (Stoica et al., 2023) and permutation-based
transformer merging (Imfeld et al., 2024).

## Pipeline overview

Entry point: `prepare_hubert_mert_merging_permutation_style.py`.

1. **Load models** — either two full teachers (HuBERT Base + MERT, with MERT
   keys remapped to the HuBERT namespace) or two distilled checkpoints.
2. **Build graphs** — each model is wrapped in a computational-graph
   representation (`graphs/`) that marks *merge points* and encodes how a
   permutation at one node propagates to the weights that consume it.
3. **Collect features** — calibration audio is passed through both models;
   intermediate activations are captured at every merge point via forward hooks
   (padding removed, batches concatenated).
4. **Match** — at each merge point, a matching algorithm converts the
   cross-model correlation matrix into a (soft or hard) permutation of model B's
   units toward model A.
5. **Merge** — permuted weights are combined: interpolation at one or more
   weights, optionally with TIES trimming/sign election.
6. **Save** — one checkpoint per interpolation weight, in the same format as a
   distilled checkpoint, ready for `multi_distiller_local` downstream evaluation.

## Graph representation (`graphs/`)

Models are represented as DAGs whose nodes are parameter tensors or structural
markers (`graphs/base_graph.py`, `NodeType` enum):

| Node type | Meaning | Example |
|-----------|---------|---------|
| `MODULE` | a parameter tensor | `self_attn.q_proj.weight` |
| `PREFIX` | a merge point / namespace marker where features are hooked | node before `fc2` |
| `SUM` | residual addition | residual after attention |

Files:

- `graphs/base_graph.py` — `BaseGraph` node/edge management
- `graphs/hubert_graph.py` — HuBERT graph with configurable merge points
- `graphs/hubert_graph_complete_merge.py` — extended graph for full-model
  merging (CNN + all encoder layers); auto-detects `num_layers`, `num_heads`,
  and embedding dim from the model, so it handles 12-layer teachers, 2–3-layer
  students, and wide (1536-d, 6-head) students with the same code
- `graphs/transformer_enc_graph.py` — subgraph for one encoder layer

## Matching algorithms (`matching_functions.py`)

Select with `--merging_algorithm <function_name>`:

| Function | Idea |
|----------|------|
| `match_tensors_permute` (default) | Hungarian assignment (`scipy.optimize.linear_sum_assignment`) on the cross-model correlation matrix — a strict one-to-one permutation |
| `match_tensors_permute_MHA` | Hungarian variant that respects multi-head attention structure (permutes within/across heads consistently, head count read from the model) |
| `match_tensors_permute_symmetric` / `..._symmetric_MHA` | symmetric variant: aligns both models toward a common midpoint instead of B→A |
| `match_tensors_zipit` / `match_tensors_zipit_MHA` | ZipIt!-style greedy correlation matching with a merge budget; hyperparameters `--zipit_a` (similarity decay) and `--zipit_b` (within-model budget) |
| `match_tensors_sinkhorn_cc` / `..._MHA` | differentiable Sinkhorn relaxation: optimizes a soft permutation with temperature annealing against a cross-correlation objective, then hardens it via Hungarian |
| `match_tensors_identity` | no-op matching (ablation: interpolation inside the permutation pipeline) |

Utility: `compute_correlation()` converts covariance to correlation.

`--merging_strategy` chooses `uniform` (one algorithm everywhere) or
`hybrid_zipit_cnn_permute_transformer` (ZipIt! for the CNN, permutation for the
transformer).

### Sinkhorn variants

Two Sinkhorn modes exist:

- **Feature-space Sinkhorn** (`--merging_algorithm match_tensors_sinkhorn_cc`):
  optimizes each merge point's soft permutation on pre-collected features.
- **End-to-end Sinkhorn** (add `--sinkhorn_e2e`): implemented in
  `sinkhorn_e2e.py` (`PermutedModelWrapper`, `optimize_sinkhorn_e2e()`). Soft
  permutations are inserted into model B's forward pass so gradients flow
  through the whole layer chain — a global search rather than per-layer greedy
  alignment.

Sinkhorn hyperparameters: `--sinkhorn_iters` (row/col normalizations, 20),
`--sinkhorn_opt_steps` (300), `--sinkhorn_lr` (0.01), `--sinkhorn_tau_max` /
`--sinkhorn_tau_min` (temperature annealing 1.0 → 0.01).

Multi-pass alignment: `--num_passes N` repeats the align-and-permute loop N
times (each pass aligns the already-permuted model B toward A again), logging
the permutation delta between passes.

## Merge orchestration (`merging_utils/model_merger_new.py`)

The `ModelMerge` class drives steps 3–5:

- registers forward hooks at PREFIX nodes and collects features
  (`compute_metrics`, `remove_pads_dynamic`),
- computes cross-model correlations per merge point (`compute_metric_corrs`),
- calls the selected matching function and propagates merge/unmerge matrices
  through the graph (`compute_transformations`, `apply_transformations_custom`),
  including the "unpermute before the next layer" bookkeeping,
- produces the merged state dict (`get_merged_state_dict`), with optional TIES,
- exposes analysis utilities (feature similarity, correlation dumps).

## What gets permuted: `--merge_type` and `--merge_cnn`

`--merge_type` controls where PREFIX (merge-point) nodes are placed inside each
transformer layer:

| `--merge_type` | Aligned parts |
|----------------|---------------|
| `ff_only` | feed-forward layers only |
| `ff+attn` (default) | feed-forward + attention output projection |
| `qkv` | query/key/value projections only |
| `qkv+attn` | Q/K/V + attention output projection |
| `qkv+ff` | Q/K/V + feed-forward |
| `all` | all of the above (incl. layer norms) |
| `none` | no transformer alignment |

`--merge_cnn` additionally inserts merge points after each CNN feature-extractor
block, so convolutional channel orderings are aligned too; without it the CNN is
interpolated unaligned. The configuration used for the main paper results is
**CNN + ff+attn** (`--merge_type ff+attn --merge_cnn`).

## Merge combination: interpolation and TIES-adaptation

After alignment, weights are combined per interpolation weight from
`--interpolation_weights w1 [w1 ...]` (model 2 weight = 1 − w1; a default sweep
0.9/0.8/0.5/0.2/0.1 is used when omitted).

With `--use_ties`, the correlation-adapted TIES implementation
(`merging_utils/ties_merger_adaptation_to_corr.py`) is used instead of plain
interpolation: it trims the smallest inter-model weight differences by quantile
(`--quantile`, default 0.8 = keep top 20%), resolves sign conflicts, and keeps
the reference model's behavior where `--maintain_hubert_behavior` is set
(default: model 1 / HuBERT is the reference and stays un-permuted; model 2 is
permuted toward it).

## Model modes: teacher vs distilled

| | `--model_mode teacher` (default) | `--model_mode distilled` |
|---|---|---|
| Inputs | full 12-layer HuBERT Base + MERT (loaded automatically) | any two distilled checkpoints via `--model1_ckpt` / `--model2_ckpt` |
| Key remapping | MERT remapped to HuBERT namespace | none (both already HuBERT-style) |
| State-dict prefix | `model.` (s3prl hub wrapper) | none |
| Graph dims | 12 layers / 12 heads / 768-d | auto-detected (2–3 layers; 768-d/12-head or 1536-d/6-head) |
| Checkpoint format in | `model_cfg` + `model_weight` | `Config` + `Distiller` |
| Checkpoint format out | HuBERT-Base-style | `Config` + `Distiller` (downstream-ready) |

In distilled mode a `DistilledModelWrapper` adapts
`MultiDistillerModel.forward(wave, pad_mask)` to the `forward(wavs)` interface
the graph expects. The distillation prediction heads (`output_layer`) are
excluded from the graph and not merged — they are unused downstream.

## Calibration data

Permutations are computed from feature correlations on a small calibration set,
configured in `merging_utils/data_config.yaml`:

- `libri_root` — root containing the audio (LibriSpeech, and Music4All for the
  combined sets); set this to your `$DATA_ROOT`.
- `train` — which bucket file (under `data/len_for_bucket/`) to draw calibration
  batches from:

| Bucket file | Content |
|-------------|---------|
| `train-clean-100-5000-samples` | 5000 LibriSpeech train-clean-100 utterances (speech only) |
| `combined-librispeech-100-5000samples-and-5000-samples-music4all` | 5000 speech + 5000 music clips |
| `combined-shuffled` | same combined data, shuffled — **recommended** (avoids ordering bias in batch-level correlation estimates) |

Use the combined-shuffled set whenever one of the models is a music model.

## Usage examples

Merge two distilled students (main-paper configuration):

```bash
python prepare_hubert_mert_merging_permutation_style.py \
    -n perm_2L_LKD \
    --model_mode distilled \
    --model1_ckpt result/pretrain/SPEECH_STUDENT/states-200000.ckpt \
    --model2_ckpt result/pretrain/MUSIC_STUDENT/states-200000.ckpt \
    --merge_type ff+attn --merge_cnn \
    --merging_algorithm match_tensors_permute \
    --interpolation_weights 0.9 0.8 0.5 0.2 0.1 \
    --experiment main
```

Merge the full pretrained teachers (HuBERT Base + MERT):

```bash
python prepare_hubert_mert_merging_permutation_style.py \
    -n hubert_mert_teacher_merge \
    --merge_type ff+attn --merge_cnn \
    --merging_algorithm match_tensors_permute \
    --experiment main
```

Sinkhorn end-to-end variant:

```bash
python prepare_hubert_mert_merging_permutation_style.py \
    -n perm_sinkhorn_e2e \
    --model_mode distilled \
    --model1_ckpt ... --model2_ckpt ... \
    --merge_type ff+attn --merge_cnn \
    --merging_algorithm match_tensors_sinkhorn_cc --sinkhorn_e2e \
    --interpolation_weights 0.9 0.8
```

## Outputs

By default (`-p/--expdir` unset), merged checkpoints are written to:

```
result/merged_pretrain_upstream/permutation-covariance/
  <expname>_merge_cnn_<bool>_use_ties_<bool>_quantile_<q>_maintain_hubert_behavior_<bool>/
    merged_..._interp_0.9_0.1.ckpt
    merged_..._interp_0.8_0.2.ckpt
    ...
```

Each checkpoint is evaluated exactly like a distilled model (see
[evaluation.md](evaluation.md)):

```bash
bash cluster_scripts/run_downstream_local.sh \
    --checkpoint result/merged_pretrain_upstream/permutation-covariance/<dir>/merged_..._interp_0.9_0.1.ckpt \
    --gpus 0,1,2 --tasks ks,singid,genreid \
    --upstream multi_distiller_local --stage train
```

A SLURM template for evaluating a batch of permutation-merged checkpoints is
provided in `cluster_scripts/` (permutation sbatch template).

## Analysis modes

These flags run an analysis and exit without saving a merged model:

| Flag | What it produces |
|------|------------------|
| `--make_permutation_analysis` | permutation-intensity analysis: fraction of non-identity entries per layer's permutation matrix (reference model ≈ 0%; the permuted model shows the real reordering); bar charts + JSON |
| `--run_correlation_heatmap_viz` | cross-model channel-correlation heatmaps before vs after permutation, plus `.npz` dumps of the correlation matrices; tune with `--correlation_viz_nodes`, `--correlation_viz_max_dim`, `--heatmap_vmin/--heatmap_vmax` |
| `--run_feature_similarity` | cosine similarity between the two models' representations across layers before/after alignment (`--sim_analysis_batches`, `--sim_analysis_nodes_count`, `--exit_after_similarity`) |

The figure scripts in `analysis_scripts/` post-process these outputs — see
[reproducing-permutation-paper.md](reproducing-permutation-paper.md).

## Relationship to task arithmetic

| Aspect | Task arithmetic | Permutation merging |
|--------|-----------------|---------------------|
| Alignment | none (relies on shared init) | explicit per-layer permutation |
| Data needed | none | small calibration set (forward passes) |
| Cost | seconds (state-dict arithmetic) | minutes–hours (feature collection + matching) |
| Different inits | not applicable | works (shared init still helps) |
| Entry point | `merge_two_models.py` / `merge_wide_models.py` | `prepare_hubert_mert_merging_permutation_style.py` |
