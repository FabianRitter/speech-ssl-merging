# Multi-Teacher Knowledge Distillation

Train a small student model (2–3 transformer layers) that mimics the hidden
representations of one or two large self-supervised teachers (12 transformer
layers each). The student predicts teacher hidden states at layers 4, 8, and 12
through DistilHuBERT-style prediction heads; the heads are discarded after
pretraining and only the student encoder is used downstream.

## Teachers

| Name | Type | Domain | Source | Notes |
|------|------|--------|--------|-------|
| `hubert_base` | HuBERT Base | Speech | torch.hub (s3prl) | fairseq-style keys, no remapping needed |
| `mert_v0_public` | MERT v0 | Music | HuggingFace `m-a-p/MERT-v0-public` | keys remapped to fairseq layout (`attention.` → `self_attn.`, `feed_forward.*` → `fc1`/`fc2`, conv and pos_conv keys) |

The MERT key remapping is handled automatically inside the pretraining expert
(`convert_mert_state_dict()`); you never need to convert checkpoints manually.

## Student architectures

Two encoder widths are used across the papers:

| Property | Standard (768-d) | Wide (1536-d) |
|----------|------------------|---------------|
| `encoder_embed_dim` | 768 | 1536 |
| `encoder_layers` | 2 or 3 | 2 |
| `encoder_attention_heads` | 12 (head_dim 64) | 6 (head_dim 256) |
| `encoder_ffn_embed_dim` | 3072 | 3072 |
| `final_dim` (prediction head output) | 768 | 768 (still matches teacher) |
| CNN feature extractor | 7 conv layers, out dim 512 (identical to HuBERT) | same |
| `post_extract_proj` | Linear(512, 768) — can init from teacher | Linear(512, 1536) — **cannot** init from teacher |
| Init | conv + proj + first encoder layers from HuBERT (or MERT) | shared random base `theta_base_wide.pt` (conv from HuBERT, rest random) |
| Params (encoder + heads) | ~18M (2L) | ~44M (~72M total) |

Wide students exist to test whether merging benefits from extra width. Because the
1536-d projection cannot be initialized from a 768-d teacher, **all wide runs must
start from the same saved base state** so that task vectors are well-defined:

```bash
python create_wide_base_model.py
# -> result/pretrain/theta_base_wide/theta_base_wide.pt  (run ONCE, reuse for all wide runs)
```

Every wide pretraining config must then point `init_model_state` at that file.

## Losses

Configured via `loss_type` in the model config:

| `loss_type` | Loss (paper notation) | Description | Key parameters |
|-------------|----------------------|-------------|----------------|
| `l1` | L_KD | L1 distance to teacher hidden states + cosine-similarity term (DistilHuBERT objective) | `cosine_loss: 1.0` |
| `barlow_old` | L_CL | Barlow-Twins-style correlation loss: cross-correlation to teacher + optional self-correlation term, computed at frame level | `lambda_coefficient` (cross-corr weight), `lambda_coeff_self_corr` (self-corr weight), `off_diag_self_cor_scale`, `frame_level_barlow_scale`, `use_average_pooling: true`, `average_pooling_value: 150`, `self_correlation: true` |
| `barlow` | L_CL (numerically hardened) | Same as `barlow_old` but with fp32 casting, epsilon 1e-4, and no in-place ops | same |

Notes:

- All paper experiments use `barlow_old`. Use it if you want to match published
  numbers; use `barlow` only if you hit numerical issues with mixed precision.
- The cosine term (`cosine_loss > 0`) is added on top of either loss and applied
  per predicted teacher layer (4, 8, 12).
- Average pooling (`average_pooling_value: 150`) reduces the time dimension
  **before** the loss, in the pretraining expert's forward pass.

## Configs

Model configs (architecture + loss) live next to the pretraining experts. Runner
configs (optimizer, steps, data) are separate.

| Config | Teachers | Student | Loss |
|--------|----------|---------|------|
| `pretrain/distiller/config_model.yaml` | HuBERT only | 2 layers, 768-d, init from HuBERT | L1 + cosine (DistilHuBERT) |
| `pretrain/multi_distiller/config_model.yaml` | MERT + HuBERT | 2 layers, 768-d, init from HuBERT | L1 + cosine |
| `pretrain/multi_distiller/config_model_single_teacher.yaml` | MERT + HuBERT | 3 layers, 768-d, init from HuBERT | L1 + cosine |
| `pretrain/multi_distiller/config_model_mert_barlow_2layers.yaml` | MERT only | 2 layers, 768-d, init from HuBERT | Barlow (L_CL) + cosine |
| `pretrain/multi_distiller/config_runner.yaml` | — | — | runner: 200k steps, lr 2e-5, batch 24, AdamW, fp16 |

To distill a **single-domain student** with the multi_distiller code path (e.g. a
music-only student for later merging), set `teacher_names` to one teacher and keep
`initialize_from: [hubert_base]` so both students share the HuBERT init.

Key config fields:

- `teacher_names` — which teachers to load (order matters for multi-teacher loss)
- `initialize_from` — which teacher provides the student init
- `init_teacher_conv_layers` / `init_teacher_encoder_layers` — copy conv+proj and
  first encoder layers from the teacher (disable both for wide models)
- `pred_layer_id: [4, 8, 12]` — teacher layers the student predicts
- `translator_type: avgpool` — how multiple teachers' features are combined
- `task.sequence_length: 250000` — training crop (~15.6 s at 16 kHz)

## Pretraining data

- Dataset: LibriSpeech 960 h (`train-clean-100`, `train-clean-360`,
  `train-other-500`); dev set `dev-clean`.
- Set the root in `pretrain/multi_distiller/config_runner.yaml`
  (`pretrain_expert.datarc.libri_root`) or override on the command line (below).
- Sequence-length bucket files are expected under `data/len_for_bucket/`
  (`pretrain_expert.datarc.file_path`). Generate them with the standard s3prl
  preprocessing script (`preprocess/generate_len_for_bucket.py`) if you use a
  fresh LibriSpeech copy.
- Some paper variants train on LibriSpeech + Music4All (+ AudioSet). The runner
  config contains the per-corpus normalization statistics as comments; select the
  matching `data_stats` block for your training mixture.

## Launching

Direct:

```bash
python run_pretrain.py \
    -u multi_distiller \
    -g pretrain/multi_distiller/config_model.yaml \
    -n my_experiment_name \
    -o "config.pretrain_expert.datarc.libri_root=$LIBRISPEECH_ROOT,,config.pretrain_expert.datarc.file_path=./data/len_for_bucket"
```

- `-u` selects the pretraining expert: `distiller` (single-teacher) or
  `multi_distiller` (one or more teachers).
- `-g` is the model config; the runner config defaults to
  `pretrain/<upstream>/config_runner.yaml` (override with `-c`).
- `-o` overrides any config value; separate multiple overrides with `,,`.
- `-n` names the experiment; outputs go to `result/pretrain/<name>/`
  (checkpoints `states-<step>.ckpt`, copied configs, TensorBoard events).
- Resume with `-e result/pretrain/<name>/states-<step>.ckpt`.

Cluster launchers (edit the header for your site) are in `cluster_scripts/`:

- `cluster_scripts/run_pretrain.sh` — background/nohup launcher template
- `cluster_scripts/run_downstream_sbatch.sh` — SLURM template (adapt for pretraining)

Wide models: add `-o "config.multi_distiller.init_model_state=result/pretrain/theta_base_wide/theta_base_wide.pt"`
(or set `init_model_state` in the config) and disable teacher init flags.

## Exporting the best checkpoint

Pretraining saves periodic `states-<step>.ckpt` files (`save_step: 10000`,
`max_keep: 2`). To select the checkpoint with the lowest dev loss after (or
during) training:

```bash
python get_best_checkpoint.py \
    -u multi_distiller \
    -g pretrain/multi_distiller/config_get_checkpoint.yaml \
    -n my_experiment_name \
    --find_best_checkpoint
```

This iterates all `states-*.ckpt` files in `result/pretrain/<name>/`, computes the
dev-set loss for each, and saves the winner as
`result/pretrain/<name>/best-train-loss.ckpt`.

For merging experiments the papers instead use **fixed-step checkpoints**
(`states-200000.ckpt` or `states-220000.ckpt`) so that both endpoints of a merge
have identical training budgets — see the reproduction guides.

## Checkpoint format

Distilled checkpoints are dicts with keys:

- `Distiller` — student state dict
- `Config` — the upstream (model) config used to rebuild the architecture at
  evaluation time; `multi_distiller` checkpoints store it under the
  `multi_distiller` key, `distiller` checkpoints under `distiller`
- `Optimizer`, `Step` — training state

The `multi_distiller_local` upstream wrapper
(`upstream/multi_distiller/builder.py`) reads `Config` and instantiates the right
architecture automatically, including wide models (`encoder_embed_dim: 1536`).
Both config keys are handled with a fallback, and the merge scripts normalize
`distiller` → `multi_distiller` when needed.
