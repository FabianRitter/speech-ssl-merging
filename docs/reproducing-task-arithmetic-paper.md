# Reproducing the Task-Arithmetic Paper

Step-by-step recipe for the multi-teacher distillation + task-arithmetic paper
(Ritter-Gutierrez et al., "Distilling a speech and music encoder with task
arithmetic," Interspeech 2025,
[arXiv:2505.13270](https://arxiv.org/abs/2505.13270)). Read
[distillation.md](distillation.md), [task-arithmetic.md](task-arithmetic.md),
and [evaluation.md](evaluation.md) first; this page only sequences them.

Do not expect bit-identical numbers: results depend on GPU nondeterminism and
library versions. The statistical tests (step 6) are the right lens for
comparing your reproduction against the paper tables — see paper for the
reference numbers.

## 0. Environment

```bash
conda create -n merge-ssl python=3.8 && conda activate merge-ssl
pip install torch torchaudio          # match your CUDA version
pip install -e .                      # from the repo root
cd s3prl                              # all commands below run from here
```

## 1. Data preparation

- **LibriSpeech 960h** (pretraining + ASR): download `train-clean-100`,
  `train-clean-360`, `train-other-500`, `dev-clean`, `test-clean` to
  `$LIBRISPEECH_ROOT`. Generate bucket files into `data/len_for_bucket/` with
  `preprocess/generate_len_for_bucket.py` if not present.
- **Downstream datasets**: Speech Commands v0.01, Fluent Speech Commands,
  VoxCeleb1, IEMOCAP, VocalSet, NSynth, GTZAN, ESC-50 — see
  [evaluation.md](evaluation.md#dataset-paths).
- **Optional (data-variant ablation)**: Music4All and AudioSet balanced subset,
  only needed for the pretraining-data ablation (step 3c).

## 2. Checkpoints

The endpoint students will be released on the HuggingFace Hub (link will
appear in the README) — with them you can skip step 3 entirely and start at
step 4. To train from scratch, continue below.

## 3. Train the endpoint students

Each merge needs a **pair** of single-teacher students with identical
initialization. Budget: each run is 200k–300k steps on one modern GPU.

### 3a. Standard 2-layer pair, L_KD loss

```bash
# Speech student (HuBERT teacher, HuBERT init)
python run_pretrain.py -u distiller \
    -g pretrain/distiller/config_model.yaml \
    -n distilhubert_speech_2L_lkd \
    -o "config.pretrain_expert.datarc.libri_root=$LIBRISPEECH_ROOT"

# Music student (MERT teacher, HuBERT init) — same seed, same init
python run_pretrain.py -u multi_distiller \
    -g pretrain/multi_distiller/config_model.yaml \
    -n distilmert_music_2L_lkd \
    -o "config.multi_distiller.teacher_names=['mert_v0_public'],,config.multi_distiller.initialize_from=['hubert_base'],,config.pretrain_expert.datarc.libri_root=$LIBRISPEECH_ROOT"
```

Both trained on LibriSpeech 960h; merged at `states-220000.ckpt` in the paper.

### 3b. The other pairs

Repeat 3a with the following variations (one pair per row of the paper's
grid); merge checkpoints at the step given in the paper (200k for 3L/wide):

| Pair | Change vs 3a |
|------|--------------|
| 2L L_CL | Barlow loss: use `pretrain/multi_distiller/config_model_mert_barlow_2layers.yaml` for the music student and the analogous HuBERT-teacher Barlow config (`loss_type: barlow_old`, cross-corr 5e-5 / self-corr 5e-6 as in the paper) for the speech student |
| 3L L_KD | `config_model_single_teacher.yaml` (3 encoder layers), one teacher per run |
| 3L L_CL | 3-layer + Barlow loss (combine the two changes above) |
| Wide L_KD / Wide L_CL | run `python create_wide_base_model.py` once, then set `init_model_state=result/pretrain/theta_base_wide/theta_base_wide.pt` and `encoder_embed_dim=1536`, `encoder_attention_heads=6` via config/overrides for all four wide runs |

### 3c. Baselines

- **Ensemble (multi-teacher) distillation**: same configs with both teachers
  active (`teacher_names: [mert_v0_public, hubert_base]`) — 2L, 3L, and wide
  variants.
- **Teachers**: HuBERT Base and MERT v0 are evaluated directly as built-in
  upstreams (no training).
- **Pretraining-data variants** (ablation): retrain the 2L L_KD pair on
  LibriSpeech+Music4All and LibriSpeech+Music4All+AudioSet; select the matching
  `data_stats` block in the runner config.

## 4. Merge

For each pair, produce merged checkpoints at
λ1 ∈ {0.9, 0.8, 0.5, 0.2, 0.1} (λ2 = 1 − λ1):

```bash
# Standard pairs (2L and 3L):
for L in 0.9 0.8 0.5 0.2 0.1; do
  python merge_two_models.py \
      --speech_ckpt result/pretrain/distilhubert_speech_2L_lkd/states-220000.ckpt \
      --music_ckpt  result/pretrain/distilmert_music_2L_lkd/states-220000.ckpt \
      --save_path   "result/pretrain/ta_2L_lkd_speech_${L}/learning_by_addition.ckpt" \
      --lambda1 $L --lambda2 $(python -c "print(round(1-$L,1))")
done

# Wide pairs (all 5 λ in one call):
python merge_wide_models.py --loss_type l1     --speech_step 200000 --music_step 200000
python merge_wide_models.py --loss_type barlow --speech_step 200000 --music_step 200000
```

For 3-layer pairs pass `--base_config pretrain/multi_distiller/config_model_single_teacher.yaml`
so the reconstructed base has 3 layers.

TIES ablation: re-run selected merges with `--use_ties --k 0.3`
(optionally `--enable_sign_interference --sign_as_in_ties`).

## 5. Evaluate

Evaluate every endpoint, baseline, and merged checkpoint on the full task set:

```bash
bash cluster_scripts/run_downstream_local.sh \
    --checkpoint result/pretrain/<MODEL_DIR>/<ckpt> \
    --gpus 0,1,2,3,4,5,6,7,0,1,2 \
    --tasks asr,ks,ic,sid,er,singid,vocid,instcls,pitchid,genreid,aec_esc50 \
    --upstream multi_distiller_local --stage train
```

- Teachers: use `-u hubert` / `-u mert_v0_public` style built-in upstreams
  without `-k` and without `-s` — the default lets the downstream head learn a
  weighted linear combination of all SSL layers (`-s paper` is only for
  distilled/merged students).
- Aggregate scores in the paper use the SUPERB-style average over the task set;
  compute them from the per-task metrics in each experiment dir.
- Multi-node/PBS templates: `cluster_scripts/run_downstream_enroot_2026.sh`,
  and `cluster_scripts/run_merge_and_downstream_enroot_2026.sh` for the fused
  merge→evaluate sweep.

## 6. Statistical tests

The merging-paper test plan (groups C4-1 TA vs ensemble, C4-2 L_KD vs L_CL per
λ, C4-3 data variants) is implemented in
`reports/stat_tests/run_stat_tests_ch4_ch5.py`:

```bash
python reports/stat_tests/run_stat_tests_ch4_ch5.py
# -> reports/stat_tests/RESULTS_ch4_ch5.md / .tsv
```

Edit the model registry at the top of the script (or stage predictions under
`reports/stat_tests/predictions_ch4_ch5/`) so the directory names match your
experiment names. See [statistical-testing.md](statistical-testing.md).

## 7. Figures

- `interpolation_weights_performances_radar_plot.ipynb` — per-task radar plots
  across interpolation weights (endpoint vs merged models).
- `analysis-merged-models.ipynb` — exploratory comparison of merged-model
  metrics.
- The λ-sweep line plots in the paper are produced from the downstream metric
  tables; the notebook cells document the exact task groupings.

Each figure script/notebook reads the downstream result directories from step 5;
update the paths at the top of each file to your `result/` locations.
