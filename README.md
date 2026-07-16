# Task Arithmetic and Permutation Merging for Distilled Speech/Audio SSL Models

This repository contains the code for two papers on building compact, multi-domain
speech/music/audio representation models. Small
student models are first distilled from large self-supervised teachers (HuBERT for
speech, MERT for music), then **merged post-hoc** either by task-vector
interpolation (task arithmetic) or by correlation-based permutation alignment
followed by interpolation. All models are evaluated with frozen features on 11
downstream tasks spanning speech (SUPERB), music (MARBLE), and environmental audio.

The code is a fork of [s3prl](https://github.com/s3prl/s3prl) v0.4.15.

## Papers

If you use this code, please cite:

**Paper 1 — task arithmetic for distillation**

> Fabian Ritter-Gutierrez, Yi-Cheng Lin, Jui-Chiang Wei, Jeremy H.M. Wong,
> Eng Siong Chng, Nancy F. Chen, Hung-yi Lee, "Distilling a speech and music
> encoder with task arithmetic," *Proc. Interspeech 2025*, pp. 3858–3862.
> [arXiv:2505.13270](https://arxiv.org/abs/2505.13270)

```bibtex
@inproceedings{rittergutierrez25_interspeech,
  title     = {Distilling a speech and music encoder with task arithmetic},
  author    = {Fabian Ritter-Gutierrez and Yi-Cheng Lin and Jui-Chiang Wei and Jeremy H.M. Wong and Eng Siong Chng and Nancy F. Chen and Hung-yi Lee},
  booktitle = {Proc. Interspeech 2025},
  year      = {2025},
  pages     = {3858--3862},
  doi       = {10.21437/Interspeech.2025-747},
}
```

**Paper 2 — Correlation-based permutation**

> Fabian Ritter-Gutierrez, Yi-Cheng Lin, Jeremy H.M. Wong, Hung-yi Lee,
> Eng Siong Chng, Nancy F. Chen, "A correlation-permutation approach for
> speech-music encoders model merging," *Proc. IEEE Automatic Speech
> Recognition and Understanding Workshop (ASRU)*, 2025.
> [arXiv:2506.11403](https://arxiv.org/abs/2506.11403)

```bibtex
@inproceedings{rittergutierrez2025correlation,
  title     = {A correlation-permutation approach for speech-music encoders model merging},
  author    = {Fabian Ritter-Gutierrez and Yi-Cheng Lin and Jeremy H.M. Wong and Hung-yi Lee and Eng Siong Chng and Nancy F. Chen},
  booktitle = {Proc. IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
  year      = {2025},
}
```

## What is in here

| Pipeline | Entry point | Docs |
|---|---|---|
| Multi-teacher knowledge distillation (HuBERT + MERT teachers) | `s3prl/run_pretrain.py -u multi_distiller` | [docs/distillation.md](docs/distillation.md) |
| Task arithmetic / linear interpolation of task vectors | `s3prl/merge_two_models.py`, `s3prl/merge_wide_models.py` | [docs/task-arithmetic.md](docs/task-arithmetic.md) |
| Correlation-based permutation merging (Hungarian / ZipIt / Sinkhorn) | `s3prl/prepare_hubert_mert_merging_permutation_style.py` | [docs/permutation-merging.md](docs/permutation-merging.md) |
| Downstream evaluation on 11 speech/music/audio tasks | `s3prl/run_downstream.py` | [docs/evaluation.md](docs/evaluation.md) |
| Statistical significance testing (McNemar, paired bootstrap, BH-FDR) | `s3prl/reports/stat_tests/run_stat_tests*.py` | [docs/statistical-testing.md](docs/statistical-testing.md) |
| Step-by-step paper reproduction | — | [docs/reproducing-task-arithmetic-paper.md](docs/reproducing-task-arithmetic-paper.md), [docs/reproducing-permutation-paper.md](docs/reproducing-permutation-paper.md) |

## Relationship to s3prl

This is a fork of **[s3prl](https://github.com/s3prl/s3prl) v0.4.15** (Apache-2.0).
All credit for the framework, the upstream/downstream abstraction, the SUPERB task
implementations, and the training runners goes to the s3prl authors. This fork
adds:

- a multi-teacher distillation upstream (`s3prl/pretrain/multi_distiller/`,
  `s3prl/upstream/multi_distiller/`),
- music (MARBLE) and environmental-audio downstream tasks,
- the model-merging code (task arithmetic and permutation merging),
- noisy-evaluation support and statistical-testing utilities.

The fork keeps s3prl's Apache-2.0 license (see `LICENSE`). If you use the
downstream benchmark, cite the SUPERB and MARBLE papers as well as the individual
dataset papers.

## Installation

```bash
# 1. Create an environment (Python 3.8+ recommended)
conda create -n merge-ssl python=3.8
conda activate merge-ssl

# 2. Install PyTorch first (pick the build matching your CUDA version)
#    See https://pytorch.org — the code is tested with torch >= 1.8 (torch 2.x works)
pip install torch torchaudio

# 3. Install this repo in editable mode
git clone https://github.com/FabianRitter/speech-ssl-merging.git
cd speech-ssl-merging
pip install -e .
```

Notes:

- Dependencies are listed under `requirements/` (`install.txt` is the core set;
  `all.txt` adds optional extras). `pip install -e .` installs the core set.
- The MERT teacher is loaded from the HuggingFace Hub (`m-a-p/MERT-v0-public`);
  the HuBERT teacher is downloaded through torch.hub on first use. Both are cached
  locally afterwards.
- Music tasks need `nnAudio` (installed via requirements) and GTZAN/NSynth
  resampling relies on `torchaudio`.

## Repository layout

```
.
├── LICENSE                      # Apache-2.0 (inherited from s3prl)
├── requirements/                # pip requirement sets
├── setup.py
└── s3prl/
    ├── run_pretrain.py          # distillation entry point
    ├── run_downstream.py        # downstream train/eval entry point
    ├── get_best_checkpoint.py   # export best pretraining checkpoint
    ├── create_wide_base_model.py# create shared init for wide (1536-d) students
    ├── merge_two_models.py      # task arithmetic merge (standard 768-d students)
    ├── merge_by_addition-both-weighting.py  # task-vector / TIES primitives
    ├── merge_wide_models.py     # task arithmetic merge (wide 1536-d students)
    ├── prepare_hubert_mert_merging_permutation_style.py  # permutation merging
    ├── matching_functions.py    # Hungarian / ZipIt / Sinkhorn matching algorithms
    ├── sinkhorn_e2e.py          # end-to-end differentiable Sinkhorn alignment
    ├── graphs/                  # computational-graph representation of HuBERT-style models
    ├── merging_utils/           # ModelMerge orchestrator, calibration data config, TIES
    ├── pretrain/
    │   ├── distiller/           # single-teacher distillation (DistilHuBERT)
    │   └── multi_distiller/     # multi-teacher distillation configs + expert
    ├── upstream/
    │   └── multi_distiller/     # upstream wrapper for distilled/merged checkpoints
    ├── downstream/              # all downstream tasks (SUPERB + MARBLE + ESC-50)
    ├── cluster_scripts/         # canonical launchers (local, SLURM, PBS/enroot)
    ├── analysis_scripts/        # thesis/paper figure scripts
    └── reports/
        ├── statistical_testing_plan.md
        └── stat_tests/          # run_stat_tests.py, run_stat_tests_ch4_ch5.py
```

## Quickstart

All commands run from the `s3prl/` directory. Set dataset paths first — see
[docs/evaluation.md](docs/evaluation.md#dataset-paths) for the override mechanism.

**1. Distill a student from a teacher** (e.g. MERT + HuBERT multi-teacher):

```bash
python run_pretrain.py -u multi_distiller \
    -g pretrain/multi_distiller/config_model.yaml \
    -n my_distilled_student \
    -o "config.pretrain_expert.datarc.libri_root=$LIBRISPEECH_ROOT"
```

**2. Merge two distilled students via task arithmetic** (speech + music):

```bash
python merge_two_models.py \
    --speech_ckpt result/pretrain/SPEECH_STUDENT/states-220000.ckpt \
    --music_ckpt  result/pretrain/MUSIC_STUDENT/states-220000.ckpt \
    --save_path   result/pretrain/merged_ta_0.9_0.1/learning_by_addition.ckpt \
    --lambda1 0.9 --lambda2 0.1
```

**3. Merge two distilled students via permutation alignment:**

```bash
python prepare_hubert_mert_merging_permutation_style.py \
    -n my_permutation_merge \
    --model_mode distilled \
    --model1_ckpt result/pretrain/SPEECH_STUDENT/states-200000.ckpt \
    --model2_ckpt result/pretrain/MUSIC_STUDENT/states-200000.ckpt \
    --merge_type ff+attn --merge_cnn \
    --merging_algorithm match_tensors_permute \
    --interpolation_weights 0.9 0.8 0.5 0.2 0.1
```

**4. Evaluate any checkpoint on a downstream task** (e.g. keyword spotting):

```bash
python run_downstream.py -m train -n my_eval_ks \
    -u multi_distiller_local -s paper \
    -k result/pretrain/merged_ta_0.9_0.1/learning_by_addition.ckpt \
    -d speech_commands -c downstream/speech_commands/config.yaml
```

`-s paper` is only for **distilled (or merged-distilled) models**: it selects the
student's last layer as the frozen representation, the standard convention for
evaluating distilled encoders. When evaluating a **teacher** (e.g.
`-u hubert_base` or `-u mert_v0_public`), omit `-s` — the default then lets the
downstream head learn a weighted linear combination of all SSL layers, as in
SUPERB.

Or run many tasks in parallel with the launcher:

```bash
bash cluster_scripts/run_downstream_local.sh \
    --checkpoint result/pretrain/merged_ta_0.9_0.1/learning_by_addition.ckpt \
    --gpus 0,1,2,3 \
    --tasks asr,ks,ic,sid,er,singid,vocid,instcls,pitchid,genreid,aec_esc50 \
    --upstream multi_distiller_local --stage train
```

## Checkpoints — coming soon

Pretrained student checkpoints (single-teacher speech and music students, wide
students, and selected merged models) will be released on the HuggingFace Hub;
the link will appear here. Until then, every checkpoint can be reproduced
from scratch with the recipes in
[docs/reproducing-task-arithmetic-paper.md](docs/reproducing-task-arithmetic-paper.md)
and [docs/reproducing-permutation-paper.md](docs/reproducing-permutation-paper.md).

## Optional Google-Sheets logging

The training and evaluation entry points accept a `--json_file` flag pointing to a
Google service-account credential JSON. It **defaults to `None`, which disables all
Google-Sheets logging** — results are then only written to the experiment directory
and TensorBoard logs, which is what you want for local use. The related flags
(`--update_results`, `--current_row*`, `--logfile_row*`, `--sheet_name`,
`--worksheet_name`) are only read when a credential file is given and can be
ignored otherwise.

## Documentation

- [docs/distillation.md](docs/distillation.md) — student architectures, teachers, losses, configs, launching, checkpoint export
- [docs/task-arithmetic.md](docs/task-arithmetic.md) — task vectors, interpolation, TIES, wide models
- [docs/permutation-merging.md](docs/permutation-merging.md) — correlation collection, matching algorithms, graphs, analysis modes
- [docs/evaluation.md](docs/evaluation.md) — the 11-task benchmark, dataset paths, noisy evaluation
- [docs/statistical-testing.md](docs/statistical-testing.md) — McNemar / paired bootstrap / BH-FDR pipeline
- [docs/reproducing-task-arithmetic-paper.md](docs/reproducing-task-arithmetic-paper.md)
- [docs/reproducing-permutation-paper.md](docs/reproducing-permutation-paper.md)

## License

Apache-2.0, inherited from s3prl. See `LICENSE`.
