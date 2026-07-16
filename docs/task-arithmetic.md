# Task Arithmetic (Interpolation-Based Merging)

Merge two single-domain distilled students — one speech (HuBERT teacher), one
music (MERT teacher) — into a single model by weighted addition of **task
vectors**, without any retraining.

## Concept

A task vector is the difference between a trained model and its initialization:

```
TV_speech = theta_speech − theta_base
TV_music  = theta_music  − theta_base
theta_merged = theta_base + λ1 · TV_speech + λ2 · TV_music
```

For this to be meaningful, **both students must start from the exact same
initialization** (`theta_base`, same seed). In this codebase:

- **Standard (768-d) students**: both are initialized from HuBERT Base
  (conv + projection + first encoder layers). The merge script reconstructs
  `theta_base` on the fly by re-running the same HuBERT-based initialization.
- **Wide (1536-d) students**: the teacher init is impossible (dimension
  mismatch), so all runs share a saved random base
  (`result/pretrain/theta_base_wide/theta_base_wide.pt`, created once by
  `create_wide_base_model.py`). The merge script loads that file directly.

The papers sweep λ1/λ2 over `(0.9, 0.1)`, `(0.8, 0.2)`, `(0.5, 0.5)`,
`(0.2, 0.8)`, `(0.1, 0.9)` with λ1 + λ2 = 1 (linear interpolation of task
vectors).

## Scripts

| Script | Role |
|--------|------|
| `merge_two_models.py` | Main entry point for standard 768-d students. Computes task vectors against the reconstructed HuBERT-based init, combines them, and writes a downstream-ready checkpoint. |
| `merge_by_addition-both-weighting.py` | Library used by the wrapper: `load_checkpoint`, `compute_task_vector`, `apply_task_vector`, `TIESMerging`, checkpoint assembly. Not usually called directly. |
| `merge_wide_models.py` | Entry point for wide 1536-d students; loads `theta_base_wide.pt` instead of reconstructing the base. |

## Merging standard (768-d) students

```bash
python merge_two_models.py \
    --speech_ckpt result/pretrain/SPEECH_STUDENT/states-220000.ckpt \
    --music_ckpt  result/pretrain/MUSIC_STUDENT/states-220000.ckpt \
    --save_path   result/pretrain/task_vector_speech_0.9_music_0.1/learning_by_addition.ckpt \
    --lambda1 0.9 --lambda2 0.1
```

Arguments:

| Flag | Meaning | Default |
|------|---------|---------|
| `--speech_ckpt`, `--music_ckpt` | endpoint checkpoints (must share init) | required |
| `--save_path` | output checkpoint path (directory is created) | required |
| `--lambda1`, `--lambda2` | weights for the speech / music task vectors | required |
| `--base_config` | model config used to rebuild `theta_base` and to stamp `teacher_names` into the merged checkpoint | `pretrain/multi_distiller/config_model.yaml` |
| `--seed` | RNG seed for the base reconstruction | 1337 |
| `--use_ties` | enable TIES merging (see below) | off |
| `--k` | TIES top-k fraction of task-vector entries to keep | 0.3 |
| `--enable_sign_interference` | resolve sign conflicts between task vectors | off |
| `--sign_as_in_ties` | use the TIES sign-election rule | off |

The wrapper preserves `teacher_names` from the base config so the merged
checkpoint's state-dict keys match the architecture that
`multi_distiller_local` rebuilds at load time. It also converts `distiller`
config keys to `multi_distiller` automatically, so you can mix a
`distiller`-trained speech student with a `multi_distiller`-trained music
student.

## Merging wide (1536-d) students

```bash
# All 5 weight combinations for the L1 (L_KD) pair:
python merge_wide_models.py --loss_type l1 --speech_step 200000 --music_step 200000

# All 5 combinations for the Barlow (L_CL) pair:
python merge_wide_models.py --loss_type barlow --speech_step 200000 --music_step 200000

# A single combination with explicit paths:
python merge_wide_models.py \
    --speech_ckpt result/pretrain/hubert_l1_wide/states-200000.ckpt \
    --music_ckpt  result/pretrain/mert_l1_wide/states-200000.ckpt \
    --base_state  result/pretrain/theta_base_wide/theta_base_wide.pt \
    --lambda1 0.5 --lambda2 0.5
```

- `--loss_type {l1,barlow}` selects the default endpoint checkpoint pair; when
  omitted, give `--speech_ckpt`/`--music_ckpt` explicitly.
- When `--lambda1/--lambda2` are omitted, all five weight combinations are
  produced in one run.
- Outputs land in
  `result/pretrain/task_vector_wide_<loss>_hubert_weight_<λ1>_mert_weight_<λ2>/learning_by_addition.ckpt`.
- TIES flags (`--use_ties`, `--k`, `--enable_sign_interference`,
  `--sign_as_in_ties`) are the same as for the standard script.

## TIES option

With `--use_ties`, task vectors are processed with the TIES-Merging recipe
(Yadav et al., 2023) before addition:

1. **Trim** — keep only the top-k fraction (`--k`, default 0.3) of each task
   vector by magnitude; zero the rest.
2. **Elect sign** (with `--enable_sign_interference`) — where the two task
   vectors disagree in sign, resolve the conflict; `--sign_as_in_ties` follows
   the original TIES majority-sign rule.
3. **Merge** — weighted addition of the processed vectors onto the base.

Plain interpolation (no TIES) is the default and is what the main paper tables
use; TIES results are reported as an ablation (see paper).

## Output format and downstream evaluation

Merged checkpoints contain `Distiller` (merged weights), `Config`
(a `multi_distiller` config matching the architecture, with
`teacher_names` preserved), `Optimizer: None`, `Step: 0`. They are drop-in
compatible with the `multi_distiller_local` upstream, so evaluation is identical
to any distilled model:

```bash
bash cluster_scripts/run_downstream_local.sh \
    --checkpoint result/pretrain/task_vector_speech_0.9_music_0.1/learning_by_addition.ckpt \
    --gpus 0,1,2,3 \
    --tasks asr,ks,ic,sid,er,singid,vocid,instcls,pitchid,genreid,aec_esc50 \
    --upstream multi_distiller_local --stage train
```

For a merged sweep, `cluster_scripts/run_merge_and_downstream_enroot_2026.sh`
shows the full merge-then-evaluate pipeline as a single PBS/enroot job (merge at
several λ, then train+evaluate each merged model on the selected tasks). Use it
as a template even if your cluster is not PBS-based.

See [evaluation.md](evaluation.md) for the task list and dataset setup, and
[reproducing-task-arithmetic-paper.md](reproducing-task-arithmetic-paper.md) for
the exact experiment grid.
