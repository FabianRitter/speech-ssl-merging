# Downstream Evaluation

Every model — distilled, task-arithmetic-merged, or permutation-merged — is
evaluated the same way: the encoder is **frozen**, features are extracted, and a
lightweight downstream head is trained per task (standard s3prl protocol). For
distilled/merged students the feature is the **last hidden layer** of the
student, selected with `-s paper` (DistilHuBERT convention).

## The 11-task benchmark

| # | Alias | Downstream name | Dataset | Domain | Benchmark | Metric | Config | Cross-val |
|---|-------|-----------------|---------|--------|-----------|--------|--------|-----------|
| 1 | `asr` | `asr` | LibriSpeech (test-clean) | Speech | SUPERB | WER ↓ | `downstream/asr/config.yaml` | no |
| 2 | `ks` | `speech_commands` | Speech Commands v0.01 | Speech | SUPERB | Acc ↑ | `downstream/speech_commands/config.yaml` | no |
| 3 | `ic` | `fluent_commands` | Fluent Speech Commands | Speech | SUPERB | Acc ↑ | `downstream/fluent_commands/config.yaml` | no |
| 4 | `sid` | `voxceleb1` | VoxCeleb1 | Speech | SUPERB | Acc ↑ | `downstream/voxceleb1/config.yaml` | no |
| 5 | `er` | `emotion` | IEMOCAP | Speech | SUPERB | Acc ↑ | `downstream/emotion/config.yaml` | 5-fold |
| 6 | `singid` | `vocalset_singer_id` | VocalSet | Music | MARBLE | Acc ↑ | `downstream/vocalset_singer_id/config.yaml` | no |
| 7 | `vocid` | `vocalset_technique_id` | VocalSet | Music | MARBLE | Acc ↑ | `downstream/vocalset_technique_id/config.yaml` | no |
| 8 | `instcls` | `instrument_nsynth` | NSynth | Music | MARBLE | Acc ↑ | `downstream/instrument_nsynth/config.yaml` | no |
| 9 | `pitchid` | `pitch_nsynth` | NSynth | Music | MARBLE | Acc ↑ | `downstream/pitch_nsynth/config.yaml` | no |
| 10 | `genreid` | `genre_gtzan` | GTZAN | Music | MARBLE | Acc ↑ | `downstream/genre_gtzan/config.yaml` | no |
| 11 | `aec_esc50` | `aec_esc50` | ESC-50 | Audio | — | Acc ↑ | `downstream/aec_esc50/config.yaml` | 5-fold |

Task specifics:

- **ASR** — model selection on `dev-clean` (`dev-clean-best.ckpt`), evaluation on
  `test-clean`; batch sizes and bucket-file paths are usually set via `-o`
  overrides.
- **SID** — caps utterances at `max_timestep: 128000`.
- **ER / ESC-50** — 5-fold cross-validation; final number is the average across
  folds. ESC-50 uses SpecAugment + Mixup.
- **GenreID (GTZAN)** — uses the `valid` split (not `dev`); the evaluation
  checkpoint is `valid-best.ckpt`. Audio is resampled to 24 kHz. 10-class genre
  set, 10k training steps.
- **InstCls / PitchID (NSynth)** — 200k training steps.

### Additional MARBLE task ports

The following MARBLE tasks are also ported under `downstream/` and can be used
with the same protocol, but are **not** part of the 11-task benchmark reported
in the papers and do not ship reference configs: `gs_key` (GiantSteps key
detection), `mer_emomusic` (EmoMusic emotion regression), `genre_mtg`,
`instrument_mtg`, `mer_mtg_mood` (MTG-Jamendo genre/instrument/mood tagging),
`mt_mtg`, and `mt_manga_tag_a_tune` (MagnaTagATune tagging). Treat them as
experimental: supply your own `config.yaml` modeled on a neighboring task.

## Dataset paths

Configs ship with **placeholder dataset paths**. Point them at your data in one
of two ways:

1. **Edit the config** — each task's `config.yaml` has the dataset root under
   `downstream_expert.datarc` (field name varies per task: `libri_root`,
   `file_path`, `speech_commands_root`, ...).
2. **Override on the command line** — anything in the config can be overridden
   with `-o`, using `,,` to separate multiple overrides:

   ```bash
   -o "config.downstream_expert.datarc.file_path=$DATA_ROOT/VocalSet"
   ```

Several tasks also carry `config_nscc.yaml` and `config_singularity.yaml`
variants. These are **cluster-profile examples** (the same task configured for a
PBS/enroot container site and a Singularity site respectively) — useful as
templates when your data lives under a container mount, but `config.yaml` is the
canonical one.

Dataset download/preparation follows stock s3prl: see `downstream/README.md`
for per-task instructions (SUPERB tasks) and the MARBLE/dataset homepages for
GTZAN, NSynth, VocalSet, and ESC-50. Expected layout:

| Dataset | Suggested location | Used by |
|---------|--------------------|---------|
| LibriSpeech | `$DATA_ROOT/LibriSpeech` | pretraining, ASR |
| Speech Commands v0.01 (+ test set) | `$DATA_ROOT/speech_commands_v0.01` | KS |
| Fluent Speech Commands | `$DATA_ROOT/fluent_speech_commands_dataset` | IC |
| VoxCeleb1 | `$DATA_ROOT/VoxCeleb1` | SID |
| IEMOCAP | `$DATA_ROOT/IEMOCAP_full_release` | ER |
| VocalSet | `$DATA_ROOT/VocalSet` | SingID, VocID |
| NSynth | `$DATA_ROOT/NSynth` | InstCls, PitchID |
| GTZAN | `$DATA_ROOT/GTZAN` | GenreID |
| ESC-50 | `$DATA_ROOT/esc50` | ESC-50 |

## Running a single task

```bash
python run_downstream.py \
    -m train \
    -n MY_MODEL_ks \
    -u multi_distiller_local -s paper \
    -k result/pretrain/MY_MODEL/states-200000.ckpt \
    -d speech_commands \
    -c downstream/speech_commands/config.yaml \
    -o "config.downstream_expert.datarc.speech_commands_root=$DATA_ROOT/speech_commands_v0.01,,config.downstream_expert.datarc.speech_commands_test_root=$DATA_ROOT/speech_commands_test_set_v0.01"
```

Then evaluate the best checkpoint:

```bash
python run_downstream.py -m evaluate \
    -e result/downstream/MY_MODEL_ks/dev-best.ckpt
```

Key flags:

- `-u multi_distiller_local` — upstream wrapper that rebuilds the student from a
  local checkpoint (`-k`). Built-in upstreams (`hubert`, `mert_v0_public`, ...)
  need no `-k`.
- `-s paper` — only for distilled (or merged-distilled) models: use the
  student's last hidden layer as the frozen representation (the standard
  convention for evaluating distilled encoders). When evaluating a teacher,
  omit `-s` — the default lets the downstream head learn a weighted linear
  combination of all SSL layers, as in SUPERB.
- `-m train` / `-m evaluate` — training vs test-set evaluation; `-t` selects the
  evaluation split (e.g. `-t test-clean` for ASR).
- Cross-val tasks: run once per fold with
  `-o "config.downstream_expert.datarc.test_fold=foldN"` and average.

## Running many tasks in parallel

`cluster_scripts/run_downstream_local.sh` dispatches one task per GPU (queue
workers), applies the correct per-task config, overrides, evaluation checkpoint
name, and fold handling, then trains **and** evaluates:

```bash
bash cluster_scripts/run_downstream_local.sh \
    --checkpoint result/pretrain/MY_MODEL/states-200000.ckpt \
    --gpus 0,1,2,3,4,5,6,7,0,1,2 \
    --tasks asr,ks,ic,sid,er,singid,vocid,instcls,pitchid,genreid,aec_esc50 \
    --upstream multi_distiller_local \
    --stage train
```

- `--stage train` trains then evaluates; `--stage evaluating` only evaluates
  existing downstream checkpoints.
- GPUs and tasks are zipped in order; repeat GPU ids to queue several tasks on
  one device.

Cluster templates (edit headers/paths for your site):

- `cluster_scripts/run_downstream_sbatch.sh` — SLURM, one task per job
- `cluster_scripts/run_downstream_enroot_2026.sh` — PBS + enroot container
- `cluster_scripts/run_inside_container_downstream.sh` — the per-task script the
  container jobs execute
- `cluster_scripts/merging_packages_singularity.sh` — extra pip packages needed
  inside a bare Singularity/enroot image

## Where results land

For each task, an experiment directory is created (default
`result/downstream/<expname>/`, the launcher scripts use
`<expname>_<task>_paper_method[_foldN]` naming). It contains:

- `dev-best.ckpt` (or `valid-best.ckpt` for GTZAN, `dev-clean-best.ckpt` for
  ASR) — best downstream head
- `events.*` — TensorBoard logs (dev/test metrics)
- `test_predict.txt` + `test_truth.txt` — per-sample predictions and labels
  written at evaluation time (IC writes `test_predict.csv`; GTZAN writes
  `valid_*`; ER writes fold-suffixed `test_fold{N}_*`); ASR writes
  `test-clean-noLM-hyp.ark` / `test-clean-noLM-ref.ark` transcripts
- copies of the config and args

The final metric is printed at the end of `-m evaluate` and stored in the
TensorBoard log. The per-sample prediction files are the input to the
statistical tests ([statistical-testing.md](statistical-testing.md)).

## Noisy evaluation (optional)

Robustness experiments evaluate clean-trained downstream heads on **noisy**
test audio. Distortions are injected by `downstream/distortions.py`
(`DistortionFactory`): additive real noise (e.g. CHiME backgrounds) mixed at a
random SNR in 10–20 dB, or Gaussian noise (`g`).

Enable per task via config overrides at evaluation time:

```bash
-o "config.downstream_expert.datarc.distortion_mode=single,,config.downstream_expert.datarc.distortion_types=['chime'],,config.downstream_expert.datarc.distortion_config=./downstream/distortion_config.yaml"
```

`downstream/distortion_config.yaml` maps noise-type names to directories of
`.wav` noise files — point it at your local CHiME (or other) noise folder. All
11 benchmark tasks accept these keys in their `datarc`. Noisy-run outputs
(including the per-sample prediction files) are written to
an `evaluation/<noise_type>/` subdirectory of the task's experiment dir, so
clean and noisy runs can be compared sample-by-sample.
