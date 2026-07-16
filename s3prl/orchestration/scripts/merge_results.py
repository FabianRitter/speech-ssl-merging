#!/usr/bin/env python3
"""
merge_results.py — Combine NTU and NSCC result TSVs into unified tables.

Used by NSCC-ANALYST to produce combined results for analysis.

Usage:
    python merge_results.py --results_dir orchestration/results/raw/ \
                            --output orchestration/results/reports/combined_results.tsv
"""
import os
import glob
import argparse
import csv
from collections import defaultdict

# FBANK and BEST baselines for Score computation
# Source: CLAUDE_NSCC_ANALYST.md — Chapter 4 reference values
# BEST = max(HuBERT, MERT) per task; for ASR BEST = min (WER, lower is better)
# None entries = not available in Chapter 4 table; those tasks are skipped in score
FBANK_BASELINES = {
    "asr": 91.54,   # WER — higher FBANK = worse = easier to normalise
    "ks": 35.39,
    "ic": 2.0,
    "sid": 9.09,
    "er": 10.0,
    "singid": 0.78,
    "vocid": None,  # not in Chapter 4 table
    "instcls": 5.0,
    "pitchid": None,  # not in Chapter 4 table
    "genreid": 10.0,
    "aec_esc50": None,  # not in Chapter 4 table
}

BEST_BASELINES = {
    "asr": 7.84,    # HuBERT WER (lower is better)
    "ks": 96.3,     # HuBERT
    "ic": 98.34,    # HuBERT
    "sid": 81.42,   # HuBERT
    "er": 64.97,    # MERT
    "singid": 70.69,  # MERT
    "vocid": 91.26,   # MERT
    "instcls": 73.97, # MERT
    "pitchid": 70.04, # HuBERT (MERT not available)
    "genreid": 63.45, # HuBERT (MERT not available)
    "aec_esc50": 70.2, # HuBERT (MERT not available)
}

SPEECH_TASKS = ["asr", "ks", "ic", "sid", "er"]
MUSIC_TASKS = ["singid", "vocid", "instcls", "pitchid", "genreid", "aec_esc50"]
ALL_TASKS = SPEECH_TASKS + MUSIC_TASKS
WER_TASKS = ["asr"]


def read_tsv(filepath):
    results = {}
    with open(filepath, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            results[row["task"]] = float(row["value"])
    return results


def compute_score(results, task_set):
    # Only score tasks that have both a result and a FBANK baseline
    scorable = [t for t in task_set if t in results and FBANK_BASELINES.get(t) is not None]
    n = len(scorable)
    if n == 0:
        return None
    total = 0.0
    for task in scorable:
        s = results[task]
        fbank = FBANK_BASELINES[task]
        best = BEST_BASELINES[task]
        if task in WER_TASKS:
            total += (fbank - s) / (fbank - best) if fbank != best else 1.0
        else:
            total += (s - fbank) / (best - fbank) if best != fbank else 1.0
    return (1000.0 / n) * total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="orchestration/results/raw/")
    parser.add_argument("--output", default="orchestration/results/reports/combined_results.tsv")
    args = parser.parse_args()

    ntu_files = glob.glob(os.path.join(args.results_dir, "*_ntu.tsv"))
    experiments = defaultdict(dict)

    for f in ntu_files:
        exp_id = os.path.basename(f).replace("_ntu.tsv", "")
        experiments[exp_id]["ntu"] = read_tsv(f)

    for exp_id in list(experiments.keys()):
        nscc_path = os.path.join(args.results_dir, f"{exp_id}_nscc.tsv")
        if os.path.exists(nscc_path):
            experiments[exp_id]["nscc"] = read_tsv(nscc_path)

    # Also find NSCC files without matching NTU
    nscc_files = glob.glob(os.path.join(args.results_dir, "*_nscc.tsv"))
    for f in nscc_files:
        exp_id = os.path.basename(f).replace("_nscc.tsv", "")
        if exp_id not in experiments:
            experiments[exp_id]["nscc"] = read_tsv(f)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        header = ["experiment"] + ALL_TASKS + ["speech_score", "music_score", "avg_score"]
        f.write("\t".join(header) + "\n")

        for exp_id in sorted(experiments.keys()):
            data = experiments[exp_id]
            combined = {}
            if "ntu" in data:
                combined.update(data["ntu"])
            if "nscc" in data:
                combined.update(data["nscc"])

            row = [exp_id]
            for task in ALL_TASKS:
                row.append(f"{combined[task]:.2f}" if task in combined else "—")

            for task_set in [SPEECH_TASKS, MUSIC_TASKS, ALL_TASKS]:
                score = compute_score(combined, task_set)
                row.append(f"{score:.2f}" if score else "—")

            f.write("\t".join(row) + "\n")

    print(f"Written: {args.output}")
    complete = sum(1 for d in experiments.values() if "ntu" in d and "nscc" in d)
    print(f"Total: {len(experiments)}, Complete (both clusters): {complete}")


if __name__ == "__main__":
    main()
