#!/usr/bin/env python3
"""
row_calculator.py — Compute Google Sheet row for any experiment.

Usage:
    python row_calculator.py 2L_LKD G1 0.9_0.1
    python row_calculator.py --all          # print all Phase 1 rows
    python row_calculator.py --pair 3L_LCL  # print all rows for a pair
"""
import argparse
import sys

BASE_ROW = 100
LOGFILE_OFFSET = 200

PAIR_OFFSETS = {
    "2L_LKD": 0,
    "2L_wide_LCL": 1,
    "3L_LKD": 2,
    "3L_LCL": 3,
}

METHOD_OFFSETS = {
    "G1": 0, "G2": 1, "G3": 2, "G4": 3, "G5": 4,
}

INTERP_INDICES = {
    "0.9_0.1": 0, "0.8_0.2": 1, "0.5_0.5": 2, "0.2_0.8": 3, "0.1_0.9": 4,
}

INTERP_LABELS = list(INTERP_INDICES.keys())


def compute_row(pair_id, method_id, lambda_label):
    row = BASE_ROW + (PAIR_OFFSETS[pair_id] * 25) + (METHOD_OFFSETS[method_id] * 5) + INTERP_INDICES[lambda_label]
    return row, row + LOGFILE_OFFSET


def main():
    parser = argparse.ArgumentParser(description="Compute Google Sheet row for experiments")
    parser.add_argument("pair_id", nargs="?", help="Model pair ID")
    parser.add_argument("method_id", nargs="?", help="Permutation method ID")
    parser.add_argument("lambda_label", nargs="?", help="Interpolation label (e.g., 0.9_0.1)")
    parser.add_argument("--all", action="store_true", help="Print all Phase 1 rows")
    parser.add_argument("--pair", help="Print all rows for a specific pair")
    parser.add_argument("--method", default="G1", help="Method to use with --pair (default: G1)")
    args = parser.parse_args()

    if args.all:
        print(f"{'Pair':<16} {'Method':<6} {'Lambda':<8} {'Row':<6} {'Logfile'}")
        print("-" * 50)
        for pair in PAIR_OFFSETS:
            for method in METHOD_OFFSETS:
                for lam in INTERP_LABELS:
                    row, logrow = compute_row(pair, method, lam)
                    print(f"{pair:<16} {method:<6} {lam:<8} {row:<6} {logrow}")
        return

    if args.pair:
        pair = args.pair
        method = args.method
        print(f"Rows for {pair} / {method}:")
        print(f"{'Lambda':<10} {'Row':<6} {'Logfile'}")
        for lam in INTERP_LABELS:
            row, logrow = compute_row(pair, method, lam)
            print(f"{lam:<10} {row:<6} {logrow}")
        return

    if not all([args.pair_id, args.method_id, args.lambda_label]):
        parser.print_help()
        sys.exit(1)

    row, logrow = compute_row(args.pair_id, args.method_id, args.lambda_label)
    print(f"pair={args.pair_id}  method={args.method_id}  lambda={args.lambda_label}")
    print(f"current_row={row}  logfile_row={logrow}")


if __name__ == "__main__":
    main()
