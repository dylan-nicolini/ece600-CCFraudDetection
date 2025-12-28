#!/usr/bin/env python3
"""
downsample-ieee-sffsd-like.py

Downsample the *normalized* IEEE dataset (S-FFSD-like schema) using 3 strategies:
  1) random_ratio
  2) stratified_loc_type
  3) entity_capped

Input must be the *modified* files you created (NOT raw IEEE train_transaction.csv).

Expected columns (at minimum):
  Time, Source, Target, Amount, Location, Type, Labels

Labels:
  - fraud is assumed to be 1
  - non-fraud is assumed to be 0
  - optionally "unknown" (e.g., 2 or -1) can be kept or dropped via flags

Outputs:
  Creates subfolders under --out-dir, one per strategy, containing:
    - ieee_sffsd_like_train.csv  (downsampled)
    - ieee_sffsd_like_test.csv   (copied unchanged if found/provided)
    - stats.json

Example:
  python data-scripts/downsample-ieee-sffsd-like.py \
    --input-dir antifraud/data/ieee_sffsd_like \
    --out-dir antifraud/data/ieee_sffsd_like_downsampled \
    --neg-per-pos 10 \
    --cap-source 200 \
    --cap-target 200 \
    --seed 42
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd


REQUIRED_COLS = ["Time", "Source", "Target", "Amount", "Location", "Type", "Labels"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, help="Folder containing normalized IEEE S-FFSD-like CSVs.")
    p.add_argument("--out-dir", required=True, help="Output folder for downsampled variants.")

    p.add_argument("--train-file", default="ieee_sffsd_like_train.csv",
                   help="Train CSV filename inside --input-dir.")
    p.add_argument("--test-file", default="ieee_sffsd_like_test.csv",
                   help="Test CSV filename inside --input-dir (copied unchanged if exists).")

    p.add_argument("--strategies", nargs="+",
                   default=["random_ratio", "stratified_loc_type", "entity_capped"],
                   choices=["random_ratio", "stratified_loc_type", "entity_capped"],
                   help="Which strategies to generate.")

    p.add_argument("--fraud-label", type=int, default=1, help="Label value representing fraud.")
    p.add_argument("--nonfraud-label", type=int, default=0, help="Label value representing non-fraud.")
    p.add_argument("--keep-other-labels", action="store_true",
                   help="If set, keep rows with Labels not equal to fraud/nonfraud (e.g., 2, -1). "
                        "If not set, those rows are dropped from TRAIN outputs.")

    p.add_argument("--neg-per-pos", type=int, default=10,
                   help="Target ratio: number of non-fraud rows per fraud row in the downsampled TRAIN.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # entity cap options (used by entity_capped strategy)
    p.add_argument("--cap-source", type=int, default=200,
                   help="Max number of NON-FRAUD rows allowed per Source (entity_capped strategy).")
    p.add_argument("--cap-target", type=int, default=200,
                   help="Max number of NON-FRAUD rows allowed per Target (entity_capped strategy).")

    # stratification options
    p.add_argument("--strata-cols", nargs="+", default=["Location", "Type"],
                   help="Columns to stratify on for stratified_loc_type (default: Location Type).")

    return p.parse_args()


def validate_df(df: pd.DataFrame, path: Path):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}\nFound columns: {list(df.columns)}")


def load_inputs(input_dir: Path, train_file: str, test_file: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    train_path = input_dir / train_file
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")

    train_df = pd.read_csv(train_path)
    train_df = train_df.loc[:, ~train_df.columns.str.contains("Unnamed")]
    validate_df(train_df, train_path)

    test_path = input_dir / test_file
    test_df = None
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        test_df = test_df.loc[:, ~test_df.columns.str.contains("Unnamed")]
        # test may not have Labels, but if it does, fine
    return train_df, test_df


def split_labels(df: pd.DataFrame, fraud_label: int, nonfraud_label: int):
    fraud_df = df[df["Labels"] == fraud_label].copy()
    nonfraud_df = df[df["Labels"] == nonfraud_label].copy()
    other_df = df[(df["Labels"] != fraud_label) & (df["Labels"] != nonfraud_label)].copy()
    return fraud_df, nonfraud_df, other_df


def target_nonfraud_count(fraud_count: int, neg_per_pos: int) -> int:
    return int(fraud_count) * int(neg_per_pos)


def downsample_random(nonfraud_df: pd.DataFrame, target_n: int, seed: int) -> pd.DataFrame:
    if target_n >= len(nonfraud_df):
        return nonfraud_df.copy()
    return nonfraud_df.sample(n=target_n, random_state=seed)


def downsample_stratified(nonfraud_df: pd.DataFrame, target_n: int, seed: int, strata_cols) -> pd.DataFrame:
    """
    Sample non-fraud to target_n while approximately preserving distribution over strata_cols.
    """
    if target_n >= len(nonfraud_df):
        return nonfraud_df.copy()

    # Ensure strata cols exist; if not, fall back to random
    for c in strata_cols:
        if c not in nonfraud_df.columns:
            return downsample_random(nonfraud_df, target_n, seed)

    grouped = nonfraud_df.groupby(strata_cols, dropna=False)
    sizes = grouped.size().reset_index(name="count")

    total = sizes["count"].sum()
    if total == 0:
        return nonfraud_df.head(0).copy()

    # proportional allocation
    sizes["alloc"] = (sizes["count"] / total * target_n).round().astype(int)

    # fix rounding drift
    drift = int(target_n - sizes["alloc"].sum())
    if drift != 0:
        # distribute remaining +/- 1 to largest groups
        sizes = sizes.sort_values("count", ascending=False).reset_index(drop=True)
        i = 0
        step = 1 if drift > 0 else -1
        while drift != 0 and i < len(sizes):
            new_val = sizes.at[i, "alloc"] + step
            if new_val >= 0:
                sizes.at[i, "alloc"] = new_val
                drift -= step
            i = (i + 1) % len(sizes)

    # sample within each stratum
    out_parts = []
    for _, row in sizes.iterrows():
        key = tuple(row[c] for c in strata_cols)
        alloc = int(row["alloc"])
        if alloc <= 0:
            continue
        sub = grouped.get_group(key)
        if alloc >= len(sub):
            out_parts.append(sub)
        else:
            out_parts.append(sub.sample(n=alloc, random_state=seed))

    out = pd.concat(out_parts, axis=0) if out_parts else nonfraud_df.head(0).copy()

    # exact trim if we ended up slightly over due to group saturation
    if len(out) > target_n:
        out = out.sample(n=target_n, random_state=seed)

    return out


def downsample_entity_capped(nonfraud_df: pd.DataFrame, target_n: int, seed: int, cap_source: int, cap_target: int) -> pd.DataFrame:
    """
    Two-pass filter on NON-FRAUD:
      - shuffle
      - accept row if counts[source] < cap_source AND counts[target] < cap_target
    Then if still > target_n, random sample down to target_n.
    """
    if target_n >= len(nonfraud_df):
        # still apply caps? Usually no need; but for consistency, we will apply caps and then return.
        pass

    df = nonfraud_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    src_counts: Dict[str, int] = {}
    tgt_counts: Dict[str, int] = {}

    keep_idx = []
    for i, r in df.iterrows():
        s = r["Source"]
        t = r["Target"]
        sc = src_counts.get(s, 0)
        tc = tgt_counts.get(t, 0)

        if sc < cap_source and tc < cap_target:
            keep_idx.append(i)
            src_counts[s] = sc + 1
            tgt_counts[t] = tc + 1

    capped = df.iloc[keep_idx].copy()

    if target_n >= len(capped):
        return capped

    return capped.sample(n=target_n, random_state=seed)


def build_output(train_fraud: pd.DataFrame,
                 train_nonfraud_sampled: pd.DataFrame,
                 train_other: pd.DataFrame,
                 keep_other: bool,
                 seed: int) -> pd.DataFrame:
    parts = [train_fraud, train_nonfraud_sampled]
    if keep_other and len(train_other) > 0:
        parts.append(train_other)

    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def write_variant(out_dir: Path,
                  variant_name: str,
                  train_out: pd.DataFrame,
                  test_df: Optional[pd.DataFrame],
                  stats: dict):
    vdir = out_dir / variant_name
    vdir.mkdir(parents=True, exist_ok=True)

    train_path = vdir / "ieee_sffsd_like_train.csv"
    train_out.to_csv(train_path, index=False)

    if test_df is not None:
        test_path = vdir / "ieee_sffsd_like_test.csv"
        test_df.to_csv(test_path, index=False)

    stats_path = vdir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"[OK] Wrote {variant_name}: {train_path}  (rows={len(train_out)})")


def compute_stats(original_train: pd.DataFrame,
                  out_train: pd.DataFrame,
                  fraud_label: int,
                  nonfraud_label: int) -> dict:
    def counts(df):
        return {
            "rows": int(len(df)),
            "fraud": int((df["Labels"] == fraud_label).sum()) if "Labels" in df.columns else None,
            "nonfraud": int((df["Labels"] == nonfraud_label).sum()) if "Labels" in df.columns else None,
            "other": int(((df["Labels"] != fraud_label) & (df["Labels"] != nonfraud_label)).sum()) if "Labels" in df.columns else None,
        }

    o = counts(original_train)
    n = counts(out_train)

    ratio = None
    if n["fraud"] and n["fraud"] > 0 and n["nonfraud"] is not None:
        ratio = float(n["nonfraud"]) / float(n["fraud"])

    return {
        "original": o,
        "downsampled": n,
        "nonfraud_per_fraud": ratio,
        "columns": list(out_train.columns),
    }


def main():
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_inputs(input_dir, args.train_file, args.test_file)

    fraud_df, nonfraud_df, other_df = split_labels(train_df, args.fraud_label, args.nonfraud_label)

    if len(fraud_df) == 0:
        raise ValueError("No fraud rows found in TRAIN. Check --fraud-label or your Labels column.")

    target_n = target_nonfraud_count(len(fraud_df), args.neg_per_pos)

    print("=== INPUT SUMMARY ===")
    print(f"Train rows: {len(train_df)}")
    print(f"Fraud rows (Labels={args.fraud_label}): {len(fraud_df)}")
    print(f"Non-fraud rows (Labels={args.nonfraud_label}): {len(nonfraud_df)}")
    print(f"Other-label rows: {len(other_df)} {'(will KEEP)' if args.keep_other_labels else '(will DROP)'}")
    print(f"Target non-fraud count: {target_n}  (neg_per_pos={args.neg_per_pos})")
    print("Strategies:", args.strategies)
    print("=====================")

    for strat in args.strategies:
        if strat == "random_ratio":
            sampled_nonfraud = downsample_random(nonfraud_df, target_n, args.seed)

        elif strat == "stratified_loc_type":
            sampled_nonfraud = downsample_stratified(nonfraud_df, target_n, args.seed, args.strata_cols)

        elif strat == "entity_capped":
            sampled_nonfraud = downsample_entity_capped(
                nonfraud_df, target_n, args.seed, args.cap_source, args.cap_target
            )

        else:
            raise ValueError(f"Unknown strategy: {strat}")

        train_out = build_output(
            train_fraud=fraud_df,
            train_nonfraud_sampled=sampled_nonfraud,
            train_other=other_df,
            keep_other=args.keep_other_labels,
            seed=args.seed
        )

        stats = {
            "strategy": strat,
            "params": {
                "neg_per_pos": args.neg_per_pos,
                "seed": args.seed,
                "cap_source": args.cap_source,
                "cap_target": args.cap_target,
                "strata_cols": args.strata_cols,
                "keep_other_labels": args.keep_other_labels,
                "fraud_label": args.fraud_label,
                "nonfraud_label": args.nonfraud_label,
            },
            "summary": compute_stats(train_df, train_out, args.fraud_label, args.nonfraud_label),
        }

        write_variant(out_dir, strat, train_out, test_df, stats)

    print("\nDone.")
    print(f"Output root: {out_dir}")


if __name__ == "__main__":
    main()
