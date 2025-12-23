#!/usr/bin/env python3
"""
Convert IEEE-CIS transaction files into an S-FFSD-like edge list with 7 columns:
Time, Source, Target, Amount, Location, Type, Labels

- Uses ONLY train_transaction.csv and test_transaction.csv (ignores identity files).
- Train keeps Labels from isFraud.
- Test Labels are set to a placeholder (default: -1).
- Source is a composite of card fields by default.
- Target defaults to R_emaildomain with fallback to P_emaildomain then UNK.
- Produces 1 output row per input row (does NOT reduce row count).

Output columns match S-FFSD schema:
  Time, Source, Target, Amount, Location, Type, Labels

Usage example:
python normalize-ieee-dataset-to-sfssd.py \
  --input-dir /path/to/ieee \
  --out-dir /path/to/out \
  --source card_composite \
  --target r_emaildomain \
  --label-placeholder -1
"""

import argparse
import os
import sys
from typing import Optional, List

import pandas as pd


SFFSD_COLS = ["Time", "Source", "Target", "Amount", "Location", "Type", "Labels"]


def _read_csv(path: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def _to_str_series(s: pd.Series, na_token: str = "UNK") -> pd.Series:
    """Convert a series to strings, replacing NaN/None with na_token."""
    s2 = s.astype("object").where(pd.notna(s), None)
    return s2.apply(lambda x: na_token if x is None else str(x))


def make_source(df: pd.DataFrame, source_mode: str, na_token: str = "UNK") -> pd.Series:
    if source_mode == "card1":
        if "card1" not in df.columns:
            raise KeyError("card1 not found in input data")
        return "card:" + _to_str_series(df["card1"], na_token=na_token)

    if source_mode == "card_composite":
        needed = ["card1", "card2", "card3", "card4", "card5", "card6"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise KeyError(f"Missing card columns for card_composite: {missing}")

        parts = [_to_str_series(df[c], na_token=na_token) for c in needed]
        comp = parts[0]
        for p in parts[1:]:
            comp = comp + "|" + p
        return "card:" + comp

    raise ValueError(f"Unknown source_mode: {source_mode}")


def make_target(df: pd.DataFrame, target_mode: str, na_token: str = "UNK") -> pd.Series:
    """Create the Target node id series with stable prefixes."""
    def prefixed(col: str, prefix: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series([f"{prefix}{na_token}"] * len(df), index=df.index)
        return prefix + _to_str_series(df[col], na_token=na_token)

    if target_mode == "r_emaildomain":
        # R_emaildomain fallback to P_emaildomain then UNK
        r = prefixed("R_emaildomain", "email:")
        if "P_emaildomain" in df.columns:
            p = prefixed("P_emaildomain", "email:")
            r = r.where(r != f"email:{na_token}", p)
        return r

    if target_mode == "p_emaildomain":
        return prefixed("P_emaildomain", "email:")

    if target_mode == "addr1":
        return prefixed("addr1", "addr:")

    if target_mode == "product":
        return prefixed("ProductCD", "product:")

    raise ValueError(f"Unknown target_mode: {target_mode}")


def make_location(df: pd.DataFrame, na_token: str = "UNK") -> pd.Series:
    # Closest comparable field to S-FFSD Location
    if "addr1" not in df.columns:
        return pd.Series([na_token] * len(df), index=df.index)
    return _to_str_series(df["addr1"], na_token=na_token)


def make_type(df: pd.DataFrame, na_token: str = "UNK") -> pd.Series:
    # Closest comparable field to S-FFSD Type
    if "ProductCD" not in df.columns:
        return pd.Series([na_token] * len(df), index=df.index)
    return _to_str_series(df["ProductCD"], na_token=na_token)


def convert_ieee_transactions(
    df: pd.DataFrame,
    is_train: bool,
    source_mode: str,
    target_mode: str,
    label_placeholder: int,
    na_token: str = "UNK",
) -> pd.DataFrame:
    """Convert raw IEEE transaction df into S-FFSD-like 7-column df."""
    if "TransactionDT" not in df.columns:
        raise KeyError("TransactionDT not found in input data")
    if "TransactionAmt" not in df.columns:
        raise KeyError("TransactionAmt not found in input data")

    out = pd.DataFrame(index=df.index)

    out["Time"] = df["TransactionDT"]
    out["Amount"] = df["TransactionAmt"]

    out["Source"] = make_source(df, source_mode=source_mode, na_token=na_token)
    out["Target"] = make_target(df, target_mode=target_mode, na_token=na_token)

    out["Location"] = make_location(df, na_token=na_token)
    out["Type"] = make_type(df, na_token=na_token)

    if is_train:
        if "isFraud" not in df.columns:
            raise KeyError("isFraud not found in train_transaction.csv")
        out["Labels"] = df["isFraud"].astype("int64")
    else:
        out["Labels"] = int(label_placeholder)

    out = out[SFFSD_COLS]
    return out


def print_summary(name: str, out_df: pd.DataFrame, is_train: bool):
    print(f"\n== {name} ==")
    print(f"Rows: {len(out_df):,}")
    print(f"Columns: {len(out_df.columns)} -> {list(out_df.columns)}")

    if is_train:
        labels = out_df["Labels"]
        fraud_rate = float((labels == 1).mean()) if len(labels) else 0.0
        print(f"Fraud rate (Labels==1): {fraud_rate:.6f}")

    for c in ["Source", "Target", "Location", "Type"]:
        if c in ["Source", "Target"]:
            unk_rate = float(out_df[c].astype(str).str.endswith("UNK").mean()) if len(out_df) else 0.0
        else:
            unk_rate = float((out_df[c] == "UNK").mean()) if len(out_df) else 0.0
        print(f"{c} UNK-rate: {unk_rate:.6f}")


def resolve_usecols(path: str, desired: List[str]) -> List[str]:
    """Return only columns that exist in the CSV header (safe for different versions)."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    return [c for c in desired if c in header]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Directory containing train_transaction.csv and test_transaction.csv")
    ap.add_argument("--out-dir", required=True, help="Output directory for S-FFSD-like CSVs")

    ap.add_argument("--train-file", default="train_transaction.csv")
    ap.add_argument("--test-file", default="test_transaction.csv")

    ap.add_argument("--source", default="card_composite", choices=["card_composite", "card1"])
    ap.add_argument("--target", default="r_emaildomain", choices=["r_emaildomain", "p_emaildomain", "addr1", "product"])

    ap.add_argument("--label-placeholder", type=int, default=-1, help="Label value to use for test file (unlabeled)")
    ap.add_argument("--na-token", default="UNK", help="Token for missing categorical values")

    ap.add_argument("--out-train-name", default="ieee_sffsd_like_train.csv")
    ap.add_argument("--out-test-name", default="ieee_sffsd_like_test.csv")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_path = os.path.join(args.input_dir, args.train_file)
    test_path = os.path.join(args.input_dir, args.test_file)

    desired_train_cols = [
        "TransactionDT", "TransactionAmt", "ProductCD",
        "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2", "P_emaildomain", "R_emaildomain",
        "isFraud"
    ]
    desired_test_cols = [c for c in desired_train_cols if c != "isFraud"]

    usecols_train = resolve_usecols(train_path, desired_train_cols)
    usecols_test = resolve_usecols(test_path, desired_test_cols)

    train_df = _read_csv(train_path, usecols=usecols_train)
    test_df = _read_csv(test_path, usecols=usecols_test)

    out_train = convert_ieee_transactions(
        train_df,
        is_train=True,
        source_mode=args.source,
        target_mode=args.target,
        label_placeholder=args.label_placeholder,
        na_token=args.na_token,
    )
    out_test = convert_ieee_transactions(
        test_df,
        is_train=False,
        source_mode=args.source,
        target_mode=args.target,
        label_placeholder=args.label_placeholder,
        na_token=args.na_token,
    )

    out_train_path = os.path.join(args.out_dir, args.out_train_name)
    out_test_path = os.path.join(args.out_dir, args.out_test_name)

    out_train.to_csv(out_train_path, index=False)
    out_test.to_csv(out_test_path, index=False)

    print_summary("TRAIN (S-FFSD-like)", out_train, is_train=True)
    print(f"Saved: {out_train_path}")

    print_summary("TEST (S-FFSD-like)", out_test, is_train=False)
    print(f"Saved: {out_test_path}")

    # Optional quick row-count verification (includes header line)
    print("\nRow-count verification (wc -l will include header):")
    print(f"  Train in:  {len(train_df):,} rows, out: {len(out_train):,} rows")
    print(f"  Test  in:  {len(test_df):,} rows, out: {len(out_test):,} rows")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise
