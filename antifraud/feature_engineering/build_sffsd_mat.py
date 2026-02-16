#!/usr/bin/env python3
"""
build_sffsd_mat.py

Convert an S-FFSD CSV into the MATLAB .mat format expected by the RGTAN loader:
  ./data/S-FFSD/S-FFSD.mat with key "data"

Matrix column order written:
  [Source, Target, Amount, Location, Time, Type, Labels]

Features Source/Target/Location/Type are label-encoded to integer IDs.
Amount/Time coerced to numeric. Labels coerced to int.

Usage:
  python build_sffsd_mat.py \
    --input data/s-ffsd/S-FFSD.csv \
    --output data/S-FFSD/S-FFSD.mat

If you omit args:
  input  defaults to: data/S-FFSD/S-FFSD.csv
  output defaults to: data/S-FFSD/S-FFSD.mat
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.io import savemat


REQUIRED_KEYS = ["time", "source", "target", "amount", "location", "type", "labels"]


def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("_", "")


def resolve_columns(df: pd.DataFrame) -> dict:
    """
    Resolve required columns case-insensitively with a small alias set.
    Returns mapping from canonical key -> actual df column name.
    """
    cols = list(df.columns)
    norm_map = {_norm(c): c for c in cols}

    aliases = {
        "time": ["time", "transactiondt", "timestamp"],
        "source": ["source", "card1", "card", "cardid", "card_id"],
        "target": ["target"],
        "amount": ["amount", "transactionamt", "amt", "transactionamount"],
        "location": ["location", "addr1", "addr2", "address", "loc"],
        "type": ["type", "productcd", "productcode"],
        "labels": ["labels", "isfraud", "fraud", "label"],
    }

    out = {}
    for key in REQUIRED_KEYS:
        found = None
        for a in aliases[key]:
            a_norm = _norm(a)
            if a_norm in norm_map:
                found = norm_map[a_norm]
                break
        if not found:
            # also allow exact normalized key match
            if key in norm_map:
                found = norm_map[key]
        if not found:
            raise ValueError(
                f"Missing required column for '{key}'.\n"
                f"Columns found: {cols}\n"
                f"Tried aliases: {aliases[key]}"
            )
        out[key] = found

    return out


def build_mat(input_csv: str, output_mat: str) -> None:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    print(f"[READ] {input_csv}", flush=True)
    df = pd.read_csv(input_csv)

    col = resolve_columns(df)

    # Standardize names & order to exactly what we want in the MAT matrix
    df2 = df[[col["source"], col["target"], col["amount"], col["location"], col["time"], col["type"], col["labels"]]].copy()
    df2.columns = ["Source", "Target", "Amount", "Location", "Time", "Type", "Labels"]

    # Coerce numeric fields
    df2["Amount"] = pd.to_numeric(df2["Amount"], errors="coerce").fillna(0.0)
    df2["Time"] = pd.to_numeric(df2["Time"], errors="coerce").fillna(0.0)

    # Labels -> int (0/1). If NaN, default to 0 but warn.
    lbl = pd.to_numeric(df2["Labels"], errors="coerce")
    if lbl.isna().any():
        n_bad = int(lbl.isna().sum())
        print(f"[WARN] {n_bad} label values were non-numeric/NaN; setting them to 0.", flush=True)
    df2["Labels"] = lbl.fillna(0).astype(int)

    # Encode categorical fields deterministically
    # (Note: this encoding is local to this conversion run; that’s OK for model input.)
    for c in ["Source", "Target", "Location", "Type"]:
        df2[c] = df2[c].astype("category").cat.codes.astype(np.int64)

    # Build final matrix
    data = df2.values.astype(np.float32)

    out_dir = os.path.dirname(output_mat) or "."
    os.makedirs(out_dir, exist_ok=True)

    print(f"[WRITE] {output_mat}", flush=True)
    savemat(output_mat, {"data": data})

    print("=" * 60)
    print("S-FFSD MAT created")
    print("Rows:", data.shape[0])
    print("Cols:", data.shape[1], "(Source,Target,Amount,Location,Time,Type,Labels)")
    print("Output:", output_mat)
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/S-FFSD/S-FFSD.csv", help="Input S-FFSD CSV path")
    ap.add_argument("--output", default="data/S-FFSD/S-FFSD.mat", help="Output MAT path")
    args = ap.parse_args()

    build_mat(args.input, args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
