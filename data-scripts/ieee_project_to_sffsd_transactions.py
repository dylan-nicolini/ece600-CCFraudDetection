#!/usr/bin/env python3
"""
ieee_project_to_sffsd_transactions.py

End-to-end IEEE-CIS transaction modification for S-FFSD pipeline compatibility:

STEP 1 (Reduce width, keep rows):
- Uses missingness_full_dataset_report.csv to keep columns <= missing-threshold
- Ensures train/test share a consistent schema (intersection of keepable columns)
- Forces keeping core/proxy columns needed for projection

STEP 2 (Project / reshape to S-FFSD transaction interface):
- Produces "projected" train/test files that include:
  Time, Amount, Source, Target, Location, Type, Labels (train only)
- These are proxy entity dimensions (not semantic sender/receiver)

STEP 3 (Validate):
- Confirms row counts preserved
- Confirms required columns exist
- Confirms train/test schemas match (except Labels)
- Outputs validation_report.json

Inputs:
- --ieee-zip (contains train_transaction.csv + test_transaction.csv)
- --missingness-report (your analysis output)

Outputs (written to --out-dir):
- train_transaction_reduced.csv
- test_transaction_reduced.csv
- ieee_projected_sffsd_train.csv
- ieee_projected_sffsd_test.csv
- kept_columns_manifest.json
- validation_report.json

Example (PowerShell):
python .\ieee_project_to_sffsd_transactions.py `
  --ieee-zip "C:\...\ieee-fraud-detection.zip" `
  --missingness-report "C:\...\missingness_full_dataset_report.csv" `
  --out-dir "C:\...\ieee_modified" `
  --missing-threshold 90 `
  --chunksize 200000
"""

import argparse
import csv
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import pandas as pd


# -------------------------
# Defaults (you can override via CLI)
# -------------------------

DEFAULT_FORCE_KEEP_TRAIN = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "isFraud",         # train only
    "card1", "card2",  # proxy entities
    "addr1",
    "ProductCD",       # keep as feature even if coarse
    "card4", "card6",  # keep as features even if coarse
]

DEFAULT_FORCE_KEEP_TEST = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "card1", "card2",
    "addr1",
    "ProductCD",
    "card4", "card6",
]

# Projection (proxy) to S-FFSD interface
DEFAULT_PROJECTION = {
    "Time": "TransactionDT",
    "Amount": "TransactionAmt",
    "Labels": "isFraud",       # train only
    "Source": "card1",
    "Target": "card2",
    "Location": "addr1",
    "Type": "ProductCD",       # preferred over card4/card6 because they are extremely coarse
}


# -------------------------
# Zip extraction helper
# -------------------------

def extract_from_zip(zip_path: Path, filename: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        matches = [n for n in z.namelist() if n.endswith(filename)]
        if not matches:
            raise FileNotFoundError(f"{filename} not found in {zip_path}")
        chosen = sorted(matches, key=len)[0]
        z.extract(chosen, out_dir)
        extracted = out_dir / chosen

        normalized = out_dir / Path(filename).name
        if extracted != normalized:
            normalized.write_bytes(extracted.read_bytes())
        return normalized


# -------------------------
# Column selection logic
# -------------------------

def load_keep_columns_from_missingness(
    missingness_csv: Path,
    threshold_pct: float,
    file_tag: str
) -> Set[str]:
    df = pd.read_csv(missingness_csv)
    sub = df[df["file"] == file_tag].copy()
    if sub.empty:
        raise ValueError(f"No rows found in missingness report for file='{file_tag}'.")
    keep = set(sub[sub["missing_pct"] <= threshold_pct]["column"].astype(str).tolist())
    return keep


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


# -------------------------
# Streaming transforms
# -------------------------

def stream_reduce_csv(
    in_csv: Path,
    out_csv: Path,
    keep_columns: List[str],
    chunksize: int
) -> int:
    """
    Writes a reduced-column CSV (no row drops) and returns row count written.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    first = True
    written_rows = 0

    for chunk in pd.read_csv(in_csv, chunksize=chunksize, low_memory=False):
        cols_present = [c for c in keep_columns if c in chunk.columns]
        out_chunk = chunk[cols_present]
        out_chunk.to_csv(out_csv, index=False, mode="w" if first else "a", header=first)
        first = False
        written_rows += len(out_chunk)

    return written_rows


def stream_project_to_sffsd(
    in_csv: Path,
    out_csv: Path,
    keep_feature_columns: List[str],
    projection: Dict[str, str],
    is_train: bool,
    chunksize: int
) -> int:
    """
    Creates an S-FFSD-compatible transaction file by:
    - adding S-FFSD interface columns: Time, Amount, Source, Target, Location, Type, Labels
    - keeping reduced IEEE features (optional)
    Returns row count written.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    first = True
    written_rows = 0

    required_out_cols = ["Time", "Source", "Target", "Amount", "Location", "Type"]
    if is_train:
        required_out_cols.append("Labels")

    # We keep features AFTER the S-FFSD interface columns
    # Exclude the projection source columns to avoid duplicates unless you explicitly want them
    projection_sources = set(projection.values())
    feature_cols_final = [c for c in keep_feature_columns if c not in projection_sources]

    out_cols = required_out_cols + feature_cols_final

    for chunk in pd.read_csv(in_csv, chunksize=chunksize, low_memory=False):
        # Build interface fields (proxy projection)
        out = pd.DataFrame(index=chunk.index)

        # Time / Amount
        out["Time"] = chunk.get(projection["Time"])
        out["Amount"] = chunk.get(projection["Amount"])

        # Proxy entities (no semantic directionality implied)
        out["Source"] = chunk.get(projection["Source"])
        out["Target"] = chunk.get(projection["Target"])
        out["Location"] = chunk.get(projection["Location"])
        out["Type"] = chunk.get(projection["Type"])

        # Labels only in train
        if is_train:
            labels_col = projection.get("Labels", "isFraud")
            out["Labels"] = chunk.get(labels_col)

        # Add remaining features
        for c in feature_cols_final:
            if c in chunk.columns:
                out[c] = chunk[c]

        # Ensure correct output column order
        out = out[out_cols]

        out.to_csv(out_csv, index=False, mode="w" if first else "a", header=first)
        first = False
        written_rows += len(out)

    return written_rows


# -------------------------
# Validation
# -------------------------

def count_rows_fast(csv_path: Path) -> int:
    """
    Fast(ish) row count without loading in pandas.
    """
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        # subtract header
        return max(0, sum(1 for _ in f) - 1)


def get_header(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        return next(reader)


def validate_outputs(
    orig_train: Path,
    orig_test: Path,
    reduced_train: Path,
    reduced_test: Path,
    proj_train: Path,
    proj_test: Path,
) -> Dict[str, object]:
    report: Dict[str, object] = {}

    # Row counts
    orig_train_rows = count_rows_fast(orig_train)
    orig_test_rows  = count_rows_fast(orig_test)
    reduced_train_rows = count_rows_fast(reduced_train)
    reduced_test_rows  = count_rows_fast(reduced_test)
    proj_train_rows = count_rows_fast(proj_train)
    proj_test_rows  = count_rows_fast(proj_test)

    report["row_counts"] = {
        "orig_train": orig_train_rows,
        "orig_test": orig_test_rows,
        "reduced_train": reduced_train_rows,
        "reduced_test": reduced_test_rows,
        "projected_train": proj_train_rows,
        "projected_test": proj_test_rows,
        "row_count_preserved_train": (orig_train_rows == reduced_train_rows == proj_train_rows),
        "row_count_preserved_test": (orig_test_rows == reduced_test_rows == proj_test_rows),
    }

    # Required columns
    train_hdr = get_header(proj_train)
    test_hdr  = get_header(proj_test)

    required_train = {"Time","Source","Target","Amount","Location","Type","Labels"}
    required_test  = {"Time","Source","Target","Amount","Location","Type"}

    report["required_columns_present"] = {
        "train": sorted(required_train - set(train_hdr)),
        "test": sorted(required_test - set(test_hdr)),
        "train_ok": required_train.issubset(set(train_hdr)),
        "test_ok": required_test.issubset(set(test_hdr)),
    }

    # Schema alignment (train/test should match except Labels)
    train_minus_labels = [c for c in train_hdr if c != "Labels"]
    report["schema_consistency"] = {
        "train_minus_labels_equals_test": (train_minus_labels == test_hdr),
        "train_cols": len(train_hdr),
        "test_cols": len(test_hdr),
    }

    return report


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ieee-zip", required=True, type=Path)
    ap.add_argument("--missingness-report", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)

    ap.add_argument("--missing-threshold", type=float, default=90.0)
    ap.add_argument("--chunksize", type=int, default=200_000)

    ap.add_argument("--force-keep-train", type=str, default=",".join(DEFAULT_FORCE_KEEP_TRAIN))
    ap.add_argument("--force-keep-test", type=str, default=",".join(DEFAULT_FORCE_KEEP_TEST))

    ap.add_argument("--projection-json", type=str, default=None,
                    help="Optional JSON string overriding DEFAULT_PROJECTION")
    ap.add_argument("--keep-features", action="store_true",
                    help="If set, keep reduced IEEE features in projected output (recommended). "
                         "If not set, projected output will include only the S-FFSD interface columns.")

    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = out_dir / "_extract"

    train_path = extract_from_zip(args.ieee_zip, "train_transaction.csv", extract_dir)
    test_path  = extract_from_zip(args.ieee_zip, "test_transaction.csv", extract_dir)

    # Load keep columns from missingness thresholds
    keep_train = load_keep_columns_from_missingness(args.missingness_report, args.missing_threshold, "IEEE_train_transaction")
    keep_test  = load_keep_columns_from_missingness(args.missingness_report, args.missing_threshold, "IEEE_test_transaction")

    keep_both = keep_train.intersection(keep_test)

    force_keep_train = parse_csv_list(args.force_keep_train)
    force_keep_test  = parse_csv_list(args.force_keep_test)

    keep_train_final = sorted(set(keep_both).union(force_keep_train))
    keep_test_final  = sorted(set(keep_both).union(force_keep_test))

    # Projection mapping
    projection = DEFAULT_PROJECTION.copy()
    if args.projection_json:
        projection.update(json.loads(args.projection_json))

    # Output paths
    reduced_train_out = out_dir / "train_transaction_reduced.csv"
    reduced_test_out  = out_dir / "test_transaction_reduced.csv"
    proj_train_out    = out_dir / "ieee_projected_sffsd_train.csv"
    proj_test_out     = out_dir / "ieee_projected_sffsd_test.csv"

    # STEP 1: reduce
    print(f"[STEP 1] Reducing columns (no row drops), threshold={args.missing_threshold}%...")
    train_written = stream_reduce_csv(train_path, reduced_train_out, keep_train_final, args.chunksize)
    test_written  = stream_reduce_csv(test_path, reduced_test_out, keep_test_final, args.chunksize)
    print(f"  Wrote reduced train rows: {train_written:,}")
    print(f"  Wrote reduced test  rows: {test_written:,}")

    # Decide which features to keep in projected file
    feature_cols_train = keep_train_final if args.keep_features else []
    feature_cols_test  = keep_test_final if args.keep_features else []

    # STEP 2: project to S-FFSD interface
    print("[STEP 2] Projecting to S-FFSD-compatible transaction interface...")
    train_proj_written = stream_project_to_sffsd(
        reduced_train_out, proj_train_out,
        keep_feature_columns=feature_cols_train,
        projection=projection,
        is_train=True,
        chunksize=args.chunksize
    )
    test_proj_written = stream_project_to_sffsd(
        reduced_test_out, proj_test_out,
        keep_feature_columns=feature_cols_test,
        projection=projection,
        is_train=False,
        chunksize=args.chunksize
    )
    print(f"  Wrote projected train rows: {train_proj_written:,}")
    print(f"  Wrote projected test  rows: {test_proj_written:,}")

    # Manifest for reproducibility
    manifest = {
        "inputs": {
            "ieee_zip": str(args.ieee_zip),
            "missingness_report": str(args.missingness_report),
        },
        "params": {
            "missing_threshold_pct": args.missing_threshold,
            "chunksize": args.chunksize,
            "keep_features_in_projected_output": bool(args.keep_features),
        },
        "forced_keep": {
            "train": force_keep_train,
            "test": force_keep_test,
        },
        "kept_columns": {
            "train_reduced": keep_train_final,
            "test_reduced": keep_test_final,
        },
        "projection": projection,
        "outputs": {
            "reduced_train": str(reduced_train_out),
            "reduced_test": str(reduced_test_out),
            "projected_train": str(proj_train_out),
            "projected_test": str(proj_test_out),
        },
        "notes": [
            "IEEE projection creates an S-FFSD-compatible interface using proxy entity dimensions.",
            "No rows are dropped; only columns are reduced and additional interface columns are created."
        ]
    }
    (out_dir / "kept_columns_manifest.json").write_text(json.dumps(manifest, indent=2))

    # STEP 3: validate
    print("[STEP 3] Validating outputs...")
    validation = validate_outputs(train_path, test_path, reduced_train_out, reduced_test_out, proj_train_out, proj_test_out)
    (out_dir / "validation_report.json").write_text(json.dumps(validation, indent=2))

    print("Done.")
    print(f"Outputs in: {out_dir.resolve()}")
    print("Key files:")
    print(" - train_transaction_reduced.csv")
    print(" - test_transaction_reduced.csv")
    print(" - ieee_projected_sffsd_train.csv")
    print(" - ieee_projected_sffsd_test.csv")
    print(" - validation_report.json")


if __name__ == "__main__":
    main()
