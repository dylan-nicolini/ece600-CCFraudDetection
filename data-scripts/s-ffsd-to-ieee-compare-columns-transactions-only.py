#!/usr/bin/env python3
"""
s-ffsd-to-ieee-compare-columns-transactions-only.py

ONE unified script that produces:
(A) Original analysis reports (sampled where appropriate):
  1) sffsd_columns.csv
  2) ieee_transaction_columns.csv
  3) exact_matches.csv
  4) fuzzy_matches_core.csv
  5) fuzzy_matches_all_sffsd_cols.csv
  6) profile_report.csv (dtype + null_rate + nunique + examples; respects --nrows)
  7) suggested_pairs.json

(B) Updated reasoning reports (FULL dataset; streamed via chunksize):
  8) missingness_full_dataset_report.csv
  9) entity_key_utility_full_dataset_report.csv

Notes:
- Transaction-only (no identity tables).
- IEEE has no true Source/Target semantics. Entity-key utility evaluates proxy entity dimensions
  (card*, addr*, ProductCD, email domains) based on coverage, repetition, coarseness, and
  estimated edge contribution under next-k chaining (as used in your data_process pipeline).
"""

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


SFFSD_CORE = ["Time", "Amount", "Source", "Target", "Location", "Type", "Labels"]

DEFAULT_ENTITY_CANDIDATES = [
    "card1","card2","card3","card4","card5","card6",
    "addr1","addr2",
    "ProductCD",
    "P_emaildomain","R_emaildomain",
]


# -----------------------------
# small helpers
# -----------------------------

def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())

def jaccard_char(a: str, b: str) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    return len(A & B) / max(1, len(A | B))

def find_csv_in_zip(zip_path: Path, filename: str, extract_dir: Path) -> Path:
    """
    Extracts filename from zip. Handles nested paths.
    Writes a normalized copy to extract_dir/<filename>.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        # exact match first
        if filename in names:
            chosen = filename
        else:
            # suffix match (nested folders)
            candidates = [n for n in names if n.endswith("/" + filename) or n.endswith("\\" + filename) or n.endswith(filename)]
            if not candidates:
                raise FileNotFoundError(
                    f"{filename} not found in {zip_path}. Found {len(names)} entries. Example: {names[:10]}"
                )
            chosen = sorted(candidates, key=len)[0]

        zf.extract(chosen, path=extract_dir)
        extracted_path = extract_dir / chosen

        out_path = extract_dir / Path(filename).name
        if extracted_path != out_path:
            out_path.write_bytes(extracted_path.read_bytes())
        return out_path

def load_csv_sample(path: Path, nrows: Optional[int]) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)

def profile_columns(df: pd.DataFrame, file_tag: str, max_examples: int = 3) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        examples = s.dropna().unique()[:max_examples]
        rows.append({
            "file": file_tag,
            "column": c,
            "norm_column": norm(c),
            "dtype": str(s.dtype),
            "null_rate": float(s.isna().mean()),
            "nunique": int(s.nunique(dropna=True)),
            "examples": "; ".join([str(x)[:60] for x in examples]),
        })
    return pd.DataFrame(rows)


# -----------------------------
# core matching reports
# -----------------------------

def exact_matches(sffsd_cols: List[str], ieee_cols: List[str]) -> pd.DataFrame:
    s_map = {norm(c): c for c in sffsd_cols}
    i_map = {norm(c): c for c in ieee_cols}
    matches = []
    for k in sorted(set(s_map) & set(i_map)):
        matches.append({"norm": k, "sffsd_column": s_map[k], "ieee_column": i_map[k]})
    return pd.DataFrame(matches)

def fuzzy_matches_for_sffsd(sffsd_cols: List[str], ieee_cols: List[str], top_n: int = 8) -> pd.DataFrame:
    ieee_norm = [(c, norm(c)) for c in ieee_cols]
    out_rows = []
    for sc in sffsd_cols:
        sn = norm(sc)
        scored = []
        for ic, icn in ieee_norm:
            score = jaccard_char(sn, icn)
            scored.append((score, ic))
        scored.sort(reverse=True, key=lambda x: x[0])
        for rank, (score, ic) in enumerate(scored[:top_n], start=1):
            out_rows.append({"sffsd_column": sc, "ieee_candidate": ic, "score": float(score), "rank": rank})
    return pd.DataFrame(out_rows)

def suggest_pairs_tx_only(ieee_tx_cols: List[str]) -> Dict[str, object]:
    cols = set(ieee_tx_cols)
    def present(*names): return [n for n in names if n in cols]
    return {
        "Time": present("TransactionDT"),
        "Amount": present("TransactionAmt"),
        "Labels": present("isFraud"),
        "Entity_candidates_card": present("card1","card2","card3","card4","card5","card6"),
        "Entity_candidates_addr": present("addr1","addr2"),
        "Entity_candidates_product": present("ProductCD"),
        "Entity_candidates_emaildomain": present("P_emaildomain","R_emaildomain"),
        "Join_key_candidates": present("TransactionID"),
        "Notes": [
            "IEEE-CIS does not expose sender/receiver semantics; entity columns are proxy dimensions.",
            "Prefer proxy entity keys with low missingness, repeated groupings, and low top-N concentration."
        ]
    }


# -----------------------------
# FULL dataset: missingness + entity utility (streaming)
# -----------------------------

def stream_missingness(csv_path: Path, file_tag: str, chunksize: int) -> pd.DataFrame:
    """
    Computes missing_count and missing_pct for ALL columns across full file.
    Streaming chunksize to avoid memory blowups.
    """
    total_rows = 0
    missing_counts: Dict[str, int] = {}
    col_order: List[str] = []

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        if not col_order:
            col_order = list(chunk.columns)
            for c in col_order:
                missing_counts[c] = 0

        total_rows += len(chunk)
        miss = chunk.isna().sum()
        for c, v in miss.items():
            missing_counts[c] = missing_counts.get(c, 0) + int(v)

    rows = []
    for c in col_order:
        mc = int(missing_counts.get(c, 0))
        pct = (mc / max(1, total_rows)) * 100.0
        rows.append({
            "file": file_tag,
            "column": c,
            "missing_count": mc,
            "total_rows": int(total_rows),
            "missing_pct": float(pct),
        })

    out = pd.DataFrame(rows).sort_values(["missing_pct","missing_count"], ascending=False).reset_index(drop=True)
    return out


def edges_next_k_for_group_size(g: int, k: int) -> int:
    """
    Exact edges created by next-k chaining within a group of size g:
    Connect each item to the next up to k items in time order.
    """
    if g <= 1:
        return 0
    if g <= k + 1:
        return g * (g - 1) // 2
    return (g - 1) * k - (k * (k - 1)) // 2


def stream_entity_utility(csv_path: Path,
                          file_tag: str,
                          candidates: List[str],
                          chunksize: int,
                          next_k: int,
                          top_freq_n: int) -> pd.DataFrame:
    """
    Streaming entity utility stats for a subset of candidate columns:
      - missing %, nunique
      - rows in groups >=2 and >=5 (based on value_counts)
      - avg/median group size for groups >=2
      - top-N concentration (coarseness proxy)
      - estimated edges from next-k chaining (exact closed form)
    """
    total_rows = 0
    missing_counts = {c: 0 for c in candidates}
    uniq_sets: Dict[str, set] = {c: set() for c in candidates}
    value_counts: Dict[str, Dict[object, int]] = {c: {} for c in candidates}

    def update_counts(col: str, series: pd.Series):
        vc = series.value_counts(dropna=True)
        d = value_counts[col]
        for key, cnt in vc.items():
            d[key] = d.get(key, 0) + int(cnt)

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        total_rows += len(chunk)
        for c in candidates:
            if c not in chunk.columns:
                continue
            s = chunk[c]
            missing_counts[c] += int(s.isna().sum())
            # cardinality: acceptable for these proxy keys; avoid doing this for V* etc.
            uniq_sets[c].update(s.dropna().unique().tolist())
            update_counts(c, s)

    rows = []
    for c in candidates:
        vc_dict = value_counts.get(c, {})
        # column might be missing entirely
        if not vc_dict and missing_counts.get(c, 0) == 0:
            continue

        vc = pd.Series(vc_dict, dtype="int64").sort_values(ascending=False)
        missing_pct = (missing_counts.get(c, 0) / max(1, total_rows)) * 100.0
        nunique = len(uniq_sets.get(c, set()))

        vc_ge2 = vc[vc >= 2]
        vc_ge5 = vc[vc >= 5]

        rows_in_ge2 = int(vc_ge2.sum()) if len(vc_ge2) else 0
        rows_in_ge5 = int(vc_ge5.sum()) if len(vc_ge5) else 0

        pct_rows_ge2 = (rows_in_ge2 / max(1, total_rows)) * 100.0
        pct_rows_ge5 = (rows_in_ge5 / max(1, total_rows)) * 100.0

        avg_group_size_ge2 = float(vc_ge2.mean()) if len(vc_ge2) else 0.0
        median_group_size_ge2 = float(vc_ge2.median()) if len(vc_ge2) else 0.0

        top_conc = (float(vc.head(top_freq_n).sum()) / max(1, total_rows)) * 100.0 if len(vc) else 0.0

        est_edges = 0
        for g in vc_ge2.values:
            est_edges += edges_next_k_for_group_size(int(g), next_k)

        rows.append({
            "file": file_tag,
            "column": c,
            "total_rows": int(total_rows),
            "missing_count": int(missing_counts.get(c, 0)),
            "missing_pct": round(missing_pct, 6),
            "nunique": int(nunique),
            "rows_in_groups_ge2": rows_in_ge2,
            "pct_rows_in_groups_ge2": round(pct_rows_ge2, 6),
            "rows_in_groups_ge5": rows_in_ge5,
            "pct_rows_in_groups_ge5": round(pct_rows_ge5, 6),
            "avg_group_size_ge2": round(avg_group_size_ge2, 6),
            "median_group_size_ge2": round(median_group_size_ge2, 6),
            f"top{top_freq_n}_concentration_pct": round(top_conc, 6),
            f"est_edges_next_k_{next_k}": int(est_edges),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["pct_rows_in_groups_ge2", f"top{top_freq_n}_concentration_pct", "missing_pct"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
    return out


# -----------------------------
# main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--sffsd-zip", type=Path, default=None)
    ap.add_argument("--sffsd-csv", type=Path, default=None)

    ap.add_argument("--ieee-zip", type=Path, default=None)
    ap.add_argument("--ieee-train-tx", type=Path, default=None)
    ap.add_argument("--ieee-test-tx", type=Path, default=None)

    ap.add_argument("--out-dir", type=Path, required=True)

    ap.add_argument("--nrows", type=int, default=200_000, help="Sample rows for schema/profile (0 = all)")
    ap.add_argument("--top-n", type=int, default=8, help="Top-N fuzzy candidates per S-FFSD column")

    ap.add_argument("--chunksize", type=int, default=200_000, help="Chunksize for FULL dataset streaming reports")
    ap.add_argument("--next-k", type=int, default=3, help="next-k chaining used in graph construction")
    ap.add_argument("--top-freq-n", type=int, default=5, help="Top-N concentration check for coarseness")
    ap.add_argument("--entity-candidates", type=str, default=",".join(DEFAULT_ENTITY_CANDIDATES),
                    help="Comma-separated candidate proxy entity columns (IEEE transaction columns)")

    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = out_dir / "_extracted"

    nrows = None if args.nrows == 0 else args.nrows
    entity_candidates = [c.strip() for c in args.entity_candidates.split(",") if c.strip()]

    # Resolve S-FFSD
    if args.sffsd_csv:
        sffsd_path = args.sffsd_csv
    elif args.sffsd_zip:
        sffsd_path = find_csv_in_zip(args.sffsd_zip, "S-FFSD.csv", extract_dir / "sffsd")
    else:
        raise ValueError("Provide --sffsd-zip or --sffsd-csv")

    # Resolve IEEE transaction paths
    if args.ieee_train_tx and args.ieee_test_tx:
        train_tx_path, test_tx_path = args.ieee_train_tx, args.ieee_test_tx
    elif args.ieee_zip:
        train_tx_path = find_csv_in_zip(args.ieee_zip, "train_transaction.csv", extract_dir / "ieee")
        test_tx_path  = find_csv_in_zip(args.ieee_zip, "test_transaction.csv", extract_dir / "ieee")
    else:
        raise ValueError("Provide --ieee-zip or both --ieee-train-tx and --ieee-test-tx")

    # -----------------------------
    # (A) Sampled reports
    # -----------------------------
    print(f"[A] Loading sampled data for schema/profile (nrows={args.nrows})...")
    sffsd = load_csv_sample(sffsd_path, nrows=nrows)
    train_tx = load_csv_sample(train_tx_path, nrows=nrows)
    test_tx = load_csv_sample(test_tx_path, nrows=nrows)

    s_cols = list(sffsd.columns)
    ieee_tx_cols = sorted(set(train_tx.columns) | set(test_tx.columns))

    pd.DataFrame({"sffsd_column": s_cols}).to_csv(out_dir / "sffsd_columns.csv", index=False)
    pd.DataFrame({"ieee_transaction_column": ieee_tx_cols}).to_csv(out_dir / "ieee_transaction_columns.csv", index=False)

    exact = exact_matches(s_cols, ieee_tx_cols)
    exact.to_csv(out_dir / "exact_matches.csv", index=False)

    fuzzy_core = fuzzy_matches_for_sffsd(SFFSD_CORE, ieee_tx_cols, top_n=args.top_n)
    fuzzy_core.to_csv(out_dir / "fuzzy_matches_core.csv", index=False)

    fuzzy_all = fuzzy_matches_for_sffsd(s_cols, ieee_tx_cols, top_n=min(args.top_n, 5))
    fuzzy_all.to_csv(out_dir / "fuzzy_matches_all_sffsd_cols.csv", index=False)

    prof = pd.concat([
        profile_columns(sffsd, "S-FFSD"),
        profile_columns(train_tx, "IEEE_train_transaction"),
        profile_columns(test_tx, "IEEE_test_transaction"),
    ], ignore_index=True)
    prof.to_csv(out_dir / "profile_report.csv", index=False)

    (out_dir / "suggested_pairs.json").write_text(json.dumps(suggest_pairs_tx_only(ieee_tx_cols), indent=2))

    # -----------------------------
    # (B) FULL dataset streamed reports
    # -----------------------------
    print(f"[B] FULL dataset missingness report (chunksize={args.chunksize})...")
    miss = pd.concat([
        stream_missingness(sffsd_path, "S-FFSD", args.chunksize),
        stream_missingness(train_tx_path, "IEEE_train_transaction", args.chunksize),
        stream_missingness(test_tx_path, "IEEE_test_transaction", args.chunksize),
    ], ignore_index=True)
    miss.to_csv(out_dir / "missingness_full_dataset_report.csv", index=False)

    print(f"[B] FULL dataset proxy entity utility report (chunksize={args.chunksize}, next_k={args.next_k})...")
    ent = pd.concat([
        stream_entity_utility(train_tx_path, "IEEE_train_transaction", entity_candidates, args.chunksize, args.next_k, args.top_freq_n),
        stream_entity_utility(test_tx_path, "IEEE_test_transaction", entity_candidates, args.chunksize, args.next_k, args.top_freq_n),
    ], ignore_index=True)
    ent.to_csv(out_dir / "entity_key_utility_full_dataset_report.csv", index=False)

    print("Done. Wrote reports to:", out_dir.resolve())
    print("Key outputs:")
    print(" - missingness_full_dataset_report.csv")
    print(" - entity_key_utility_full_dataset_report.csv")
    print(" - fuzzy_matches_core.csv")
    print(" - profile_report.csv")


if __name__ == "__main__":
    main()
