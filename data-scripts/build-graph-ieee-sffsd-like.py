#!/usr/bin/env python3
"""
build-graph-ieee-sffsd-like.py

Builds a DGL graph from IEEE-CIS data that has already been projected into an
"S-FFSD-like" transaction schema (i.e., it contains at minimum:
Time, Amount, Source, Target, Location, Type, and (for train) Labels).

Graph construction mirrors the S-FFSD pipeline:
- Sort by Time
- For each of Source/Target/Location/Type:
    group by that column
    connect each txn to the next K txns within the group (next-k chaining)
- Attach node features (all cols except Labels) and node labels
- Create train/test masks
- Save as graph-IEEE.bin (or your chosen name)

This makes the resulting dataset consumable by GTAN/RGTAN in the same way the
S-FFSD graph is consumed.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import dgl
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


REQUIRED_COLS = ["Time", "Amount", "Source", "Target", "Location", "Type"]


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def ensure_required_cols(df: pd.DataFrame, tag: str) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{tag} is missing required columns: {missing}")


def build_edges_next_k(df: pd.DataFrame, group_cols, next_k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Mirrors the optimized edge build used in S-FFSD:
    - presort by Time
    - for each group col:
        within each group (time-ordered), connect i -> i+1..i+K
    """
    all_src = []
    all_dst = []

    df_sorted = df.sort_values("Time")  # preserves temporal ordering within groups

    for col in group_cols:
        print(f"  building edges for grouping column: {col}")
        for _, gdf in tqdm(df_sorted.groupby(col), desc=f"groupby({col})"):
            idx = gdf.index.to_numpy()
            if len(idx) <= 1:
                continue
            for j in range(1, next_k + 1):
                if len(idx) > j:
                    all_src.append(idx[:-j])
                    all_dst.append(idx[j:])

    if not all_src:
        raise RuntimeError("No edges were created. Your proxy entity columns may be too sparse/coarse.")

    src = np.concatenate(all_src)
    dst = np.concatenate(all_dst)
    return src, dst


def label_encode_inplace(df: pd.DataFrame, cols: list[str]) -> None:
    """
    Matches S-FFSD behavior: LabelEncoder over string-casted values.
    """
    for c in cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].apply(str).values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, required=True, help="Projected IEEE train CSV (has Labels)")
    ap.add_argument("--test-csv", type=Path, required=True, help="Projected IEEE test CSV (no Labels is ok)")
    ap.add_argument("--out-graph", type=Path, required=True, help="Output .bin path (e.g., graph-IEEE.bin)")
    ap.add_argument("--next-k", type=int, default=3, help="Next-k chaining within each proxy entity group")
    ap.add_argument("--fill-test-label", type=int, default=0,
                    help="If test has no Labels, fill with this value (default 0).")
    args = ap.parse_args()

    print("Loading projected IEEE train/test...")
    train = load_csv(args.train_csv)
    test = load_csv(args.test_csv)

    ensure_required_cols(train, "train")
    ensure_required_cols(test, "test")

    # Labels handling
    if "Labels" not in train.columns:
        raise ValueError("Train CSV must contain a Labels column.")
    if "Labels" not in test.columns:
        test = test.copy()
        test["Labels"] = args.fill_test_label

    # Preserve row counts and create masks
    n_train = len(train)
    n_test = len(test)
    print(f"Row counts: train={n_train:,} test={n_test:,} total={n_train+n_test:,}")

    df = pd.concat([train, test], ignore_index=True)

    # Basic cleanup
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Build edges (same grouping columns as S-FFSD data_process)
    group_cols = ["Source", "Target", "Location", "Type"]
    print("Building graph edges (next-k chaining)...")
    src, dst = build_edges_next_k(df, group_cols, next_k=args.next_k)

    g = dgl.graph((src, dst), num_nodes=len(df))

    # Encode categorical proxy columns in-place (mirrors S-FFSD behavior)
    # Note: this only encodes the entity key columns; other columns remain numeric as-is.
    print("Encoding categorical proxy columns: Source/Target/Location/Type")
    label_encode_inplace(df, group_cols)

    # Attach features and labels
    labels = df["Labels"].astype(int).to_numpy()
    feat_df = df.drop(columns=["Labels"])

    # Ensure all features are numeric
    # If any non-numeric columns exist, coerce them (should be rare in projected IEEE)
    for c in feat_df.columns:
        if feat_df[c].dtype == "object":
            feat_df[c] = pd.to_numeric(feat_df[c], errors="coerce").fillna(0)

    g.ndata["label"] = torch.from_numpy(labels).long()
    g.ndata["feat"] = torch.from_numpy(feat_df.to_numpy()).float()

    # Masks
    train_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
    test_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
    train_mask[:n_train] = True
    test_mask[n_train:] = True

    g.ndata["train_mask"] = train_mask
    g.ndata["test_mask"] = test_mask

    # Save
    args.out_graph.parent.mkdir(parents=True, exist_ok=True)
    dgl.data.utils.save_graphs(str(args.out_graph), [g])

    print("Done.")
    print(f"Saved graph: {args.out_graph}")
    print(f"Graph summary: nodes={g.num_nodes():,} edges={g.num_edges():,} feat_dim={g.ndata['feat'].shape[1]}")


if __name__ == "__main__":
    main()
