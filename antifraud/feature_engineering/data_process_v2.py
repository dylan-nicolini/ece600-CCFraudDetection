#!/usr/bin/env python3
"""
data_process_v2.py

Fast preprocessing pipeline for S-FFSD *or* any IEEE "S-FFSD-like" CSV.

Why v2:
- The original featmap_gen() is extremely slow (row-by-row + dataframe filtering).
- This version replaces it with an O(N * W) vectorized implementation using
  cumulative sums + searchsorted over sorted Time (W = number of time windows).
- Neighbor features are computed via fast DGL message passing (update_all),
  not per-node k-hop subgraph calls.

Inputs:
- A CSV containing (case-sensitive) columns:
    Time, Source, Target, Amount, Location, Type, Labels
  (Extra columns are allowed; they will be preserved and used as numeric features.)

Outputs (in --output-dir):
- <tag>neofull.csv              : engineered feature CSV
- graph-<tag>.bin               : DGL graph with ndata['feat'] and ndata['label']
- <tag>_neigh_feat.csv           : optional neighbor structural features (scaled)
- <tag>_meta.json                : run metadata (params + basic stats)

Example:
  python3 data_process_v2.py \
    --input-file ./data/S-FFSD.csv \
    --output-dir ./data \
    --tag S-FFSD \
    --edge-per-trans 3

For an IEEE experiment file that is S-FFSD-like:
  python3 data_process_v2.py \
    --input-file ./data/ieee_experiment1.csv \
    --output-dir ./data/processed/ieee_e1 \
    --tag ieee_e1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

import torch
import dgl
import dgl.data.utils

from sklearn.preprocessing import LabelEncoder, StandardScaler


REQUIRED_COLS = ["Time", "Source", "Target", "Amount", "Location", "Type", "Labels"]


@dataclass
class Outputs:
    neofull_csv: str
    graph_bin: str
    neigh_csv: str
    meta_json: str


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _now() -> float:
    return time.time()


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # drop accidental index columns
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed", case=False, regex=True)]
    return df


def _validate_required(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            f"Expected at least: {REQUIRED_COLS}"
        )


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Time"] = pd.to_numeric(out["Time"], errors="coerce")
    out["Amount"] = pd.to_numeric(out["Amount"], errors="coerce")
    out["Labels"] = pd.to_numeric(out["Labels"], errors="coerce")
    out = out.dropna(subset=["Time", "Amount", "Labels"]).reset_index(drop=True)
    out["Labels"] = out["Labels"].astype(int)
    return out


def featmap_gen_fast_amount_only(df: pd.DataFrame, windows: List[float]) -> pd.DataFrame:
    """
    Fast rolling-window features over Time for Amount:
      - trans_at_avg_<w>, trans_at_totl_<w>, trans_at_std_<w>, trans_at_bias_<w>, trans_at_num_<w>

    Complexity: O(N * len(windows)) after sorting by Time.
    """
    df = df.sort_values("Time").reset_index(drop=True)

    t = df["Time"].to_numpy(dtype=np.float64)
    amt = df["Amount"].to_numpy(dtype=np.float64)

    csum = np.cumsum(amt)
    csum2 = np.cumsum(amt * amt)

    idx = np.arange(len(df), dtype=np.int64)

    out = df.copy()

    for w in windows:
        w = float(w)
        left = np.searchsorted(t, t - w, side="left").astype(np.int64)

        left_minus_1 = left - 1
        left_sum = np.where(left > 0, csum[left_minus_1], 0.0)
        left_sum2 = np.where(left > 0, csum2[left_minus_1], 0.0)

        win_sum = csum[idx] - left_sum
        win_sum2 = csum2[idx] - left_sum2
        win_n = (idx - left + 1).astype(np.float64)

        win_mean = win_sum / win_n
        win_var = np.maximum(win_sum2 / win_n - win_mean * win_mean, 0.0)
        win_std = np.sqrt(win_var)

        tag = str(int(w)) if w.is_integer() else str(w).replace(".", "_")

        out[f"trans_at_avg_{tag}"] = win_mean
        out[f"trans_at_totl_{tag}"] = win_sum
        out[f"trans_at_std_{tag}"] = win_std
        out[f"trans_at_bias_{tag}"] = amt - win_mean
        out[f"trans_at_num_{tag}"] = win_n.astype(np.int64)

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def build_graph_from_groups(df: pd.DataFrame, edge_per_trans: int) -> dgl.DGLGraph:
    """
    Build edges by:
      for each of [Source, Target, Location, Type]:
        groupby(column), within each group sort by Time,
        connect i -> i+1 .. i+edge_per_trans
    """
    pair = ["Source", "Target", "Location", "Type"]

    all_src = []
    all_dst = []

    df_sorted_time = df.sort_values("Time")

    for col in pair:
        for _, gdf in df_sorted_time.groupby(col, sort=False):
            idx = gdf.index.to_numpy()
            if len(idx) <= 1:
                continue
            for j in range(1, edge_per_trans + 1):
                if len(idx) > j:
                    all_src.append(idx[:-j])
                    all_dst.append(idx[j:])

    if not all_src:
        raise RuntimeError("No edges were created; check column distributions and edge_per_trans.")

    src = np.concatenate(all_src)
    dst = np.concatenate(all_dst)
    return dgl.graph((src, dst))


def label_encode_inplace(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        le = LabelEncoder()
        out[c] = le.fit_transform(out[c].astype(str).values)
    return out


def attach_node_data(g: dgl.DGLGraph, feat_df: pd.DataFrame, labels: pd.Series) -> None:
    feat_numeric = feat_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    g.ndata["feat"] = torch.from_numpy(feat_numeric.to_numpy(dtype=np.float32))
    g.ndata["label"] = torch.from_numpy(labels.to_numpy(dtype=np.int64))


def compute_neighbor_features_fast(g: dgl.DGLGraph, scale: bool = True) -> pd.DataFrame:
    """
    Fast neighbor structural features using DGL message passing.
    Produces:
      degree, riskstat, 1hop_degree, 2hop_degree, 1hop_riskstat, 2hop_riskstat

    NOTE: riskstat uses labels (can leak). Use --no-risk-features for leakage-safe.
    """
    import dgl.function as fn

    deg = g.in_degrees().float().unsqueeze(1)
    risk = (g.ndata["label"] == 1).float().unsqueeze(1)

    g.ndata["_deg"] = deg
    g.ndata["_risk"] = risk

    g.update_all(fn.copy_u("_deg", "m"), fn.sum("m", "_deg_1"))
    g.update_all(fn.copy_u("_risk", "m"), fn.sum("m", "_risk_1"))

    g.update_all(fn.copy_u("_deg_1", "m"), fn.sum("m", "_deg_2"))
    g.update_all(fn.copy_u("_risk_1", "m"), fn.sum("m", "_risk_2"))

    feats = torch.cat(
        [g.ndata["_deg"], g.ndata["_risk"], g.ndata["_deg_1"], g.ndata["_deg_2"], g.ndata["_risk_1"], g.ndata["_risk_2"]],
        dim=1,
    ).cpu().numpy()

    cols = ["degree", "riskstat", "1hop_degree", "2hop_degree", "1hop_riskstat", "2hop_riskstat"]
    df = pd.DataFrame(feats, columns=cols)

    if scale:
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=cols)

    for k in ["_deg", "_risk", "_deg_1", "_risk_1", "_deg_2", "_risk_2"]:
        if k in g.ndata:
            del g.ndata[k]
    return df


def compute_neighbor_features_degree_only(g: dgl.DGLGraph, scale: bool = True) -> pd.DataFrame:
    """
    Leakage-safe neighbor features:
      degree, 1hop_degree, 2hop_degree
    """
    import dgl.function as fn

    deg = g.in_degrees().float().unsqueeze(1)
    g.ndata["_deg"] = deg

    g.update_all(fn.copy_u("_deg", "m"), fn.sum("m", "_deg_1"))
    g.update_all(fn.copy_u("_deg_1", "m"), fn.sum("m", "_deg_2"))

    feats = torch.cat([g.ndata["_deg"], g.ndata["_deg_1"], g.ndata["_deg_2"]], dim=1).cpu().numpy()
    cols = ["degree", "1hop_degree", "2hop_degree"]
    df = pd.DataFrame(feats, columns=cols)

    if scale:
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=cols)

    for k in ["_deg", "_deg_1", "_deg_2"]:
        if k in g.ndata:
            del g.ndata[k]
    return df


def resolve_outputs(output_dir: str, tag: str) -> Outputs:
    if tag == "S-FFSD":
        neofull = os.path.join(output_dir, "S-FFSDneofull.csv")
        graphb = os.path.join(output_dir, "graph-S-FFSD.bin")
        neigh = os.path.join(output_dir, "S-FFSD_neigh_feat.csv")
        meta = os.path.join(output_dir, "S-FFSD_meta.json")
    else:
        neofull = os.path.join(output_dir, f"{tag}neofull.csv")
        graphb = os.path.join(output_dir, f"graph-{tag}.bin")
        neigh = os.path.join(output_dir, f"{tag}_neigh_feat.csv")
        meta = os.path.join(output_dir, f"{tag}_meta.json")
    return Outputs(neofull, graphb, neigh, meta)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", required=True, help="CSV with Time,Source,Target,Amount,Location,Type,Labels")
    ap.add_argument("--output-dir", required=True, help="Directory to write artifacts")
    ap.add_argument("--tag", default="S-FFSD", help="Tag used in output filenames (default: S-FFSD)")
    ap.add_argument("--edge-per-trans", type=int, default=3, help="Edges per transaction within each group (default: 3)")
    ap.add_argument("--windows", default="2,3,5,15,20,50,100,150,200,300,864,2590,5100,10000,24000",
                    help="Comma-separated time windows for rolling features")
    ap.add_argument("--skip-fe", action="store_true", help="Skip feature engineering; use input as-is")
    ap.add_argument("--skip-neigh", action="store_true", help="Skip neighbor feature generation")
    ap.add_argument("--no-risk-features", action="store_true", help="Neighbor features use degree only (no labels) to avoid leakage")
    ap.add_argument("--force", action="store_true", help="Recompute even if outputs already exist")
    args = ap.parse_args()

    _ensure_dir(args.output_dir)
    outs = resolve_outputs(args.output_dir, args.tag)

    # caching
    have_neigh = args.skip_neigh or os.path.exists(outs.neigh_csv)
    if (not args.force) and os.path.exists(outs.neofull_csv) and os.path.exists(outs.graph_bin) and have_neigh:
        print(f"[cache] Outputs already exist in {args.output_dir}. Use --force to recompute.", flush=True)
        print(f"  {outs.neofull_csv}", flush=True)
        print(f"  {outs.graph_bin}", flush=True)
        if not args.skip_neigh:
            print(f"  {outs.neigh_csv}", flush=True)
        return 0

    t0 = _now()
    df = _load_csv(args.input_file)
    _validate_required(df)
    df = _coerce_types(df)

    # feature engineering
    if args.skip_fe:
        fe_df = df.copy()
        fe_elapsed = 0.0
        print("[fe] Skipped feature engineering; using input as-is.", flush=True)
    else:
        windows = [float(x.strip()) for x in args.windows.split(",") if x.strip()]
        t_fe0 = _now()
        fe_df = featmap_gen_fast_amount_only(df, windows=windows)
        fe_elapsed = _now() - t_fe0
        print(f"[fe] Generated rolling Amount features for {len(windows)} windows in {fe_elapsed:.2f}s", flush=True)

    fe_df.to_csv(outs.neofull_csv, index=False)
    print(f"[out] Wrote: {outs.neofull_csv}", flush=True)

    # graph
    t_g0 = _now()
    g = build_graph_from_groups(fe_df, edge_per_trans=int(args.edge_per_trans))
    g_elapsed = _now() - t_g0
    print(f"[graph] Built graph with {g.num_nodes()} nodes, {g.num_edges()} edges in {g_elapsed:.2f}s", flush=True)

    # encode categoricals for node features
    t_enc0 = _now()
    encoded = label_encode_inplace(fe_df, ["Source", "Target", "Location", "Type"])
    enc_elapsed = _now() - t_enc0
    print(f"[enc] Label-encoded categoricals in {enc_elapsed:.2f}s", flush=True)

    labels = encoded["Labels"].astype(int)
    feat_df = encoded.drop(columns=["Labels"])

    attach_node_data(g, feat_df, labels)

    dgl.data.utils.save_graphs(outs.graph_bin, [g])
    print(f"[out] Wrote: {outs.graph_bin}", flush=True)

    neigh_elapsed = 0.0
    if not args.skip_neigh:
        t_n0 = _now()
        if args.no_risk_features:
            neigh_df = compute_neighbor_features_degree_only(g, scale=True)
        else:
            neigh_df = compute_neighbor_features_fast(g, scale=True)
        neigh_df.to_csv(outs.neigh_csv, index=False)
        neigh_elapsed = _now() - t_n0
        print(f"[neigh] Wrote: {outs.neigh_csv} in {neigh_elapsed:.2f}s", flush=True)
    else:
        print("[neigh] Skipped neighbor feature generation.", flush=True)

    meta = {
        "input_file": os.path.abspath(args.input_file),
        "output_dir": os.path.abspath(args.output_dir),
        "tag": args.tag,
        "edge_per_trans": int(args.edge_per_trans),
        "windows": args.windows,
        "skip_fe": bool(args.skip_fe),
        "skip_neigh": bool(args.skip_neigh),
        "no_risk_features": bool(args.no_risk_features),
        "rows": int(len(df)),
        "cols_in": list(df.columns),
        "cols_out": int(len(fe_df.columns)),
        "label_distribution": {str(k): int(v) for k, v in pd.Series(df["Labels"]).value_counts().items()},
        "timing_seconds": {
            "feature_engineering": fe_elapsed,
            "graph_build": g_elapsed,
            "label_encode": enc_elapsed,
            "neighbor_features": neigh_elapsed,
            "total": _now() - t0,
        },
    }
    with open(outs.meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[out] Wrote: {outs.meta_json}", flush=True)

    print(f"\nDONE in {meta['timing_seconds']['total']:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
