#!/usr/bin/env python3
"""
prep_ieee_v2.py

IEEE-CIS Fraud Detection "v2" prep pipeline for graph-based models.

Goal
----
Build an IEEE transaction graph that is entity-driven (card/address/email/device),
with supernode controls and reproducible artifacts that you can feed into GTAN/RGTAN.

This script:
  1) Loads train_transaction.csv (+ optional train_identity.csv)
  2) Builds composite entity keys:
       - card_entity   = card1|card2|card3|card4|card5|card6
       - addr_entity   = addr1|addr2
       - email_entity  = P_emaildomain|R_emaildomain
       - device_entity = DeviceType|DeviceInfo (from identity if provided)
  3) Computes entity frequencies (counts)
  4) Applies gating:
       - keep entities with count >= min_freq and <= max_degree
  5) Builds txn<->txn edges by chaining within each entity group sorted by TransactionDT:
       - each txn connects to up to neighbor_k next transactions within the entity group
     This avoids clique explosion and reduces supernode smear.
  6) Writes:
       - ieee_v2_train.csv          (transaction features + label)
       - graph_edges.csv            (src,dst,etype)
       - entity_stats.csv           (entity_type, count histogram + kept counts)
       - gating_report.json         (parameters + totals)
       - graph-IEEE-v2.bin          (optional, if dgl is available and --save-dgl-bin)

Recommended starting params:
  --min-freq 3 --max-degree 500 --neighbor-k 3

Notes
-----
- This script is designed to be "thesis friendly": it produces a gating report and stats.
- For IEEE scale (~590k rows), this should run on a modest EC2 instance, but edges can be large.
- It builds edges using "next-k chaining" within entity groups (not full clique). This is intentional.

Example
-------
python3 prep_ieee_v2.py \
  --tx-csv ~/ece600-CCFraudDetection/antifraud/data/ieee_modified/_extract/train_transaction.csv \
  --id-csv ~/ece600-CCFraudDetection/antifraud/data/ieee_modified/_extract/train_identity.csv \
  --out-dir ~/ece600-CCFraudDetection/antifraud/data/ieee_v2 \
  --min-freq 3 \
  --max-degree 500 \
  --neighbor-k 3 \
  --save-dgl-bin
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Utilities
# -----------------------------

def _ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def _log(msg: str) -> None:
    print(f"[prep_ieee_v2] {msg}", flush=True)


def _safe_str(x) -> str:
    if pd.isna(x):
        return "NA"
    s = str(x).strip()
    return s if s else "NA"


def _join_key(parts: Iterable) -> str:
    # stable composite key
    return "|".join(_safe_str(p) for p in parts)


def _try_import_dgl():
    try:
        import dgl  # type: ignore
        import torch  # type: ignore
        return dgl, torch
    except Exception:
        return None, None


@dataclass
class GatingParams:
    min_freq: int
    max_degree: int
    neighbor_k: int
    entity_types: List[str]
    chunksize: int
    save_dgl_bin: bool


@dataclass
class GatingTotals:
    n_rows: int
    fraud_rate: float
    edges_written: int
    unique_entities_total: Dict[str, int]
    unique_entities_kept: Dict[str, int]
    entities_dropped_low_freq: Dict[str, int]
    entities_dropped_supernode: Dict[str, int]


# -----------------------------
# Entity logic
# -----------------------------

ENTITY_COLS_TX = {
    "card": ["card1", "card2", "card3", "card4", "card5", "card6"],
    "addr": ["addr1", "addr2"],
    "email": ["P_emaildomain", "R_emaildomain"],
}
ENTITY_COLS_ID = {
    "device": ["DeviceType", "DeviceInfo"],
}

# minimal columns we want for features
BASE_TX_COLS = ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]


def _required_columns(entity_types: List[str], have_identity: bool) -> Tuple[List[str], List[str]]:
    """Return (tx_cols, id_cols) needed."""
    tx_cols = set(BASE_TX_COLS)
    id_cols = set(["TransactionID"])

    for et in entity_types:
        if et in ENTITY_COLS_TX:
            tx_cols.update(ENTITY_COLS_TX[et])
        elif et in ENTITY_COLS_ID:
            # entity derived from identity file
            if have_identity:
                id_cols.update(ENTITY_COLS_ID[et])

    return sorted(tx_cols), sorted(id_cols)


def _build_entity_series(df: pd.DataFrame, entity_type: str) -> pd.Series:
    """Create composite entity key column for a given entity type."""
    if entity_type in ENTITY_COLS_TX:
        cols = ENTITY_COLS_TX[entity_type]
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        return df[cols].astype("object").apply(lambda r: _join_key(r.values.tolist()), axis=1)
    if entity_type in ENTITY_COLS_ID:
        cols = ENTITY_COLS_ID[entity_type]
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        return df[cols].astype("object").apply(lambda r: _join_key(r.values.tolist()), axis=1)
    raise ValueError(f"Unknown entity type: {entity_type}")


def _is_all_na_key(key: str) -> bool:
    # composite key like "NA|NA|NA"
    return all(part == "NA" for part in key.split("|"))


# -----------------------------
# Main pipeline
# -----------------------------

def compute_entity_counts(
    tx_csv: str,
    id_csv: Optional[str],
    entity_types: List[str],
    chunksize: int,
) -> Tuple[Counter, Dict[str, Counter], int, float]:
    """
    First pass over data:
      - compute entity counts per entity type
      - compute row count and fraud rate
    Returns: (global_counter_unused, counts_by_type, n_rows, fraud_rate)
    """
    have_identity = bool(id_csv and os.path.exists(id_csv))
    tx_cols, id_cols = _required_columns(entity_types, have_identity)

    _log(f"Pass1: reading tx cols={len(tx_cols)} identity={have_identity}")
    counts_by_type: Dict[str, Counter] = {et: Counter() for et in entity_types}

    n_rows = 0
    fraud_sum = 0

    if have_identity:
        _log(f"Pass1: loading identity subset (cols={len(id_cols)}) into memory: {id_csv}")
        # identity is smaller; load as dataframe keyed by TransactionID
        id_df = pd.read_csv(id_csv, usecols=id_cols, low_memory=False)
        # ensure unique TransactionID
        id_df = id_df.drop_duplicates(subset=["TransactionID"])
        id_df = id_df.set_index("TransactionID")
    else:
        id_df = None

    for chunk_idx, tx_chunk in enumerate(pd.read_csv(tx_csv, usecols=tx_cols, chunksize=chunksize, low_memory=False)):
        # Ensure expected cols exist
        for c in BASE_TX_COLS:
            if c not in tx_chunk.columns:
                raise RuntimeError(f"Missing required column {c} in transaction chunk")

        if id_df is not None:
            tx_chunk = tx_chunk.join(id_df, on="TransactionID", how="left")

        n_rows += len(tx_chunk)
        # fraud label
        if "isFraud" in tx_chunk.columns:
            fraud_sum += int(tx_chunk["isFraud"].fillna(0).sum())

        # compute keys and update counts
        for et in entity_types:
            keys = _build_entity_series(tx_chunk, et)
            # drop all-NA keys (no signal)
            keys = keys[~keys.map(_is_all_na_key)]
            counts_by_type[et].update(keys.tolist())

        if (chunk_idx + 1) % 10 == 0:
            _log(f"Pass1: processed chunks={chunk_idx+1} rows={n_rows:,}")

    fraud_rate = (fraud_sum / n_rows) if n_rows else 0.0
    _log(f"Pass1 done: n_rows={n_rows:,} fraud_sum={fraud_sum:,} fraud_rate={fraud_rate:.6f}")
    return Counter(), counts_by_type, n_rows, fraud_rate


def build_edges_and_features(
    tx_csv: str,
    id_csv: Optional[str],
    out_dir: str,
    entity_types: List[str],
    counts_by_type: Dict[str, Counter],
    min_freq: int,
    max_degree: int,
    neighbor_k: int,
    chunksize: int,
) -> Tuple[str, str, GatingTotals]:
    """
    Second pass:
      - build per-entity lists of (TransactionDT, node_id)
      - write prepared feature CSV (single file)
      - after pass: generate edges (next-k chaining) and write to graph_edges.csv
    """
    os.makedirs(out_dir, exist_ok=True)
    have_identity = bool(id_csv and os.path.exists(id_csv))
    tx_cols, id_cols = _required_columns(entity_types, have_identity)

    # Decide which entities to keep based on counts
    kept_entities: Dict[str, set] = {}
    dropped_low: Dict[str, int] = {}
    dropped_high: Dict[str, int] = {}
    unique_total: Dict[str, int] = {}
    unique_kept: Dict[str, int] = {}

    for et in entity_types:
        c = counts_by_type[et]
        unique_total[et] = len(c)
        kept = {k for k, v in c.items() if v >= min_freq and v <= max_degree and not _is_all_na_key(k)}
        kept_entities[et] = kept
        unique_kept[et] = len(kept)
        dropped_low[et] = sum(1 for _, v in c.items() if v < min_freq)
        dropped_high[et] = sum(1 for _, v in c.items() if v > max_degree)

    _log("Pass2: kept entities by type:")
    for et in entity_types:
        _log(f"  {et}: kept={unique_kept[et]:,} of total={unique_total[et]:,} "
             f"(drop_low<{min_freq}={dropped_low[et]:,}, drop_super>{max_degree}={dropped_high[et]:,})")

    # Load identity map if present
    if have_identity:
        _log(f"Pass2: loading identity subset into memory: {id_csv}")
        id_df = pd.read_csv(id_csv, usecols=id_cols, low_memory=False)
        id_df = id_df.drop_duplicates(subset=["TransactionID"]).set_index("TransactionID")
    else:
        id_df = None

    # We'll write a prepared training CSV with:
    #   node_id, TransactionID, TransactionDT, TransactionAmt, isFraud,
    #   plus entity frequency features per type and missingness counts.
    out_feat_csv = os.path.join(out_dir, "ieee_v2_train.csv")
    _log(f"Pass2: writing features -> {out_feat_csv}")

    # We need stable node ids: row order across the entire CSV stream.
    node_id_offset = 0

    # For edge construction later: entity -> list of (TransactionDT, node_id)
    # Only for KEPT entities.
    ent_to_list: Dict[str, Dict[str, List[Tuple[int, int]]]] = {et: defaultdict(list) for et in entity_types}

    # Create feature file writer
    with open(out_feat_csv, "w", newline="") as f_out:
        writer = None

        for chunk_idx, tx_chunk in enumerate(pd.read_csv(tx_csv, usecols=tx_cols, chunksize=chunksize, low_memory=False)):
            if id_df is not None:
                tx_chunk = tx_chunk.join(id_df, on="TransactionID", how="left")

            # assign node ids
            n = len(tx_chunk)
            node_ids = np.arange(node_id_offset, node_id_offset + n, dtype=np.int64)
            node_id_offset += n

            # transaction dt and amount
            # TransactionDT is numeric seconds-from-start in IEEE; keep as int64
            tx_dt = tx_chunk.get("TransactionDT", pd.Series([0]*n)).fillna(0).astype(np.int64).to_numpy()
            tx_amt = tx_chunk.get("TransactionAmt", pd.Series([0.0]*n)).fillna(0.0).astype(np.float32).to_numpy()
            labels = tx_chunk.get("isFraud", pd.Series([0]*n)).fillna(0).astype(np.int64).to_numpy()

            # entity keys + degree features
            feat_rows = {
                "node_id": node_ids,
                "TransactionID": tx_chunk["TransactionID"].astype(np.int64).to_numpy(),
                "TransactionDT": tx_dt,
                "TransactionAmt": tx_amt,
                "Labels": labels,
            }

            # Missingness count across entity raw columns used
            used_cols = []
            for et in entity_types:
                if et in ENTITY_COLS_TX:
                    used_cols += ENTITY_COLS_TX[et]
                elif et in ENTITY_COLS_ID:
                    used_cols += ENTITY_COLS_ID[et]
            used_cols = [c for c in used_cols if c in tx_chunk.columns]
            if used_cols:
                miss_cnt = tx_chunk[used_cols].isna().sum(axis=1).astype(np.int16).to_numpy()
            else:
                miss_cnt = np.zeros(n, dtype=np.int16)
            feat_rows["missing_count"] = miss_cnt

            # Create per-entity key, per-entity frequency (and logfreq)
            for et in entity_types:
                keys = _build_entity_series(tx_chunk, et)
                # freq lookup
                c = counts_by_type[et]
                freqs = keys.map(lambda k: c.get(k, 0)).astype(np.int32).to_numpy()
                # log1p(freq)
                logfreq = np.log1p(freqs.astype(np.float32))
                feat_rows[f"{et}_freq"] = freqs
                feat_rows[f"{et}_logfreq"] = logfreq

                # For edge lists: keep only eligible keys
                kept = kept_entities[et]
                for i in range(n):
                    k = keys.iat[i]
                    if k in kept:
                        ent_to_list[et][k].append((int(tx_dt[i]), int(node_ids[i])))

            # Write chunk to CSV
            df_out = pd.DataFrame(feat_rows)
            if writer is None:
                writer = csv.DictWriter(f_out, fieldnames=list(df_out.columns))
                writer.writeheader()
            for row in df_out.to_dict(orient="records"):
                writer.writerow(row)

            if (chunk_idx + 1) % 10 == 0:
                _log(f"Pass2: processed chunks={chunk_idx+1} nodes_so_far={node_id_offset:,}")

    _log(f"Pass2 done: total_nodes={node_id_offset:,}")

    # Now build edges by iterating entity groups and chaining next-k by time
    out_edges_csv = os.path.join(out_dir, "graph_edges.csv")
    _log(f"Edges: writing -> {out_edges_csv} (neighbor_k={neighbor_k})")

    edges_written = 0
    with open(out_edges_csv, "w", newline="") as f_edges:
        w = csv.writer(f_edges)
        w.writerow(["src", "dst", "etype"])  # etype = entity type name

        for et in entity_types:
            groups = ent_to_list[et]
            _log(f"Edges: processing entity_type={et} groups={len(groups):,}")
            # process each entity group
            for key, pairs in groups.items():
                if len(pairs) < 2:
                    continue
                # sort by TransactionDT (then node id for stability)
                pairs.sort(key=lambda x: (x[0], x[1]))
                nodes = [nid for _, nid in pairs]
                # chain next-k
                for i in range(len(nodes)):
                    u = nodes[i]
                    # connect to next neighbor_k nodes
                    for j in range(1, neighbor_k + 1):
                        if i + j >= len(nodes):
                            break
                        v = nodes[i + j]
                        # write undirected edges (both directions) for message passing
                        w.writerow([u, v, et])
                        w.writerow([v, u, et])
                        edges_written += 2

    totals = GatingTotals(
        n_rows=node_id_offset,
        fraud_rate=float(pd.read_csv(out_feat_csv, usecols=["Labels"])["Labels"].mean()),
        edges_written=edges_written,
        unique_entities_total=unique_total,
        unique_entities_kept=unique_kept,
        entities_dropped_low_freq=dropped_low,
        entities_dropped_supernode=dropped_high,
    )

    _log(f"Edges done: edges_written={edges_written:,}")
    return out_feat_csv, out_edges_csv, totals


def save_entity_stats(out_dir: str, counts_by_type: Dict[str, Counter], min_freq: int, max_degree: int) -> str:
    """
    Save entity stats:
      - per entity type: total unique, kept unique, histogram buckets
    """
    out_stats = os.path.join(out_dir, "entity_stats.csv")
    rows = []
    for et, c in counts_by_type.items():
        vals = np.array(list(c.values()), dtype=np.int64) if c else np.array([], dtype=np.int64)
        total = int(len(vals))
        kept = int(np.sum((vals >= min_freq) & (vals <= max_degree))) if total else 0

        # buckets
        buckets = [
            ("lt_min", int(np.sum(vals < min_freq))) if total else 0,
            ("min_to_5", int(np.sum((vals >= min_freq) & (vals <= 5)))) if total else 0,
            ("6_to_20", int(np.sum((vals >= 6) & (vals <= 20)))) if total else 0,
            ("21_to_100", int(np.sum((vals >= 21) & (vals <= 100)))) if total else 0,
            ("101_to_max", int(np.sum((vals >= 101) & (vals <= max_degree)))) if total else 0,
            ("gt_max", int(np.sum(vals > max_degree))) if total else 0,
        ]
        for bname, bcount in buckets:
            rows.append({
                "entity_type": et,
                "bucket": bname,
                "count": bcount,
                "unique_total": total,
                "unique_kept": kept,
                "min_freq": min_freq,
                "max_degree": max_degree,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_stats, index=False)
    _log(f"Saved entity stats -> {out_stats}")
    return out_stats


def maybe_save_dgl_graph(
    out_dir: str,
    n_nodes: int,
    edges_csv: str,
    feat_csv: str,
    save: bool,
) -> Optional[str]:
    """
    Optional: save a DGLGraph bin:
      - homogeneous graph with n_nodes nodes
      - edges from edges_csv (src,dst)
      - node features from feat_csv (excluding IDs)
    """
    if not save:
        return None

    dgl, torch = _try_import_dgl()
    if dgl is None or torch is None:
        _log("DGL not available; skipping --save-dgl-bin output.")
        return None

    _log("Building DGL graph from edges CSV (this can take a bit)...")
    t0 = time.time()

    # read edges
    edf = pd.read_csv(edges_csv, usecols=["src", "dst"])
    src = torch.tensor(edf["src"].to_numpy(dtype=np.int64))
    dst = torch.tensor(edf["dst"].to_numpy(dtype=np.int64))

    g = dgl.graph((src, dst), num_nodes=int(n_nodes))

    # read features
    fdf = pd.read_csv(feat_csv)
    # drop non-feature columns
    drop_cols = {"node_id", "TransactionID"}
    keep_cols = [c for c in fdf.columns if c not in drop_cols]
    # Ensure float32 features where appropriate
    # Labels should be int64 for masks/labels
    labels = torch.tensor(fdf["Labels"].to_numpy(dtype=np.int64))
    g.ndata["label"] = labels

    # Numeric features: everything except Labels
    feat_cols = [c for c in keep_cols if c != "Labels"]
    feats = fdf[feat_cols].to_numpy()
    # make float32
    feats = feats.astype(np.float32, copy=False)
    g.ndata["feat"] = torch.from_numpy(feats)

    out_bin = os.path.join(out_dir, "graph-IEEE-v2.bin")
    dgl.data.utils.save_graphs(out_bin, [g])
    _log(f"Saved DGL graph -> {out_bin} (elapsed={time.time()-t0:.1f}s)")

    return out_bin


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tx-csv", required=True, help="Path to IEEE train_transaction.csv (must include isFraud)")
    p.add_argument("--id-csv", default=None, help="Optional path to IEEE train_identity.csv")
    p.add_argument("--out-dir", required=True, help="Output directory for v2 artifacts")
    p.add_argument("--min-freq", type=int, default=3, help="Minimum entity frequency to keep")
    p.add_argument("--max-degree", type=int, default=500, help="Maximum entity degree to keep (supernode cap)")
    p.add_argument("--neighbor-k", type=int, default=3, help="Next-k chaining within each entity group")
    p.add_argument("--entity-types", default="card,addr,email,device",
                   help="Comma-separated subset of: card,addr,email,device")
    p.add_argument("--chunksize", type=int, default=200_000, help="CSV chunksize for streaming reads")
    p.add_argument("--save-dgl-bin", action="store_true", help="Also write graph-IEEE-v2.bin if DGL is installed")
    args = p.parse_args()

    entity_types = [x.strip() for x in args.entity_types.split(",") if x.strip()]
    for et in entity_types:
        if et not in set(ENTITY_COLS_TX.keys()) | set(ENTITY_COLS_ID.keys()):
            raise SystemExit(f"Unknown entity type '{et}'. Valid: card,addr,email,device")

    if not os.path.exists(args.tx_csv):
        raise SystemExit(f"tx-csv not found: {args.tx_csv}")
    if args.id_csv and not os.path.exists(args.id_csv):
        _log(f"WARNING: id-csv not found, ignoring: {args.id_csv}")
        args.id_csv = None

    os.makedirs(args.out_dir, exist_ok=True)

    params = GatingParams(
        min_freq=args.min_freq,
        max_degree=args.max_degree,
        neighbor_k=args.neighbor_k,
        entity_types=entity_types,
        chunksize=args.chunksize,
        save_dgl_bin=bool(args.save_dgl_bin),
    )

    _log(f"Start {_ts()}")
    _log(f"Params: {asdict(params)}")
    _log(f"tx_csv={args.tx_csv}")
    _log(f"id_csv={args.id_csv}")

    # Pass1: counts
    _, counts_by_type, n_rows, fraud_rate = compute_entity_counts(
        tx_csv=args.tx_csv,
        id_csv=args.id_csv,
        entity_types=entity_types,
        chunksize=args.chunksize,
    )

    # Save stats
    stats_csv = save_entity_stats(args.out_dir, counts_by_type, args.min_freq, args.max_degree)

    # Pass2: features + edges
    feat_csv, edges_csv, totals = build_edges_and_features(
        tx_csv=args.tx_csv,
        id_csv=args.id_csv,
        out_dir=args.out_dir,
        entity_types=entity_types,
        counts_by_type=counts_by_type,
        min_freq=args.min_freq,
        max_degree=args.max_degree,
        neighbor_k=args.neighbor_k,
        chunksize=args.chunksize,
    )

    # Optional DGL graph
    out_bin = maybe_save_dgl_graph(
        out_dir=args.out_dir,
        n_nodes=totals.n_rows,
        edges_csv=edges_csv,
        feat_csv=feat_csv,
        save=bool(args.save_dgl_bin),
    )

    # Gating report
    report = {
        "timestamp_utc": _ts(),
        "tx_csv": args.tx_csv,
        "id_csv": args.id_csv,
        "out_dir": args.out_dir,
        "params": asdict(params),
        "pass1": {
            "n_rows": n_rows,
            "fraud_rate": fraud_rate,
            "unique_entities_total": {k: len(v) for k, v in counts_by_type.items()},
        },
        "outputs": {
            "features_csv": feat_csv,
            "edges_csv": edges_csv,
            "entity_stats_csv": stats_csv,
            "dgl_graph_bin": out_bin,
        },
        "totals": asdict(totals),
    }
    out_report = os.path.join(args.out_dir, "gating_report.json")
    with open(out_report, "w") as f:
        json.dump(report, f, indent=2)
    _log(f"Saved gating report -> {out_report}")
    _log(f"Done {_ts()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
