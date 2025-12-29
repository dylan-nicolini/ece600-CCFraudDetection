#!/usr/bin/env python3
"""
IEEE-CIS v2 Preparation Pipeline

Generates:
  - ieee_split.parquet            (TransactionID, TransactionDT, split)
  - ieee_features_v2.parquet      (TransactionID, split, label + engineered features)
  - ieee_entities_v2.parquet      (TransactionID -> composite entity keys)
  - ieee_edges_v2.parquet         (transaction -> entity edges, gated + supernode-capped)

Key design constraints:
  - Time-based split using TransactionDT
  - No leakage: frequency encodings and entity frequency gating are computed on TRAIN split only
  - Deterministic supernode capping with seed
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Helpers
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_any(path: str) -> pd.DataFrame:
    """Read CSV or Parquet."""
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def write_parquet(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, index=False)


def canonicalize_series(s: pd.Series) -> pd.Series:
    """Canonicalize categorical text: lowercase, strip, map NaN/empty to '__NULL__'."""
    s2 = s.astype("string")
    s2 = s2.str.lower().str.strip()
    s2 = s2.fillna("__NULL__")
    s2 = s2.replace({"": "__NULL__"})
    return s2


def safe_concat(cols: List[pd.Series], sep: str = "_") -> pd.Series:
    """Concatenate columns into a composite entity key; nulls become '__NULL__'."""
    parts = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(c):
            parts.append(c.fillna(-999999).astype("int64", errors="ignore").astype("string"))
        else:
            parts.append(c.astype("string").fillna("__NULL__"))
    out = parts[0]
    for p in parts[1:]:
        out = out + sep + p
    out = out.fillna("__NULL__")
    return out


def device_family(device_info: pd.Series) -> pd.Series:
    """
    Coarse device family extractor. IEEE DeviceInfo often looks like 'SM-G930V' etc.
    We bucket by leading token before first non-alphanumeric run or first '-'/'/'/' '.
    """
    s = canonicalize_series(device_info)
    # split on common separators
    fam = s.str.split(r"[-/ ]+", n=1, regex=True).str[0]
    fam = fam.fillna("__NULL__")
    fam = fam.replace({"__null__": "__NULL__"})
    return fam


def assign_time_split(df: pd.DataFrame, dt_col: str, train_frac: float, valid_frac: float) -> pd.Series:
    """
    Sort by TransactionDT and assign split.
    train = first train_frac
    valid = next valid_frac
    test  = remainder
    """
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0,1)")
    if not (0.0 <= valid_frac < 1.0):
        raise ValueError("valid_frac must be in [0,1)")
    if train_frac + valid_frac >= 1.0:
        raise ValueError("train_frac + valid_frac must be < 1.0")

    order = df[dt_col].argsort(kind="mergesort")  # stable
    n = len(df)
    n_train = int(math.floor(n * train_frac))
    n_valid = int(math.floor(n * valid_frac))

    split = np.array(["test"] * n, dtype=object)
    split[order[:n_train]] = "train"
    split[order[n_train:n_train + n_valid]] = "valid"
    split[order[n_train + n_valid:]] = "test"
    return pd.Series(split, index=df.index, name="split")


def compute_freq_map(train_series: pd.Series) -> Dict[str, int]:
    counts = train_series.value_counts(dropna=False)
    # ensure string keys
    return {str(k): int(v) for k, v in counts.items()}


def apply_freq_encoding(series: pd.Series, freq_map: Dict[str, int]) -> Tuple[pd.Series, pd.Series]:
    """
    Produce freq and logfreq from a train-derived map.
    Unseen categories get 0.
    """
    s = series.astype("string").fillna("__NULL__")
    freq = s.map(lambda x: freq_map.get(str(x), 0)).astype("int32")
    logfreq = np.log1p(freq).astype("float32")
    return freq, logfreq


def cap_edges_per_entity(
    edges: pd.DataFrame,
    entity_col: str,
    max_degree: int,
    seed: int
) -> pd.DataFrame:
    """
    Cap number of transactions connected to any entity to max_degree.
    Deterministic using seed (random sample within each entity).
    """
    if max_degree <= 0:
        return edges

    rng = np.random.default_rng(seed)
    kept = []

    # groupby on entity id; sample if too many rows
    for ent, grp in edges.groupby(entity_col, sort=False):
        if len(grp) <= max_degree:
            kept.append(grp)
        else:
            idx = rng.choice(grp.index.to_numpy(), size=max_degree, replace=False)
            kept.append(grp.loc[idx])

    return pd.concat(kept, axis=0, ignore_index=True)


# ----------------------------
# Main Pipeline
# ----------------------------

@dataclass
class Config:
    tx_path: str
    id_path: Optional[str]
    out_dir: str
    label_col: str
    tx_id_col: str
    dt_col: str
    train_frac: float
    valid_frac: float
    seed: int
    min_entity_freq: int
    supernode_cap: int
    freq_encode_cols: List[str]


def load_and_merge(cfg: Config) -> pd.DataFrame:
    tx = read_any(cfg.tx_path)
    if cfg.id_path:
        ident = read_any(cfg.id_path)
        df = tx.merge(ident, on=cfg.tx_id_col, how="left")
    else:
        df = tx

    # basic checks
    for c in [cfg.tx_id_col, cfg.dt_col, cfg.label_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in merged dataframe.")
    return df


def build_features_and_entities(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      features_df: TransactionID, split, label, engineered numeric features
      entities_df: TransactionID, split, entity columns
    """
    # Split first (based on time)
    df = df.copy()
    df["split"] = assign_time_split(df, cfg.dt_col, cfg.train_frac, cfg.valid_frac)

    # Canonicalize frequency-encoded categoricals
    for col in cfg.freq_encode_cols:
        if col in df.columns:
            df[col] = canonicalize_series(df[col])
        else:
            # if user included a col that doesn't exist, skip quietly
            pass

    # Missingness indicators for kept columns (including numeric + categ in freq_encode_cols)
    # We do this for all columns that exist in df and are referenced, plus a small subset of numeric cols.
    # For practicality: create missing flags for freq_encode_cols and for all numeric columns in df.
    missing_flag_cols = []
    for col in cfg.freq_encode_cols:
        if col in df.columns:
            flag = f"isnull_{col}"
            df[flag] = df[col].isna().astype("int8")
            missing_flag_cols.append(flag)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # avoid massive width: keep only numeric cols that are not obvious IDs/labels
    drop_numeric_like = {cfg.label_col}
    base_numeric_cols = [c for c in numeric_cols if c not in drop_numeric_like]

    # Frequency + log-frequency encodings computed on TRAIN only (no leakage)
    train_mask = df["split"].eq("train")
    freq_feat_cols = []
    for col in cfg.freq_encode_cols:
        if col not in df.columns:
            continue
        freq_map = compute_freq_map(df.loc[train_mask, col])
        f, lf = apply_freq_encoding(df[col], freq_map)
        fcol = f"freq_{col}"
        lfcol = f"logfreq_{col}"
        df[fcol] = f
        df[lfcol] = lf
        freq_feat_cols.extend([fcol, lfcol])

    # ----------------------------
    # Composite Entities
    # ----------------------------
    # Card entity: card1..card6 if present
    card_parts = [c for c in ["card1", "card2", "card3", "card4", "card5", "card6"] if c in df.columns]
    if card_parts:
        df["card_entity"] = safe_concat([df[c] for c in card_parts], sep="_")
    else:
        df["card_entity"] = "__NULL__"

    # Email entities
    if "P_emaildomain" in df.columns:
        df["p_email_entity"] = canonicalize_series(df["P_emaildomain"])
    else:
        df["p_email_entity"] = "__NULL__"

    if "R_emaildomain" in df.columns:
        df["r_email_entity"] = canonicalize_series(df["R_emaildomain"])
    else:
        df["r_email_entity"] = "__NULL__"

    # Device entities
    # IEEE sometimes has DeviceType and DeviceInfo
    if "DeviceInfo" in df.columns:
        df["device_family"] = device_family(df["DeviceInfo"])
    else:
        df["device_family"] = "__NULL__"

    if "DeviceType" in df.columns:
        df["device_type_norm"] = canonicalize_series(df["DeviceType"])
    else:
        df["device_type_norm"] = "__NULL__"

    df["device_entity"] = df["device_type_norm"].astype("string") + "_" + df["device_family"].astype("string")

    # Address entity
    addr_parts = [c for c in ["addr1", "addr2"] if c in df.columns]
    if addr_parts:
        df["addr_entity"] = safe_concat([df[c] for c in addr_parts], sep="_")
    else:
        df["addr_entity"] = "__NULL__"

    # Build entities_df (minimal cols)
    entities_cols = [
        cfg.tx_id_col, "split", cfg.dt_col,
        "card_entity", "p_email_entity", "r_email_entity", "device_entity", "addr_entity"
    ]
    entities_cols = [c for c in entities_cols if c in df.columns]
    entities_df = df[entities_cols].copy()

    # Build features_df
    # Keep: label + split + TransactionDT + missing flags + numeric base + freq features
    keep_cols = [cfg.tx_id_col, "split", cfg.dt_col, cfg.label_col]
    # To avoid huge width: include numeric base cols but drop TransactionDT/label/id duplicates
    base_numeric_keep = [c for c in base_numeric_cols if c not in {cfg.tx_id_col, cfg.dt_col, cfg.label_col}]
    features_cols = keep_cols + missing_flag_cols + freq_feat_cols + base_numeric_keep
    features_cols = [c for c in features_cols if c in df.columns]
    features_df = df[features_cols].copy()

    return features_df, entities_df


def build_edges(entities_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Create transaction -> entity edges from entities_df,
    with train-only entity frequency gating and supernode capping.
    """
    # We'll create edges for each entity type
    entity_specs = [
        ("card_entity", "card"),
        ("p_email_entity", "p_email"),
        ("r_email_entity", "r_email"),
        ("device_entity", "device"),
        ("addr_entity", "addr"),
    ]

    edges_all = []
    tx_id = cfg.tx_id_col

    # Frequency gating must be computed on TRAIN only (no leakage)
    train_entities = entities_df[entities_df["split"].eq("train")].copy()

    for col, etype in entity_specs:
        if col not in entities_df.columns:
            continue

        # Compute train frequencies for this entity
        ent_counts = train_entities[col].astype("string").fillna("__NULL__").value_counts(dropna=False)
        # Entities that pass min freq
        keep_ents = set(ent_counts[ent_counts >= cfg.min_entity_freq].index.astype("string").tolist())

        tmp = entities_df[[tx_id, "split", col]].copy()
        tmp[col] = tmp[col].astype("string").fillna("__NULL__")

        # Apply gating: keep only edges where entity in keep_ents and not NULL
        gated = tmp[tmp[col].isin(keep_ents) & ~tmp[col].eq("__NULL__")].copy()
        gated.rename(columns={col: "entity_id"}, inplace=True)
        gated["entity_type"] = etype

        edges_all.append(gated[[tx_id, "split", "entity_type", "entity_id"]])

    if not edges_all:
        raise ValueError("No entity columns found to build edges from.")

    edges = pd.concat(edges_all, axis=0, ignore_index=True)

    # Supernode capping: cap per (entity_type, entity_id)
    if cfg.supernode_cap and cfg.supernode_cap > 0:
        # Cap separately for each entity_type to keep behavior stable
        capped_chunks = []
        for etype, grp in edges.groupby("entity_type", sort=False):
            grp2 = cap_edges_per_entity(grp, entity_col="entity_id", max_degree=cfg.supernode_cap, seed=cfg.seed)
            capped_chunks.append(grp2)
        edges = pd.concat(capped_chunks, axis=0, ignore_index=True)

    return edges


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tx", required=True, help="Path to train_transaction (csv or parquet)")
    ap.add_argument("--id", default=None, help="Path to train_identity (csv or parquet). Optional.")
    ap.add_argument("--out", required=True, help="Output directory for v2 artifacts")
    ap.add_argument("--label-col", default="isFraud", help="Label column name (default: isFraud)")
    ap.add_argument("--tx-id-col", default="TransactionID", help="Transaction ID column (default: TransactionID)")
    ap.add_argument("--dt-col", default="TransactionDT", help="Time column for split (default: TransactionDT)")
    ap.add_argument("--train-frac", type=float, default=0.70, help="Train fraction by time (default: 0.70)")
    ap.add_argument("--valid-frac", type=float, default=0.15, help="Valid fraction by time (default: 0.15)")
    ap.add_argument("--seed", type=int, default=2023, help="Random seed for capping (default: 2023)")
    ap.add_argument("--min-entity-freq", type=int, default=5, help="Min entity frequency for edges (default: 5)")
    ap.add_argument("--supernode-cap", type=int, default=2000, help="Max edges per entity (default: 2000)")
    ap.add_argument(
        "--freq-cols",
        default="Type,M1,M2,M3,M4,M5,M6,M7,M8,M9,P_emaildomain,R_emaildomain,card4,card6,DeviceType,DeviceInfo",
        help="Comma-separated list of categorical columns to canonicalize + freq-encode"
    )

    args = ap.parse_args()

    cfg = Config(
        tx_path=args.tx,
        id_path=args.id,
        out_dir=args.out,
        label_col=args.label_col,
        tx_id_col=args.tx_id_col,
        dt_col=args.dt_col,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        seed=args.seed,
        min_entity_freq=args.min_entity_freq,
        supernode_cap=args.supernode_cap,
        freq_encode_cols=[c.strip() for c in args.freq_cols.split(",") if c.strip()],
    )

    ensure_dir(cfg.out_dir)

    print("Loading and merging IEEE files...")
    df = load_and_merge(cfg)
    print(f"Merged shape: {df.shape}")

    print("Building v2 features + entities (time split, canonicalize, freq enc, composite entities)...")
    features_df, entities_df = build_features_and_entities(df, cfg)

    # Split output
    split_df = features_df[[cfg.tx_id_col, cfg.dt_col, "split"]].copy()
    split_path = os.path.join(cfg.out_dir, "ieee_split.parquet")
    write_parquet(split_df, split_path)
    print(f"Wrote: {split_path}  shape={split_df.shape}")

    # Features output
    feat_path = os.path.join(cfg.out_dir, "ieee_features_v2.parquet")
    write_parquet(features_df, feat_path)
    print(f"Wrote: {feat_path}  shape={features_df.shape}")

    # Entities output
    ent_path = os.path.join(cfg.out_dir, "ieee_entities_v2.parquet")
    write_parquet(entities_df, ent_path)
    print(f"Wrote: {ent_path}  shape={entities_df.shape}")

    # Edge output
    print("Building v2 edges (train-only gating + supernode cap)...")
    edges_df = build_edges(entities_df, cfg)
    edges_path = os.path.join(cfg.out_dir, "ieee_edges_v2.parquet")
    write_parquet(edges_df, edges_path)
    print(f"Wrote: {edges_path}  shape={edges_df.shape}")

    # Quick sanity stats (useful for your thesis)
    print("\n--- Sanity Stats ---")
    print(split_df["split"].value_counts())
    fraud_rate = features_df.groupby("split")[cfg.label_col].mean()
    print("\nFraud rate by split:")
    print(fraud_rate)

    print("\nEdge counts by type:")
    print(edges_df["entity_type"].value_counts())

    # Degree-1 proxy: entity frequency distribution
    print("\nEntity degree summary (post-cap) by type:")
    deg = edges_df.groupby(["entity_type", "entity_id"])[cfg.tx_id_col].count().rename("deg")
    print(deg.groupby(level=0).describe()[["count", "mean", "50%", "max"]])

    print("\nDone. Next step: rebuild your graph using ieee_edges_v2 + ieee_features_v2.")


if __name__ == "__main__":
    main()
