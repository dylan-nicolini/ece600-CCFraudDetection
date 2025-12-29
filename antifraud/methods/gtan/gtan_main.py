# ~/ece600-CCFraudDetection/antifraud/methods/gtan/gtan_main.py
import os
import math
import inspect
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from comet_ml import Experiment

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

import dgl
from dgl.dataloading import MultiLayerFullNeighborSampler

try:
    from dgl.dataloading import NodeDataLoader
except ImportError:
    from dgl.dataloading import DataLoader as NodeDataLoader

from torch.optim.lr_scheduler import MultiStepLR

from .gtan_model import GraphAttnModel
from .gtan_lpa import load_lpa_subtensor  # keep for repo compatibility


# -------------------------
# Helpers
# -------------------------

def _force_dropout(args: dict):
    """
    Ensure dropout is ALWAYS a 2-item list [input_dropout, output_dropout].
    """
    drop = args.get("dropout", args.get("drop", None))
    if drop is None:
        drop = [0.2, 0.1]
    if isinstance(drop, (int, float)):
        drop = [float(drop), float(drop)]
    if isinstance(drop, (list, tuple)):
        drop = list(drop)
        if len(drop) == 1:
            drop = [float(drop[0]), float(drop[0])]
        elif len(drop) >= 2:
            drop = [float(drop[0]), float(drop[1])]
        else:
            drop = [0.2, 0.1]
    else:
        drop = [0.2, 0.1]

    args["dropout"] = drop
    return drop


def _force_activation(args: dict):
    """
    Return a torch activation function object (callable), default ELU.
    """
    act = args.get("activation", args.get("act", "elu"))
    if act is None:
        act = "elu"

    if isinstance(act, str):
        s = act.lower().strip()
        if s in ("relu",):
            return torch.relu
        if s in ("tanh",):
            return torch.tanh
        if s in ("gelu",):
            return torch.nn.functional.gelu
        if s in ("leakyrelu", "leaky_relu"):
            return torch.nn.functional.leaky_relu
        # default
        return torch.nn.functional.elu

    # Already callable
    return act


def _build_graphattn_model(input_dim: int, args: dict) -> nn.Module:
    """
    Build a GraphAttnModel with the constructor signature used in this repo.

    Important:
      - `drop` must be a 2-item list: [input_dropout, output_dropout]
      - `heads` must be a list (len == n_layers), even if all 1s
      - we default to n2v_feat=False unless explicitly enabled with the needed data
    """
    drop = _force_dropout(args)
    activation = _force_activation(args)
    n_layers = int(args.get("n_layers", 2))
    hid_dim = int(args.get("hid_dim", 128))

    heads = args.get("heads")
    if heads is None:
        # Default to 1 head per layer unless the user specifies otherwise
        heads = [1] * n_layers
    elif isinstance(heads, int):
        heads = [heads] * n_layers

    # Avoid silently enabling N2V/TransEmbedding unless the inputs are present.
    n2v_feat = bool(args.get("n2v_feat", False))
    ref_df = args.get("ref_df", None)
    cat_features = args.get("cat_features", None)
    if n2v_feat and ref_df is None:
        # ref_df is required for TransEmbedding in this fork; disable if not provided.
        n2v_feat = False

    print(
        f"[GTAN MODEL] build: input_dim={input_dim} hid_dim={hid_dim} n_layers={n_layers} "
        f"heads={heads} dropout={drop} activation={getattr(activation, '__name__', str(activation))} n2v_feat={n2v_feat}",
        flush=True,
    )

    return GraphAttnModel(
        in_feats=input_dim,
        hidden_dim=hid_dim,
        n_layers=n_layers,
        n_classes=2,
        heads=heads,
        activation=activation,
        gated=bool(args.get("gated", True)),
        layer_norm=bool(args.get("layer_norm", True)),
        post_proc=bool(args.get("post_proc", True)),
        n2v_feat=n2v_feat,
        drop=drop,
        ref_df=ref_df,
        cat_features=cat_features,
        device=args.get("device", "cpu"),
    )


# -------------------------
# Training entrypoint
# -------------------------

def gtan_main(feat_df, graph, train_idx, test_idx, labels, args, cat_features=None, experiment: Experiment = None):
    if cat_features is None:
        cat_features = []

    # HARDEN these immediately so nothing downstream sees None
    _force_dropout(args)
    args["activation"] = _force_activation(args)

    device = args["device"]
    graph = graph.to(device)

    oof_predictions = torch.from_numpy(np.zeros([len(feat_df), 2])).float().to(device)
    # We compute test predictions fold-by-fold and average them (test_idx is held-out from CV folds).
    test_pred_sum = torch.zeros(len(feat_df), dtype=torch.float32, device=device)

    kfold = StratifiedKFold(n_splits=args["n_fold"], shuffle=True, random_state=args["seed"])
    y_target = labels.iloc[train_idx].values

    num_feat = torch.from_numpy(feat_df.values).float().to(device)

    cat_feat = {}
    for col in (cat_features or []):
        cat_feat[col] = torch.from_numpy(feat_df[col].values).long().to(device)

    labels_t = torch.from_numpy(labels.values).long().to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)

    for fold, (trn_i, val_i) in enumerate(kfold.split(train_idx, y_target)):
        print(f"Training fold {fold + 1}", flush=True)

        trn_idx = np.array(train_idx)[trn_i]
        val_idx = np.array(train_idx)[val_i]

        model = _build_graphattn_model(num_feat.shape[1], args).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["wd"])
        scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.5)

        sampler = MultiLayerFullNeighborSampler(args["n_layers"])

        train_loader = NodeDataLoader(
            graph, trn_idx, sampler,
            batch_size=args["batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=0
        )
        valid_loader = NodeDataLoader(
            graph, val_idx, sampler,
            batch_size=args["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=0
        )

        best_ap = -1.0
        best_state = None
        patience = 0
        early_stopping = int(args.get("early_stopping", 10))

        for epoch in range(int(args["max_epochs"])):
            model.train()
            train_losses = []
            y_true_epoch, y_score_epoch = [], []

            for step, (input_nodes, output_nodes, blocks) in enumerate(train_loader):
                blocks = [b.to(device) for b in blocks]
                input_nodes = input_nodes.to(device)
                seeds = output_nodes.to(device)

                batch_num_feat, _, batch_labels, propagate_labels = load_lpa_subtensor(
                    num_feat, cat_feat, labels_t, seeds, input_nodes, device
                )

                logits = model(blocks, batch_num_feat, propagate_labels)
                loss = loss_fn(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(float(loss.item()))
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                y_score_epoch.extend(probs.tolist())
                y_true_epoch.extend(batch_labels.detach().cpu().numpy().tolist())

                if step % 10 == 0:
                    try:
                        ap_now = average_precision_score(y_true_epoch, y_score_epoch) if len(set(y_true_epoch)) > 1 else 0.0
                    except Exception:
                        ap_now = 0.0
                    print(
                        f"Fold {fold+1} Epoch {epoch+1}/{args['max_epochs']} Step {step} "
                        f"loss={loss.item():.5f} train_ap={ap_now:.5f}",
                        flush=True,
                    )

            scheduler.step()

            # Validate
            model.eval()
            val_losses = []
            val_true, val_score = [], []
            with torch.no_grad():
                for step, (input_nodes, output_nodes, blocks) in enumerate(valid_loader):
                    blocks = [b.to(device) for b in blocks]
                    input_nodes = input_nodes.to(device)
                    seeds = output_nodes.to(device)

                    batch_num_feat, _, batch_labels, propagate_labels = load_lpa_subtensor(
                        num_feat, cat_feat, labels_t, seeds, input_nodes, device
                    )

                    logits = model(blocks, batch_num_feat, propagate_labels)
                    loss = loss_fn(logits, batch_labels)
                    val_losses.append(float(loss.item()))

                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    val_score.extend(probs.tolist())
                    val_true.extend(batch_labels.detach().cpu().numpy().tolist())

                    if step % 10 == 0:
                        try:
                            val_ap = average_precision_score(val_true, val_score) if len(set(val_true)) > 1 else 0.0
                            val_auc = roc_auc_score(val_true, val_score) if len(set(val_true)) > 1 else 0.0
                        except Exception:
                            val_ap, val_auc = 0.0, 0.0
                        print(
                            f"[VAL] Fold {fold+1} Epoch {epoch+1}/{args['max_epochs']} Step {step} "
                            f"val_loss={np.mean(val_losses):.5f} val_auc={val_auc:.5f} val_ap={val_ap:.5f}",
                            flush=True,
                        )

            # fold epoch metric
            try:
                epoch_val_ap = average_precision_score(val_true, val_score) if len(set(val_true)) > 1 else 0.0
            except Exception:
                epoch_val_ap = 0.0

            if experiment is not None:
                try:
                    experiment.log_metrics(
                        {
                            "fold": fold + 1,
                            "epoch": epoch + 1,
                            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
                            "val_loss": float(np.mean(val_losses)) if val_losses else 0.0,
                            "val_ap": float(epoch_val_ap),
                        },
                        step=(fold * int(args["max_epochs"]) + epoch),
                    )
                except Exception:
                    pass

            if epoch_val_ap > best_ap:
                best_ap = epoch_val_ap
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= early_stopping:
                    break

        if best_state is not None:
            model.load_state_dict(best_state, strict=False)

        # OOF predictions for val
        model.eval()
        val_loader = NodeDataLoader(
            graph, val_idx, MultiLayerFullNeighborSampler(args["n_layers"]),
            batch_size=args["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=0
        )
        with torch.no_grad():
            for input_nodes, output_nodes, blocks in val_loader:
                blocks = [b.to(device) for b in blocks]
                input_nodes = input_nodes.to(device)
                seeds = output_nodes.to(device)
                batch_num_feat, _, _, propagate_labels = load_lpa_subtensor(
                    num_feat, cat_feat, labels_t, seeds, input_nodes, device
                )
                logits = model(blocks, batch_num_feat, propagate_labels)
                oof_predictions[seeds] = logits

        # Fold test inference (held-out set). We average probabilities across folds.
        test_loader = NodeDataLoader(
            graph, test_idx, MultiLayerFullNeighborSampler(args["n_layers"]),
            batch_size=args["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=0
        )
        with torch.no_grad():
            for input_nodes, output_nodes, blocks in test_loader:
                blocks = [b.to(device) for b in blocks]
                input_nodes = input_nodes.to(device)
                seeds = output_nodes.to(device)
                batch_num_feat, _, _, propagate_labels = load_lpa_subtensor(
                    num_feat, cat_feat, labels_t, seeds, input_nodes, device
                )
                logits = model(blocks, batch_num_feat, propagate_labels)
                probs = torch.softmax(logits, dim=1)[:, 1]
                test_pred_sum[seeds] += probs

    # Metrics
    oof = torch.softmax(oof_predictions[train_idx], dim=1)[:, 1].detach().cpu().numpy()
    y_true = labels.iloc[train_idx].values

    try:
        oof_auc = roc_auc_score(y_true, oof) if len(set(y_true)) > 1 else 0.0
        oof_f1 = f1_score(y_true, (oof > 0.5).astype(int)) if len(set(y_true)) > 1 else 0.0
        oof_ap = average_precision_score(y_true, oof) if len(set(y_true)) > 1 else 0.0
    except Exception:
        oof_auc, oof_f1, oof_ap = 0.0, 0.0, 0.0

    print("NN out of fold AP is:", oof_ap)
    print("oof AUC:", oof_auc)
    print("oof f1:", oof_f1)

    test_probs = (test_pred_sum[test_idx] / float(args["n_fold"])).detach().cpu().numpy()
    test_true = labels.iloc[test_idx].values
    try:
        test_auc = roc_auc_score(test_true, test_probs) if len(set(test_true)) > 1 else 0.0
        test_f1 = f1_score(test_true, (test_probs > 0.5).astype(int)) if len(set(test_true)) > 1 else 0.0
        test_ap = average_precision_score(test_true, test_probs) if len(set(test_true)) > 1 else 0.0
    except Exception:
        test_auc, test_f1, test_ap = 0.0, 0.0, 0.0

    print("test AUC:", test_auc)
    print("test f1:", test_f1)
    print("test AP:", test_ap)


# -------------------------
# IEEE RAW/NORM/V2 graph helpers (unchanged in spirit)
# -------------------------

def _read_ieee_from_zip(zip_path: str):
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as z:
        # Try common Kaggle structure
        tx_name = None
        id_name = None
        for name in z.namelist():
            low = name.lower()
            if low.endswith("train_transaction.csv"):
                tx_name = name
            if low.endswith("train_identity.csv"):
                id_name = name

        if tx_name is None:
            raise FileNotFoundError("train_transaction.csv not found in zip")

        with z.open(tx_name) as f:
            tx = pd.read_csv(f)

        identity = None
        if id_name is not None:
            with z.open(id_name) as f:
                identity = pd.read_csv(f)

        return tx, identity


def _find_ieee_raw_files(prefix_dir: str):
    tx_candidates = [
        os.path.join(prefix_dir, "train_transaction.csv"),
        os.path.join(prefix_dir, "IEEE", "train_transaction.csv"),
        os.path.join(prefix_dir, "ieee", "train_transaction.csv"),
        os.path.join(prefix_dir, "ieee-fraud-detection", "train_transaction.csv"),
    ]
    tx_path = next((p for p in tx_candidates if os.path.exists(p)), None)

    id_candidates = [
        os.path.join(prefix_dir, "train_identity.csv"),
        os.path.join(prefix_dir, "IEEE", "train_identity.csv"),
        os.path.join(prefix_dir, "ieee", "train_identity.csv"),
        os.path.join(prefix_dir, "ieee-fraud-detection", "train_identity.csv"),
    ]
    id_path = next((p for p in id_candidates if os.path.exists(p)), None)

    return tx_path, id_path


def _build_ieee_raw_graph(df_feat: pd.DataFrame, time_col="TransactionDT", edge_per_trans=3, max_group_size=5000):
    n = len(df_feat)
    src, dst = [], []

    # Sort by time for edges
    if time_col in df_feat.columns:
        order = np.argsort(df_feat[time_col].values)
    else:
        order = np.arange(n)

    # Simple temporal linking: connect each transaction to the previous K
    for i in range(n):
        u = order[i]
        for k in range(1, edge_per_trans + 1):
            if i - k >= 0:
                v = order[i - k]
                src.append(u)
                dst.append(v)

    src = np.array(src, dtype=np.int64)
    dst = np.array(dst, dtype=np.int64)

    g = dgl.graph((src, dst), num_nodes=n)
    g = dgl.to_simple(g)
    g = dgl.add_self_loop(g)
    return g


def _encode_m_cols(df: pd.DataFrame, m_cols):
    for c in m_cols:
        if c not in df.columns:
            continue
        df[c] = df[c].astype(str)
        df[c] = df[c].replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
        df[c] = df[c].map({"T": 1, "F": 0}).fillna(2).astype(int)
    return df


def _ieee_mode_v2_features(tx: pd.DataFrame, identity: pd.DataFrame = None):
    """
    IEEE v2: strict numeric-only feature set that GTAN can ingest (no object dtype).
    This is correctness-first; performance improvements require adding categorical encoding later.
    """
    df = tx.copy()
    if identity is not None:
        df = df.merge(identity, on="TransactionID", how="left")

    # Target
    labels = df["isFraud"].fillna(0).astype(int)
    df = df.drop(columns=["isFraud"])

    # Handle M columns as 0/1/2
    m_cols = [c for c in df.columns if c.startswith("M")]
    df = _encode_m_cols(df, m_cols)

    # Drop obvious high-cardinality categoricals + raw IDs if present
    drop_like = [
        "TransactionID",
        "P_emaildomain", "R_emaildomain",
        "card4", "card6",
        "DeviceType", "DeviceInfo",
    ]
    for c in drop_like:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Keep only numeric columns
    df_num = df.select_dtypes(include=[np.number]).copy()

    # Fill NAs
    df_num = df_num.fillna(0)

    return df_num, labels


def load_gtan_data(prefix: str, dataset: str, test_size: float = 0.4, seed: int = 2023,
                  ieee_mode: str = None):
    """
    Load data for GTAN model.
    """
    if dataset.upper() == "IEEE":
        # load from zip if present
        zip_candidates = [
            os.path.join(prefix, "ieee-fraud-detection.zip"),
            os.path.join(prefix, "IEEE-CIS-Fraud-Detection.zip"),
            os.path.join(prefix, "IEEE.zip"),
        ]
        zip_path = next((p for p in zip_candidates if os.path.exists(p)), None)

        if zip_path is not None:
            tx, identity = _read_ieee_from_zip(zip_path)
        else:
            tx_path, id_path = _find_ieee_raw_files(prefix)
            if tx_path is None:
                raise FileNotFoundError(
                    "IEEE train_transaction.csv not found. Put it in prefix dir or provide ieee-fraud-detection.zip"
                )
            tx = pd.read_csv(tx_path)
            identity = pd.read_csv(id_path) if id_path is not None and os.path.exists(id_path) else None

        if (ieee_mode or "").lower() == "v2":
            feat_df, labels = _ieee_mode_v2_features(tx, identity)
            g = _build_ieee_raw_graph(tx, time_col="TransactionDT")
        else:
            # Fallback: basic numeric-only with minimal handling
            df = tx.copy()
            labels = df["isFraud"].fillna(0).astype(int)
            df = df.drop(columns=["isFraud"])
            df = df.select_dtypes(include=[np.number]).fillna(0)
            feat_df = df
            g = _build_ieee_raw_graph(tx, time_col="TransactionDT")

        # Split indices
        idx_all = np.arange(len(feat_df))
        train_idx, test_idx = train_test_split(
            idx_all, test_size=test_size, random_state=seed, stratify=labels.values
        )

        return feat_df, labels, train_idx, test_idx, g, []

    raise NotImplementedError(f"dataset={dataset} is not supported in this loader")
