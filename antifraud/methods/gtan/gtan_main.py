# ~/ece600-CCFraudDetection/antifraud/methods/gtan/gtan_main.py

import os
import time
import inspect
import pickle

import numpy as np
import pandas as pd
import dgl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from dgl.dataloading import MultiLayerFullNeighborSampler
try:
    from dgl.dataloading import NodeDataLoader
except ImportError:
    from dgl.dataloading import DataLoader as NodeDataLoader

from torch.optim.lr_scheduler import MultiStepLR

from .gtan_model import GraphAttnModel
from .gtan_lpa import load_lpa_subtensor  # keep for repo compatibility
from . import *  # keep for repo compatibility

try:
    from comet_ml import Experiment
except Exception:
    Experiment = None


# -------------------------
# Utilities
# -------------------------

def _ensure_self_loops(g: dgl.DGLGraph) -> dgl.DGLGraph:
    try:
        g = dgl.remove_self_loop(g)
    except Exception:
        pass
    try:
        g = dgl.add_self_loop(g)
    except Exception:
        pass
    return g


def _safe_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    for c in obj_cols:
        df[c] = df[c].astype(str).fillna("NA")
        df[c] = df[c].astype("category").cat.codes
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df


def _normalize_dropout(drop):
    """
    GraphAttnModel in your repo does: nn.Dropout(drop[0]) and likely drop[1] too.
    So we must ALWAYS provide a 2-item list.
    """
    if drop is None:
        return [0.2, 0.1]
    if isinstance(drop, (int, float)):
        x = float(drop)
        return [x, x]
    if isinstance(drop, (list, tuple)):
        if len(drop) == 0:
            return [0.2, 0.1]
        if len(drop) == 1:
            x = float(drop[0])
            return [x, x]
        return [float(drop[0]), float(drop[1])]
    return [0.2, 0.1]


def _force_dropout(args: dict) -> list:
    """
    HARD GUARANTEE: get a valid dropout list and write it back into args['dropout'].
    This prevents 'NoneType is not subscriptable' regardless of config parsing.
    """
    raw = None
    if isinstance(args, dict):
        raw = args.get("dropout", None)
        if raw is None:
            raw = args.get("drop", None)
        if raw is None:
            raw = args.get("drop_rate", None)

    drop = _normalize_dropout(raw)

    if isinstance(args, dict):
        args["dropout"] = drop

    return drop


def _force_activation(args: dict):
    """
    Your GraphAttnModel requires 'activation' positional/kw.
    Default to F.elu if not provided.
    """
    act = None
    if isinstance(args, dict):
        act = args.get("activation", None)
    return act if act is not None else F.elu


def _build_graphattn_model(input_dim: int, args: dict) -> nn.Module:
    """
    Robust constructor for GraphAttnModel that:
      - NEVER passes drop=None
      - ALWAYS passes activation
      - Tries kwargs first, then positional patterns that include activation
    """
    drop = _force_dropout(args)
    activation = _force_activation(args)

    # Debug: this should ALWAYS show a list for dropout
    print(
        f"[GTAN MODEL] build: input_dim={input_dim} hid_dim={args.get('hid_dim')} "
        f"n_layers={args.get('n_layers')} dropout={drop} activation={getattr(activation, '__name__', str(activation))}",
        flush=True
    )

    sig = inspect.signature(GraphAttnModel.__init__)
    supported = set(sig.parameters.keys())
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    def _first_supported(keys):
        for k in keys:
            if k in supported:
                return k
        return None

    # Prefer kwargs when possible
    if has_var_kw:
        canon = {}
        k_in = _first_supported(["in_dim", "in_feats", "input_dim", "nfeat", "in_features"])
        k_h  = _first_supported(["hidden_dim", "hid_dim", "nhid", "hidden"])
        k_o  = _first_supported(["out_dim", "nclass", "num_class", "num_classes"])
        k_l  = _first_supported(["n_layers", "num_layers", "layers"])
        k_hd = _first_supported(["n_heads", "num_heads", "heads"])
        k_dr = _first_supported(["dropout", "drop"])
        k_ac = _first_supported(["activation", "act", "nonlinearity"])

        if k_in: canon[k_in] = input_dim
        if k_h:  canon[k_h] = args.get("hid_dim")
        if k_o:  canon[k_o] = 2
        if k_l:  canon[k_l] = args.get("n_layers")
        if k_hd: canon[k_hd] = 1
        if k_dr: canon[k_dr] = drop
        if k_ac: canon[k_ac] = activation

        try:
            return GraphAttnModel(**canon)
        except TypeError:
            pass

    # Try supported kwargs subset
    desired = {
        "in_dim": input_dim,
        "in_feats": input_dim,
        "input_dim": input_dim,
        "hidden_dim": args.get("hid_dim"),
        "hid_dim": args.get("hid_dim"),
        "out_dim": 2,
        "nclass": 2,
        "n_layers": args.get("n_layers"),
        "num_layers": args.get("n_layers"),
        "n_heads": 1,
        "dropout": drop,
        "drop": drop,
        "activation": activation,
        "act": activation,
        "nonlinearity": activation,
    }
    kwargs = {k: v for k, v in desired.items() if k in supported and v is not None}
    if kwargs:
        try:
            return GraphAttnModel(**kwargs)
        except TypeError:
            pass

    # Positional patterns (ALWAYS include activation)
    # Common forks:
    #   (in_dim, hid_dim, out_dim, n_layers, n_heads, drop, activation)
    #   (in_dim, hid_dim, out_dim, n_layers, drop, activation)
    try:
        return GraphAttnModel(input_dim, args.get("hid_dim"), 2, args.get("n_layers"), 1, drop, activation)
    except TypeError:
        return GraphAttnModel(input_dim, args.get("hid_dim"), 2, args.get("n_layers"), drop, activation)


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

            for step, (_, output_nodes, blocks) in enumerate(train_loader):
                blocks = [b.to(device) for b in blocks]
                batch_nid = output_nodes.to(device)

                batch_labels = labels_t[batch_nid]
                batch_num_feat = num_feat[batch_nid]
                batch_cat_feat = {k: v[batch_nid] for k, v in cat_feat.items()}

                logits = model(blocks, batch_num_feat, batch_cat_feat)
                loss = loss_fn(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(float(loss.item()))
                probs = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
                y_score_epoch.extend(probs.tolist())
                y_true_epoch.extend(batch_labels.detach().cpu().numpy().tolist())

                if step % 10 == 0:
                    try:
                        train_ap = average_precision_score(y_true_epoch, y_score_epoch) if len(set(y_true_epoch)) > 1 else 0.0
                        train_auc = roc_auc_score(y_true_epoch, y_score_epoch) if len(set(y_true_epoch)) > 1 else 0.0
                        train_acc = float((np.array(y_score_epoch) > 0.5).mean()) if len(y_score_epoch) else 0.0
                    except Exception:
                        train_ap, train_auc, train_acc = 0.0, 0.0, 0.0

                    print(
                        f"In epoch:{epoch:03d}|batch:{step:04d}, "
                        f"train_loss:{np.mean(train_losses):.6f}, "
                        f"train_ap:{train_ap:.4f}, train_acc:{train_acc:.4f}, train_auc:{train_auc:.4f}",
                        flush=True
                    )

            scheduler.step()

            model.eval()
            val_losses = []
            val_true, val_score = [], []

            with torch.no_grad():
                for step, (_, output_nodes, blocks) in enumerate(valid_loader):
                    blocks = [b.to(device) for b in blocks]
                    batch_nid = output_nodes.to(device)

                    batch_labels = labels_t[batch_nid]
                    batch_num_feat = num_feat[batch_nid]
                    batch_cat_feat = {k: v[batch_nid] for k, v in cat_feat.items()}

                    logits = model(blocks, batch_num_feat, batch_cat_feat)
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
                            f"In epoch:{epoch:03d}|batch:{step:04d}, "
                            f"val_loss:{np.mean(val_losses):.6f}, val_ap:{val_ap:.4f}, val_auc:{val_auc:.4f}",
                            flush=True
                        )

            try:
                epoch_val_ap = average_precision_score(val_true, val_score) if len(set(val_true)) > 1 else 0.0
            except Exception:
                epoch_val_ap = 0.0

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
            for _, output_nodes, blocks in val_loader:
                blocks = [b.to(device) for b in blocks]
                batch_nid = output_nodes.to(device)
                batch_num_feat = num_feat[batch_nid]
                batch_cat_feat = {k: v[batch_nid] for k, v in cat_feat.items()}
                logits = model(blocks, batch_num_feat, batch_cat_feat)
                oof_predictions[batch_nid] = logits

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

    test_probs = torch.softmax(oof_predictions[test_idx], dim=1)[:, 1].detach().cpu().numpy()
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
        names = z.namelist()
        tx_name = next((n for n in names if n.endswith("train_transaction.csv")), None)
        id_name = next((n for n in names if n.endswith("train_identity.csv")), None)
        if tx_name is None:
            raise FileNotFoundError("train_transaction.csv not found inside IEEE zip")

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

    entity_cols = [c for c in [
        "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2", "P_emaildomain", "R_emaildomain", "DeviceType"
    ] if c in df_feat.columns]

    if len(entity_cols) == 0:
        if time_col in df_feat.columns:
            order = np.argsort(pd.to_numeric(df_feat[time_col], errors="coerce").fillna(0).values)
        else:
            order = np.arange(n)
        for i in range(len(order) - edge_per_trans):
            for k in range(1, edge_per_trans + 1):
                src.append(int(order[i]))
                dst.append(int(order[i + k]))
        return dgl.graph((src, dst), num_nodes=n)

    tmp = df_feat.copy()
    if time_col in tmp.columns:
        tmp[time_col] = pd.to_numeric(tmp[time_col], errors="coerce").fillna(0)

    tmp["_grpkey"] = tmp[entity_cols].astype(str).agg("|".join, axis=1)

    for _, idxs in tmp.groupby("_grpkey").groups.items():
        idxs = list(idxs)
        if len(idxs) <= 1:
            continue
        if len(idxs) > max_group_size:
            idxs = idxs[:max_group_size]

        if time_col in tmp.columns:
            idxs = sorted(idxs, key=lambda i: tmp.at[i, time_col])
        else:
            idxs = sorted(idxs)

        for i in range(len(idxs) - edge_per_trans):
            for k in range(1, edge_per_trans + 1):
                src.append(int(idxs[i]))
                dst.append(int(idxs[i + k]))

    return dgl.graph((src, dst), num_nodes=n)


def _find_ieee_norm_train(prefix_dir: str):
    cand = os.path.join(prefix_dir, "ieee_sffsd_like", "ieee_sffsd_like_train.csv")
    return cand if os.path.exists(cand) else None


def _looks_like_ieee_norm(df_head: pd.DataFrame) -> bool:
    required = {"Time", "Source", "Target", "Amount", "Location", "Type", "Labels"}
    return required.issubset(set(df_head.columns))


def _build_ieee_norm_graph(df: pd.DataFrame, edge_per_trans=3):
    n = len(df)
    src, dst = [], []

    tmp = df.copy()
    tmp["Time"] = pd.to_numeric(tmp["Time"], errors="coerce").fillna(0)
    tmp["_grpkey"] = tmp[["Target", "Location", "Type"]].astype(str).agg("|".join, axis=1)

    for _, idxs in tmp.groupby("_grpkey").groups.items():
        idxs = list(idxs)
        if len(idxs) <= 1:
            continue
        idxs = sorted(idxs, key=lambda i: tmp.at[i, "Time"])
        for i in range(len(idxs) - edge_per_trans):
            for k in range(1, edge_per_trans + 1):
                src.append(int(idxs[i]))
                dst.append(int(idxs[i + k]))

    return dgl.graph((src, dst), num_nodes=n)


# -------------------------
# Dataset loader used by main.py
# -------------------------

def load_gtan_data(dataset: str, test_size: float, ieee_mode: str = "auto"):
    """
    Returns: feat_data, labels, train_idx, test_idx, g, cat_features
    """
    prefix_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    prefix_dir = os.path.abspath(prefix_dir)

    # ---- S-FFSD (repo default)
    if dataset == "S-FFSD":
        cat_features = ["Target", "Location", "Type"]

        df = pd.read_csv(os.path.join(prefix_dir, "S-FFSDneofull.csv"))
        df = df.loc[:, ~df.columns.str.contains("Unnamed")]
        data = df[df["Labels"] <= 2].reset_index(drop=True)

        for p in ["Source", "Target", "Location", "Type"]:
            le = LabelEncoder()
            data[p] = le.fit_transform(data[p].apply(str).values)

        src = data["Source"].values
        dst = data["Target"].values
        g = dgl.graph((src, dst))
        g = _ensure_self_loops(g)

        labels = data["Labels"].astype(int)
        feat_data = data.drop("Labels", axis=1)
        feat_data = _safe_numeric_frame(feat_data)

        g.ndata["label"] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata["feat"] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

        index = list(range(len(labels)))
        train_idx, test_idx, _, _ = train_test_split(
            index, labels, stratify=labels, test_size=test_size / 2,
            random_state=2, shuffle=True
        )
        return feat_data, labels, train_idx, test_idx, g, cat_features

    # ---- IEEE
    if str(dataset).lower() in ("ieee", "ieee-cis", "ieeecis"):
        ieee_mode = (ieee_mode or "auto").lower()

        # V2 mode: expects prebuilt CSV + graph
        if ieee_mode == "v2":
            v2_dir = os.path.join(prefix_dir, "ieee_v2")
            feat_path = os.path.join(v2_dir, "ieee_v2_train.csv")
            graph_path = os.path.join(v2_dir, "graph-IEEE-v2.bin")

            print(f"[IEEE V2] feature_csv={feat_path} exists={os.path.exists(feat_path)}", flush=True)
            print(f"[IEEE V2] graph_bin={graph_path} exists={os.path.exists(graph_path)}", flush=True)

            if not os.path.exists(feat_path):
                raise FileNotFoundError(f"Missing IEEE v2 features: {feat_path}")
            if not os.path.exists(graph_path):
                raise FileNotFoundError(f"Missing IEEE v2 graph: {graph_path}")

            df_v2 = pd.read_csv(feat_path, low_memory=False)
            if "Labels" not in df_v2.columns:
                raise ValueError("IEEE v2 CSV must include 'Labels' column")

            df_v2 = df_v2[df_v2["Labels"] <= 2].reset_index(drop=True)
            labels = df_v2["Labels"].astype(int)

            drop_cols = {"Labels", "TransactionID", "node_id"}
            feat_cols = [c for c in df_v2.columns if c not in drop_cols]
            feat_data = _safe_numeric_frame(df_v2[feat_cols].copy())

            t0 = time.time()
            graphs, _ = dgl.data.utils.load_graphs(graph_path)
            g = graphs[0]
            g = _ensure_self_loops(g)
            print(f"[IEEE V2] Loaded graph in {time.time()-t0:.2f}s", flush=True)
            print(f"[IEEE V2] graph stats: nodes={g.number_of_nodes():,} edges={g.number_of_edges():,}", flush=True)

            if g.number_of_nodes() != len(df_v2):
                raise ValueError(
                    f"IEEE v2 mismatch: graph_nodes={g.number_of_nodes():,} df_rows={len(df_v2):,}"
                )

            g.ndata["label"] = torch.from_numpy(labels.to_numpy()).to(torch.long)
            g.ndata["feat"] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

            index = list(range(len(labels)))
            train_idx, test_idx, _, _ = train_test_split(
                index, labels, stratify=labels, test_size=test_size / 2,
                random_state=2, shuffle=True
            )
            return feat_data, labels, train_idx, test_idx, g, []

        # Decide norm vs raw
        norm_path = _find_ieee_norm_train(prefix_dir)
        can_use_norm = False
        if norm_path and os.path.exists(norm_path):
            try:
                head = pd.read_csv(norm_path, nrows=5)
                can_use_norm = _looks_like_ieee_norm(head)
            except Exception:
                can_use_norm = False

        use_norm = (ieee_mode == "norm") or (ieee_mode == "auto" and can_use_norm)
        use_raw = (ieee_mode == "raw") or (not use_norm)

        # ---- IEEE NORM
        if use_norm:
            cat_features = ["Target", "Location", "Type"]

            df = pd.read_csv(norm_path)
            data = df[df["Labels"] <= 2].reset_index(drop=True)

            cache_path = os.path.join(prefix_dir, "graph-IEEE-norm.bin")
            try:
                exists = os.path.exists(cache_path)
                size = os.path.getsize(cache_path) if exists else 0
                mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(cache_path))) if exists else "NA"
                print(f"[IEEE GRAPH] norm-cache path={cache_path} exists={exists} size_bytes={size} mtime={mtime}", flush=True)
            except Exception:
                pass

            g = None
            if os.path.exists(cache_path):
                try:
                    print("[IEEE GRAPH] Loading cached graph-IEEE-norm.bin ...", flush=True)
                    t0 = time.time()
                    graphs, _ = dgl.data.utils.load_graphs(cache_path)
                    g = graphs[0]
                    print(f"[IEEE GRAPH] Loaded cached graph in {time.time()-t0:.2f}s", flush=True)
                    print(f"[IEEE GRAPH] loaded-norm stats: nodes={g.number_of_nodes():,} edges={g.number_of_edges():,}", flush=True)
                    if g.number_of_nodes() != len(data):
                        print(f"[IEEE GRAPH] WARNING node count mismatch: graph_nodes={g.number_of_nodes():,} df_rows={len(data):,} -> will rebuild", flush=True)
                        g = None
                except Exception as e:
                    print(f"[IEEE GRAPH] WARNING failed to load cached norm graph ({e}); will rebuild", flush=True)
                    g = None

            if g is None:
                print("[IEEE GRAPH] Building graph (norm) ...", flush=True)
                t0 = time.time()
                g = _build_ieee_norm_graph(data, edge_per_trans=3)
                g = _ensure_self_loops(g)
                print(f"[IEEE GRAPH] Built graph in {time.time()-t0:.2f}s", flush=True)
                print(f"[IEEE GRAPH] built-norm stats: nodes={g.number_of_nodes():,} edges={g.number_of_edges():,}", flush=True)
                try:
                    dgl.data.utils.save_graphs(cache_path, [g])
                    print(f"[IEEE GRAPH] Saved graph to {cache_path}", flush=True)
                except Exception as e:
                    print(f"[IEEE GRAPH] WARNING failed to save norm graph cache ({e})", flush=True)

            # Encode categorical columns to ints
            for col in ["Source", "Target", "Location", "Type"]:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].apply(str).values)

            labels = data["Labels"].astype(int)
            feat_data = data.drop("Labels", axis=1)
            feat_data = _safe_numeric_frame(feat_data)

            g.ndata["label"] = torch.from_numpy(labels.to_numpy()).to(torch.long)
            g.ndata["feat"] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

            index = list(range(len(labels)))
            train_idx, test_idx, _, _ = train_test_split(
                index, labels, stratify=labels, test_size=test_size,
                random_state=2, shuffle=True
            )
            return feat_data, labels, train_idx, test_idx, g, cat_features

        # ---- IEEE RAW
        if use_raw:
            # Transaction-only by default
            use_identity = False

            zip_path = os.path.join(prefix_dir, "ieee-fraud-detection.zip")
            tx_path, id_path = _find_ieee_raw_files(prefix_dir)

            if os.path.exists(zip_path):
                tx, identity = _read_ieee_from_zip(zip_path)
            elif tx_path is not None:
                tx = pd.read_csv(tx_path, low_memory=False)
                identity = pd.read_csv(id_path, low_memory=False) if (use_identity and id_path is not None) else None
            else:
                raise FileNotFoundError(
                    "IEEE dataset not found. Place either:\n"
                    f"  - {zip_path}\n"
                    "  - data/train_transaction.csv\n"
                    "  - data/IEEE/train_transaction.csv\n"
                    "  - data/ieee/train_transaction.csv\n"
                    "  - data/ieee-fraud-detection/train_transaction.csv\n"
                )

            if use_identity and identity is not None and "TransactionID" in tx.columns and "TransactionID" in identity.columns:
                df = tx.merge(identity, on="TransactionID", how="left")
            else:
                df = tx.copy()

            if "isFraud" not in df.columns:
                raise ValueError("IEEE train_transaction.csv must include 'isFraud' column")

            labels = df["isFraud"].astype(int)

            drop_cols = [c for c in ["TransactionID", "isFraud"] if c in df.columns]
            feat_df = df.drop(columns=drop_cols)

            cache_path = os.path.join(prefix_dir, "graph-IEEE-raw.bin")
            try:
                exists = os.path.exists(cache_path)
                size = os.path.getsize(cache_path) if exists else 0
                mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(cache_path))) if exists else "NA"
                print(f"[IEEE GRAPH] raw-cache path={cache_path} exists={exists} size_bytes={size} mtime={mtime}", flush=True)
            except Exception:
                pass

            g = None
            if os.path.exists(cache_path):
                try:
                    print("[IEEE GRAPH] Loading cached graph-IEEE-raw.bin ...", flush=True)
                    t0 = time.time()
                    graphs, _ = dgl.data.utils.load_graphs(cache_path)
                    g = graphs[0]
                    print(f"[IEEE GRAPH] Loaded cached graph in {time.time()-t0:.2f}s", flush=True)
                    print(f"[IEEE GRAPH] loaded-raw stats: nodes={g.number_of_nodes():,} edges={g.number_of_edges():,}", flush=True)
                    if g.number_of_nodes() != len(feat_df):
                        print(f"[IEEE GRAPH] WARNING node count mismatch: graph_nodes={g.number_of_nodes():,} df_rows={len(feat_df):,} -> will rebuild", flush=True)
                        g = None
                except Exception as e:
                    print(f"[IEEE GRAPH] WARNING failed to load cached raw graph ({e}); will rebuild", flush=True)
                    g = None

            if g is None:
                print("[IEEE GRAPH] Building graph (raw) ...", flush=True)
                t0 = time.time()
                g = _build_ieee_raw_graph(feat_df, time_col="TransactionDT", edge_per_trans=3, max_group_size=5000)
                g = _ensure_self_loops(g)
                print(f"[IEEE GRAPH] Built graph in {time.time()-t0:.2f}s", flush=True)
                print(f"[IEEE GRAPH] built-raw stats: nodes={g.number_of_nodes():,} edges={g.number_of_edges():,}", flush=True)
                try:
                    dgl.data.utils.save_graphs(cache_path, [g])
                    print(f"[IEEE GRAPH] Saved graph to {cache_path}", flush=True)
                except Exception as e:
                    print(f"[IEEE GRAPH] WARNING failed to save raw graph cache ({e})", flush=True)

            feat_data = _safe_numeric_frame(feat_df)

            g.ndata["label"] = torch.from_numpy(labels.to_numpy()).to(torch.long)
            g.ndata["feat"] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

            index = list(range(len(labels)))
            train_idx, test_idx, _, _ = train_test_split(
                index, labels, stratify=labels, test_size=test_size,
                random_state=2, shuffle=True
            )
            return feat_data, labels, train_idx, test_idx, g, []

    # ---- YelpChi fallback
    data_file = loadmat(os.path.join(prefix_dir, "YelpChi.mat"))
    labels = pd.DataFrame(data_file["label"].flatten())[0]
    feat_data = pd.DataFrame(data_file["features"].todense().A)

    with open(os.path.join(prefix_dir, "yelp_homo_adjlists.pickle"), "rb") as file:
        homo = pickle.load(file)

    index = list(range(len(labels)))
    train_idx, test_idx, _, _ = train_test_split(
        index, labels, stratify=labels, test_size=test_size,
        random_state=2, shuffle=True
    )

    src, tgt = [], []
    for i in range(len(homo)):
        for j in homo[i]:
            src.append(i)
            tgt.append(j)

    g = dgl.graph((src, tgt), num_nodes=len(labels))
    g = _ensure_self_loops(g)

    g.ndata["label"] = torch.from_numpy(labels.to_numpy()).to(torch.long)
    g.ndata["feat"] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

    return feat_data, labels, train_idx, test_idx, g, []
