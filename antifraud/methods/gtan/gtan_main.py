import numpy as np
import dgl
import torch
import os
import time
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import torch.optim as optim
from scipy.io import loadmat
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from dgl.dataloading import MultiLayerFullNeighborSampler

try:
    from dgl.dataloading import NodeDataLoader
except ImportError:
    from dgl.dataloading import DataLoader as NodeDataLoader

from torch.optim.lr_scheduler import MultiStepLR

from .gtan_model import GraphAttnModel
from .gtan_lpa import load_lpa_subtensor
from . import *

try:
    from comet_ml import Experiment
except Exception:
    Experiment = None


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


def gtan_main(feat_df, graph, train_idx, test_idx, labels, args, cat_features, experiment: Experiment = None):
    device = args['device']
    graph = graph.to(device)

    oof_predictions = torch.from_numpy(np.zeros([len(feat_df), 2])).float().to(device)
    test_predictions = torch.from_numpy(np.zeros([len(feat_df), 2])).float().to(device)

    kfold = StratifiedKFold(n_splits=args['n_fold'], shuffle=True, random_state=args['seed'])

    y_target = labels.iloc[train_idx].values
    num_feat = torch.from_numpy(feat_df.values).float().to(device)
    cat_feat = {col: torch.from_numpy(feat_df[col].values).long().to(device) for col in cat_features}

    y = labels
    labels = torch.from_numpy(y.values).long().to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)

    if experiment is not None:
        try:
            experiment.log_parameters({
                "method": "gtan",
                "dataset": args.get("dataset"),
                "ieee_mode": args.get("ieee_mode"),
                "seed": args.get("seed"),
                "n_fold": args.get("n_fold"),
                "lr": args.get("lr"),
                "wd": args.get("wd"),
                "hid_dim": args.get("hid_dim"),
                "n_layers": args.get("n_layers"),
                "dropout": args.get("dropout"),
                "batch_size": args.get("batch_size"),
            })
        except Exception:
            pass

    for fold, (trn_idx, val_idx) in enumerate(kfold.split(train_idx, y_target)):
        print(f"Training fold {fold + 1}", flush=True)

        trn_idx = np.array(train_idx)[trn_idx]
        val_idx = np.array(train_idx)[val_idx]

        model = GraphAttnModel(
            in_dim=num_feat.shape[1],
            hidden_dim=args['hid_dim'],
            out_dim=2,
            n_layers=args['n_layers'],
            n_heads=1,
            dropout=args['dropout'],
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
        scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.5)

        sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        train_loader = NodeDataLoader(
            graph, trn_idx, sampler,
            batch_size=args['batch_size'],
            shuffle=True,
            drop_last=False,
            num_workers=0
        )
        valid_loader = NodeDataLoader(
            graph, val_idx, sampler,
            batch_size=args['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=0
        )

        best_ap = -1
        best_state = None
        patience = 0

        for epoch in range(args['max_epochs']):
            model.train()
            train_losses = []

            y_true_epoch = []
            y_score_epoch = []

            for step, (input_nodes, output_nodes, blocks) in enumerate(train_loader):
                blocks = [b.to(device) for b in blocks]
                batch_nid = output_nodes.to(device)

                batch_labels = labels[batch_nid]

                batch_num_feat = num_feat[batch_nid]
                batch_cat_feat = {k: v[batch_nid] for k, v in cat_feat.items()}

                logits = model(blocks, batch_num_feat, batch_cat_feat)
                loss = loss_fn(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

                probs = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
                y_score_epoch.extend(probs.tolist())
                y_true_epoch.extend(batch_labels.detach().cpu().numpy().tolist())

                if step % 10 == 0:
                    try:
                        train_ap = average_precision_score(y_true_epoch, y_score_epoch) if len(set(y_true_epoch)) > 1 else 0.0
                        train_auc = roc_auc_score(y_true_epoch, y_score_epoch) if len(set(y_true_epoch)) > 1 else 0.0
                        train_acc = (np.array(y_score_epoch) > 0.5).mean() if len(y_score_epoch) else 0.0
                    except Exception:
                        train_ap, train_auc, train_acc = 0.0, 0.0, 0.0

                    print(
                        f"In epoch:{epoch:03d}|batch:{step:04d}, "
                        f"train_loss:{np.mean(train_losses):.6f}, "
                        f"train_ap:{train_ap:.4f}, train_acc:{train_acc:.4f}, train_auc:{train_auc:.4f}",
                        flush=True
                    )

            scheduler.step()

            # ---- validation ----
            model.eval()
            val_losses = []
            val_true = []
            val_score = []

            with torch.no_grad():
                for step, (input_nodes, output_nodes, blocks) in enumerate(valid_loader):
                    blocks = [b.to(device) for b in blocks]
                    batch_nid = output_nodes.to(device)

                    batch_labels = labels[batch_nid]
                    batch_num_feat = num_feat[batch_nid]
                    batch_cat_feat = {k: v[batch_nid] for k, v in cat_feat.items()}

                    logits = model(blocks, batch_num_feat, batch_cat_feat)
                    loss = loss_fn(logits, batch_labels)
                    val_losses.append(loss.item())

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

            if experiment is not None:
                try:
                    experiment.log_metric("train_loss", float(np.mean(train_losses)), step=fold * 1000 + epoch)
                    experiment.log_metric("val_loss", float(np.mean(val_losses)), step=fold * 1000 + epoch)
                    experiment.log_metric("val_ap", float(epoch_val_ap), step=fold * 1000 + epoch)
                except Exception:
                    pass

            # early stopping on AP
            if epoch_val_ap > best_ap:
                best_ap = epoch_val_ap
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= args.get("early_stopping", 10):
                    break

        # restore best state
        if best_state is not None:
            model.load_state_dict(best_state, strict=False)

        # ---- OOF predictions ----
        model.eval()
        all_loader = NodeDataLoader(
            graph, val_idx, MultiLayerFullNeighborSampler(args['n_layers']),
            batch_size=args['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=0
        )
        with torch.no_grad():
            for input_nodes, output_nodes, blocks in all_loader:
                blocks = [b.to(device) for b in blocks]
                batch_nid = output_nodes.to(device)
                batch_num_feat = num_feat[batch_nid]
                batch_cat_feat = {k: v[batch_nid] for k, v in cat_feat.items()}
                logits = model(blocks, batch_num_feat, batch_cat_feat)
                oof_predictions[batch_nid] = logits

    # ---- final eval on test split ----
    oof = torch.softmax(oof_predictions[train_idx], dim=1)[:, 1].detach().cpu().numpy()
    y_true = y.iloc[train_idx].values

    try:
        oof_auc = roc_auc_score(y_true, oof) if len(set(y_true)) > 1 else 0.0
        oof_f1 = f1_score(y_true, (oof > 0.5).astype(int)) if len(set(y_true)) > 1 else 0.0
        oof_ap = average_precision_score(y_true, oof) if len(set(y_true)) > 1 else 0.0
    except Exception:
        oof_auc, oof_f1, oof_ap = 0.0, 0.0, 0.0

    print("NN out of fold AP is:", oof_ap)
    print("oof AUC:", oof_auc)
    print("oof f1:", oof_f1)

    if experiment is not None:
        try:
            experiment.log_metric("oof_auc", float(oof_auc))
            experiment.log_metric("oof_f1", float(oof_f1))
            experiment.log_metric("oof_ap", float(oof_ap))
        except Exception:
            pass

    # test set metrics (if labels exist)
    test_probs = None
    if isinstance(test_idx, (list, np.ndarray)) and len(test_idx) > 0:
        test_logits = torch.softmax(oof_predictions[test_idx], dim=1)[:, 1].detach().cpu().numpy()
        test_probs = test_logits

    if test_probs is not None and len(test_probs) > 0:
        test_true = y.iloc[test_idx].values
        try:
            test_auc = roc_auc_score(test_true, test_probs) if len(set(test_true)) > 1 else 0.0
            test_f1 = f1_score(test_true, (test_probs > 0.5).astype(int)) if len(set(test_true)) > 1 else 0.0
            test_ap = average_precision_score(test_true, test_probs) if len(set(test_true)) > 1 else 0.0
        except Exception:
            test_auc, test_f1, test_ap = 0.0, 0.0, 0.0

        print("test AUC:", test_auc)
        print("test f1:", test_f1)
        print("test AP:", test_ap)

        if experiment is not None:
            try:
                experiment.log_metric("test_auc", float(test_auc))
                experiment.log_metric("test_f1", float(test_f1))
                experiment.log_metric("test_ap", float(test_ap))
            except Exception:
                pass


# -------------------------
# IEEE RAW helpers
# -------------------------

def _find_ieee_files(prefix_dir: str):
    """
    Find IEEE train_transaction.csv and train_identity.csv in a few common locations.
    Returns (tx_path, id_path) where either can be None.
    """
    candidates = [
        os.path.join(prefix_dir, "train_transaction.csv"),
        os.path.join(prefix_dir, "IEEE", "train_transaction.csv"),
        os.path.join(prefix_dir, "ieee", "train_transaction.csv"),
        os.path.join(prefix_dir, "ieee-fraud-detection", "train_transaction.csv"),
    ]
    tx_path = next((p for p in candidates if os.path.exists(p)), None)

    id_candidates = [
        os.path.join(prefix_dir, "train_identity.csv"),
        os.path.join(prefix_dir, "IEEE", "train_identity.csv"),
        os.path.join(prefix_dir, "ieee", "train_identity.csv"),
        os.path.join(prefix_dir, "ieee-fraud-detection", "train_identity.csv"),
    ]
    id_path = next((p for p in id_candidates if os.path.exists(p)), None)

    return tx_path, id_path


def _read_ieee_from_zip(zip_path: str):
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as z:
        def _read(name):
            with z.open(name) as f:
                return pd.read_csv(f)

        # common kaggle path
        names = z.namelist()
        tx_name = next((n for n in names if n.endswith("train_transaction.csv")), None)
        id_name = next((n for n in names if n.endswith("train_identity.csv")), None)
        if tx_name is None:
            raise FileNotFoundError("train_transaction.csv not found inside IEEE zip")

        tx = _read(tx_name)
        identity = _read(id_name) if id_name is not None else None
        return tx, identity


def _build_ieee_raw_graph(df_feat: pd.DataFrame, time_col="TransactionDT", edge_per_trans=3, max_group_size=5000):
    """
    Build a "transaction graph" from IEEE raw transaction-like features.

    Strategy (proxy entity groups):
      - Group by a handful of high-signal entity-ish columns if present
      - Within each group, connect transactions chronologically (next-k edges)
      - Limit huge groups to keep graph size manageable
    """
    n = len(df_feat)
    src, dst = [], []

    # choose candidate entity columns if present
    entity_cols = [c for c in ["card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2",
                              "P_emaildomain", "R_emaildomain", "DeviceType"] if c in df_feat.columns]

    # fallback: if none exist, just chain by time
    if len(entity_cols) == 0:
        if time_col in df_feat.columns:
            order = np.argsort(df_feat[time_col].fillna(0).values)
        else:
            order = np.arange(n)
        for i in range(len(order) - edge_per_trans):
            for k in range(1, edge_per_trans + 1):
                src.append(int(order[i]))
                dst.append(int(order[i + k]))
        g = dgl.graph((src, dst), num_nodes=n)
        return g

    # otherwise build per-group chaining
    tmp = df_feat.copy()
    if time_col in tmp.columns:
        tmp[time_col] = pd.to_numeric(tmp[time_col], errors="coerce").fillna(0)

    group_keys = tmp[entity_cols].astype(str).agg("|".join, axis=1)
    tmp["_grpkey"] = group_keys

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

    g = dgl.graph((src, dst), num_nodes=n)
    return g


# -------------------------
# IEEE NORM helpers
# -------------------------

def _find_ieee_norm_train(prefix_dir: str):
    """
    Find normalized IEEE "S-FFSD-like" file produced by normalize-ieee-dataset-to-sfssd.py
    Expected: data/ieee_sffsd_like/ieee_sffsd_like_train.csv
    """
    cand = os.path.join(prefix_dir, "ieee_sffsd_like", "ieee_sffsd_like_train.csv")
    if os.path.exists(cand):
        return cand
    return None


def _looks_like_ieee_norm(df_head: pd.DataFrame) -> bool:
    required = {"Time", "Source", "Target", "Amount", "Location", "Type", "Labels"}
    return required.issubset(set(df_head.columns))


def _load_ieee_norm_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def _build_ieee_norm_graph(df: pd.DataFrame, edge_per_trans=3):
    """
    Build a transaction graph similar to S-FFSD:
      - nodes = transactions
      - edges = next-k within each proxy entity group (Target + Location + Type)
    """
    n = len(df)
    src, dst = [], []
    # group key
    key = df[["Target", "Location", "Type"]].astype(str).agg("|".join, axis=1)
    tmp = df.copy()
    tmp["_grpkey"] = key

    # ensure time numeric
    tmp["Time"] = pd.to_numeric(tmp["Time"], errors="coerce").fillna(0)

    for _, idxs in tmp.groupby("_grpkey").groups.items():
        idxs = list(idxs)
        if len(idxs) <= 1:
            continue

        idxs = sorted(idxs, key=lambda i: tmp.at[i, "Time"])
        for i in range(len(idxs) - edge_per_trans):
            for k in range(1, edge_per_trans + 1):
                src.append(int(idxs[i]))
                dst.append(int(idxs[i + k]))

    g = dgl.graph((src, dst), num_nodes=n)
    return g


def load_gtan_data(dataset: str, test_size: float, ieee_mode: str = "auto"):
    """
    Load graph, features, labels for dataset.
    Returns: feat_data, labels, train_idx, test_idx, g, cat_features
    """
    prefix = os.path.join(os.path.dirname(__file__), "..", "..", "data/")

    if dataset == "S-FFSD":
        cat_features = ["Target", "Location", "Type"]

        df = pd.read_csv(prefix + "S-FFSDneofull.csv")
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        data = df[df["Labels"] <= 2].reset_index(drop=True)

        alls, allt = [], []
        pair = ["Source", "Target", "Location", "Type"]

        for p in pair:
            val = list(data[p].unique())
            le = LabelEncoder()
            data[p] = le.fit_transform(data[p].apply(str).values)

        for i in pair:
            if i == "Source":
                alls = data[i].values
            else:
                allt = data[i].values

        src = alls
        dst = allt

        g = dgl.graph((src, dst))
        g = _ensure_self_loops(g)

        labels = data["Labels"]
        feat_data = data.drop("Labels", axis=1)

        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

        try:
            dgl.data.utils.save_graphs(prefix + f"graph-{dataset}.bin", [g])
        except Exception:
            pass

        index = list(range(len(labels)))
        train_idx, test_idx, _, _ = train_test_split(
            index, labels, stratify=labels, test_size=test_size / 2,
            random_state=2, shuffle=True
        )

    elif str(dataset).lower() in ("ieee", "ieee-cis", "ieeecis"):
        # -------------------------
        # IEEE V2 (entity-gated builder)
        # -------------------------
        if (ieee_mode or "auto").lower() == "v2":
            v2_dir = os.path.join(prefix, "ieee_v2")
            feat_path = os.path.join(v2_dir, "ieee_v2_train.csv")
            graph_path = os.path.join(v2_dir, "graph-IEEE-v2.bin")

            print(f"[IEEE V2] feature_csv={feat_path} exists={os.path.exists(feat_path)}", flush=True)
            print(f"[IEEE V2] graph_bin={graph_path} exists={os.path.exists(graph_path)}", flush=True)

            if not os.path.exists(feat_path):
                raise FileNotFoundError(
                    f"IEEE v2 features not found: {feat_path}. "
                    f"Run prep_ieee_v2.py to generate ieee_v2_train.csv into {v2_dir}"
                )
            if not os.path.exists(graph_path):
                raise FileNotFoundError(
                    f"IEEE v2 graph bin not found: {graph_path}. "
                    f"Run prep_ieee_v2.py with --save-dgl-bin to generate graph-IEEE-v2.bin into {v2_dir}"
                )

            df_v2 = pd.read_csv(feat_path, low_memory=False)
            if "Labels" not in df_v2.columns:
                raise ValueError(f"IEEE v2 CSV must contain a 'Labels' column: {feat_path}")

            df_v2 = df_v2[df_v2["Labels"] <= 2].reset_index(drop=True)
            labels = df_v2["Labels"].astype(int)

            drop_cols = {"Labels", "TransactionID", "node_id"}
            feat_cols = [c for c in df_v2.columns if c not in drop_cols]
            feat_data = df_v2[feat_cols].copy()

            # Defensive: encode any object columns
            obj_cols = [c for c in feat_data.columns if feat_data[c].dtype == "object"]
            for col in obj_cols:
                feat_data[col] = feat_data[col].astype(str).fillna("NA")
                feat_data[col] = feat_data[col].astype("category").cat.codes

            feat_data = feat_data.apply(pd.to_numeric, errors="coerce").fillna(0)

            # Load prebuilt graph
            t0 = time.time()
            graphs, _ = dgl.data.utils.load_graphs(graph_path)
            g = graphs[0]
            g = _ensure_self_loops(g)
            dt = time.time() - t0

            print(f"[IEEE V2] Loaded cached graph in {dt:.2f}s", flush=True)
            print(f"[IEEE V2] graph stats: nodes={g.number_of_nodes():,} edges={g.number_of_edges():,}", flush=True)

            if g.number_of_nodes() != len(df_v2):
                raise ValueError(
                    f"IEEE v2 graph/node mismatch: graph_nodes={g.number_of_nodes():,} df_rows={len(df_v2):,}. "
                    f"Re-run prep_ieee_v2.py to regenerate BOTH ieee_v2_train.csv and graph-IEEE-v2.bin together."
                )

            g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
            g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

            index = list(range(len(labels)))
            train_idx, test_idx, _, _ = train_test_split(
                index, labels, stratify=labels, test_size=test_size / 2,
                random_state=2, shuffle=True
            )
            cat_features = []
            return feat_data, labels, train_idx, test_idx, g, cat_features

        # Decide raw vs norm
        ieee_mode = (ieee_mode or "auto").lower()
        norm_path = _find_ieee_norm_train(prefix)
        can_use_norm = False
        if norm_path and os.path.exists(norm_path):
            try:
                tmp = pd.read_csv(norm_path, nrows=5)
                can_use_norm = _looks_like_ieee_norm(tmp)
            except Exception:
                can_use_norm = False

        use_norm = (ieee_mode == "norm") or (ieee_mode == "auto" and can_use_norm)
        use_raw = (ieee_mode == "raw") or (not use_norm)

        if use_norm:
            # --- Normalized IEEE (S-FFSD-like) ---
            cat_features = ["Target", "Location", "Type"]

            df = _load_ieee_norm_df(norm_path)
            data = df[df["Labels"] <= 2].reset_index(drop=True)

            # Graph cache handling (norm)
            cache_path = os.path.join(prefix, "graph-IEEE-norm.bin")
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
                    # Validate node alignment
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

            for col in ["Source", "Target", "Location", "Type"]:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].apply(str).values)

            feat_data = data.drop("Labels", axis=1)
            labels = data["Labels"]

            index = list(range(len(labels)))
            train_idx, test_idx, _, _ = train_test_split(
                index, labels, stratify=labels, test_size=test_size,
                random_state=2, shuffle=True
            )

        else:
            # --- Raw IEEE (train_transaction.csv) ---
            # Transaction-only by default (identity ignored)
            use_identity = False
            cat_features = []

            zip_path = os.path.join(prefix, "ieee-fraud-detection.zip")
            tx_path, id_path = _find_ieee_files(prefix)

            if os.path.exists(zip_path):
                tx, identity = _read_ieee_from_zip(zip_path)
            elif tx_path is not None:
                tx = pd.read_csv(tx_path)
                identity = pd.read_csv(id_path) if (use_identity and id_path is not None) else None
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

            # prevent label leakage
            drop_cols = [c for c in ["TransactionID", "isFraud"] if c in df.columns]
            feat_df = df.drop(columns=drop_cols)

            # Graph cache handling (raw)
            cache_path = os.path.join(prefix, "graph-IEEE-raw.bin")
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

            obj_cols = [c for c in feat_df.columns if feat_df[c].dtype == "object"]
            for col in obj_cols:
                le = LabelEncoder()
                feat_df[col] = le.fit_transform(feat_df[col].astype(str).fillna("NA").values)

            feat_df = feat_df.apply(pd.to_numeric, errors="coerce").fillna(0)
            feat_data = feat_df

            g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
            g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

            index = list(range(len(labels)))
            train_idx, test_idx, _, _ = train_test_split(
                index, labels, stratify=labels, test_size=test_size,
                random_state=2, shuffle=True
            )

    else:
        # Default/other datasets from original repo
        cat_features = []
        data_file = loadmat(prefix + 'YelpChi.mat')
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)

        with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
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

        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

    return feat_data, labels, train_idx, test_idx, g, cat_features
