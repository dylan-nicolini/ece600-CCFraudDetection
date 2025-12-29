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

from sklearn.metrics import confusion_matrix


def train(args, graph, feat_df, labels, train_idx, test_idx, cat_features=None, experiment=None):
    """Training process for GTAN."""
    if cat_features is None:
        cat_features = []

    in_size = feat_df.shape[1]
    num_classes = 2

    # device
    device = torch.device(args['device'])

    # graph
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

    # Training fold
    for fold, (trn_idx, val_idx) in enumerate(kfold.split(train_idx, y_target)):
        print(f"Training fold {fold + 1}", flush=True)

        # prepare dataloader
        sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        dataloader = NodeDataLoader(
            graph, train_idx[trn_idx], sampler,
            batch_size=args['batch_size'],
            shuffle=True,
            drop_last=False,
            num_workers=0
        )

        # model
        from .models import GTAN
        model = GTAN(
            in_size=in_size,
            hidden_size=args['hid_dim'],
            num_classes=num_classes,
            num_layers=args['n_layers'],
            dropout=args['dropout'],
            device=device,
            gated=args['gated']
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])

        best_val_auc = 0
        best_val_ap = 0
        best_val_f1 = 0
        early_stop_cnt = 0

        # training epochs
        for epoch in range(args['max_epochs']):
            model.train()
            train_loss_list = []
            train_auc_list, train_ap_list, train_acc_list = [], [], []

            for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                blocks = [b.to(device) for b in blocks]
                batch_feat = num_feat[input_nodes]
                batch_labels = labels[output_nodes]

                logits = model(blocks, batch_feat, {k: v[input_nodes] for k, v in cat_feat.items()})
                loss = loss_fn(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # stats
                prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                y_true = batch_labels.detach().cpu().numpy()

                train_auc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else 0.0
                train_ap = average_precision_score(y_true, prob) if len(np.unique(y_true)) > 1 else 0.0
                pred = (prob >= 0.5).astype(int)
                train_acc = (pred == y_true).mean()

                train_loss_list.append(loss.item())
                train_auc_list.append(train_auc)
                train_ap_list.append(train_ap)
                train_acc_list.append(train_acc)

                if batch_id % 10 == 0:
                    print(f"In epoch:{epoch:03d}|batch:{batch_id:04d}, "
                          f"train_loss:{np.mean(train_loss_list):.6f}, "
                          f"train_ap:{np.mean(train_ap_list):.4f}, "
                          f"train_acc:{np.mean(train_acc_list):.4f}, "
                          f"train_auc:{np.mean(train_auc_list):.4f}", flush=True)

            # validation
            model.eval()
            val_nodes = train_idx[val_idx]

            val_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
            val_loader = NodeDataLoader(
                graph, val_nodes, val_sampler,
                batch_size=args['batch_size'],
                shuffle=False,
                drop_last=False,
                num_workers=0
            )

            val_loss_list = []
            val_auc_list, val_ap_list = [], []

            with torch.no_grad():
                for batch_id, (in_nodes, out_nodes, blocks) in enumerate(val_loader):
                    blocks = [b.to(device) for b in blocks]
                    batch_feat = num_feat[in_nodes]
                    batch_labels = labels[out_nodes]

                    logits = model(blocks, batch_feat, {k: v[in_nodes] for k, v in cat_feat.items()})
                    loss = loss_fn(logits, batch_labels)
                    val_loss_list.append(loss.item())

                    prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                    y_true = batch_labels.detach().cpu().numpy()

                    val_auc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else 0.0
                    val_ap = average_precision_score(y_true, prob) if len(np.unique(y_true)) > 1 else 0.0

                    val_auc_list.append(val_auc)
                    val_ap_list.append(val_ap)

                    if batch_id % 10 == 0:
                        print(f"In epoch:{epoch:03d}|batch:{batch_id:04d}, "
                              f"val_loss:{np.mean(val_loss_list):.6f}, "
                              f"val_ap:{np.mean(val_ap_list):.4f}, "
                              f"val_auc:{np.mean(val_auc_list):.4f}", flush=True)

            val_loss = float(np.mean(val_loss_list)) if val_loss_list else 0.0
            val_auc = float(np.mean(val_auc_list)) if val_auc_list else 0.0
            val_ap = float(np.mean(val_ap_list)) if val_ap_list else 0.0

            # log to comet
            if experiment is not None:
                try:
                    experiment.log_metric("train_loss", float(np.mean(train_loss_list)), step=epoch)
                    experiment.log_metric("train_auc", float(np.mean(train_auc_list)), step=epoch)
                    experiment.log_metric("train_ap", float(np.mean(train_ap_list)), step=epoch)
                    experiment.log_metric("val_loss", val_loss, step=epoch)
                    experiment.log_metric("val_auc", val_auc, step=epoch)
                    experiment.log_metric("val_ap", val_ap, step=epoch)
                except Exception:
                    pass

            # early stopping based on AP (you can change to AUC if desired)
            improved = val_ap > best_val_ap
            if improved:
                best_val_ap = val_ap
                best_val_auc = val_auc
                early_stop_cnt = 0
                # Save best model for fold
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                early_stop_cnt += 1

            if early_stop_cnt >= args.get('early_stopping', 10):
                print(f"Early stopping at epoch={epoch} (best_val_ap={best_val_ap:.6f})", flush=True)
                break

        # restore best
        if 'best_state' in locals():
            model.load_state_dict(best_state)

        # predict OOF on full train_idx fold split
        model.eval()
        all_nodes = train_idx
        all_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        all_loader = NodeDataLoader(
            graph, all_nodes, all_sampler,
            batch_size=args['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=0
        )

        with torch.no_grad():
            for in_nodes, out_nodes, blocks in all_loader:
                blocks = [b.to(device) for b in blocks]
                batch_feat = num_feat[in_nodes]
                logits = model(blocks, batch_feat, {k: v[in_nodes] for k, v in cat_feat.items()})
                probs = torch.softmax(logits, dim=1)
                oof_predictions[out_nodes] = probs

    # final evaluation on test split
    test_nodes = test_idx
    test_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
    test_loader = NodeDataLoader(
        graph, test_nodes, test_sampler,
        batch_size=args['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    model.eval()
    test_probs = []
    test_true = []

    with torch.no_grad():
        for in_nodes, out_nodes, blocks in test_loader:
            blocks = [b.to(device) for b in blocks]
            batch_feat = num_feat[in_nodes]
            logits = model(blocks, batch_feat, {k: v[in_nodes] for k, v in cat_feat.items()})
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            test_probs.append(probs)
            test_true.append(labels[out_nodes].detach().cpu().numpy())

    test_probs = np.concatenate(test_probs) if test_probs else np.array([])
    test_true = np.concatenate(test_true) if test_true else np.array([])

    if len(test_probs) > 0 and len(np.unique(test_true)) > 1:
        test_auc = roc_auc_score(test_true, test_probs)
        test_ap = average_precision_score(test_true, test_probs)
    else:
        test_auc = 0.0
        test_ap = 0.0

    test_pred = (test_probs >= 0.5).astype(int) if len(test_probs) > 0 else np.array([])
    test_f1 = f1_score(test_true, test_pred) if len(test_pred) > 0 and len(np.unique(test_true)) > 1 else 0.0

    # log final metrics
    if experiment is not None:
        try:
            experiment.log_metric("test_auc", float(test_auc))
            experiment.log_metric("test_f1", float(test_f1))
            experiment.log_metric("test_ap", float(test_ap))
        except Exception:
            pass

    return float(test_auc), float(test_f1), float(test_ap)


# -------------------------
# IEEE RAW helpers
# -------------------------
def _find_ieee_files(prefix_dir: str):
    candidates = [
        (os.path.join(prefix_dir, "train_transaction.csv"), os.path.join(prefix_dir, "train_identity.csv")),
        (os.path.join(prefix_dir, "IEEE", "train_transaction.csv"), os.path.join(prefix_dir, "IEEE", "train_identity.csv")),
        (os.path.join(prefix_dir, "ieee", "train_transaction.csv"), os.path.join(prefix_dir, "ieee", "train_identity.csv")),
        (os.path.join(prefix_dir, "ieee-fraud-detection", "train_transaction.csv"),
         os.path.join(prefix_dir, "ieee-fraud-detection", "train_identity.csv")),
    ]
    for tx, ident in candidates:
        if os.path.exists(tx):
            return tx, ident if os.path.exists(ident) else None
    return None, None


def _load_ieee_raw_df(train_tx_path: str, train_id_path: str = None):
    df_tx = pd.read_csv(train_tx_path, low_memory=False)

    if train_id_path and os.path.exists(train_id_path):
        df_id = pd.read_csv(train_id_path, low_memory=False)
        df = df_tx.merge(df_id, on="TransactionID", how="left")
    else:
        df = df_tx

    return df


# -------------------------
# IEEE NORM helpers
# -------------------------
def _find_ieee_norm_train(prefix_dir: str):
    candidates = [
        os.path.join(prefix_dir, "ieee_sffsd_like", "ieee_sffsd_like_train.csv"),
        os.path.join(prefix_dir, "ieee_sffsd_like_train.csv"),
        os.path.join(prefix_dir, "IEEE_sffsd_like", "ieee_sffsd_like_train.csv"),
        os.path.join(prefix_dir, "ieee_sffsd_like", "train.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _looks_like_ieee_norm(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    required = {"Time", "Source", "Target", "Amount", "Location", "Type", "Labels"}
    return required.issubset(cols)


def _load_ieee_norm_df(norm_path: str):
    df = pd.read_csv(norm_path, low_memory=False)
    # enforce expected columns if present
    if "Amount" in df.columns and "TransactionAmt" in df.columns:
        df = df.rename(columns={"TransactionAmt": "Amount"})
    return df


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


def _build_ieee_norm_graph(data: pd.DataFrame, edge_per_trans: int = 3) -> dgl.DGLGraph:
    """
    Match S-FFSD behavior: connect by {Source, Target, Location, Type}, ordered by Time.
    """
    alls, allt = [], []
    pair = ["Source", "Target", "Location", "Type"]

    for column in pair:
        src, tgt = [], []
        for _, c_df in data.groupby(column):
            c_df = c_df.sort_values(by="Time")
            sorted_idxs = c_df.index.to_list()
            df_len = len(sorted_idxs)
            for i in range(df_len):
                for j in range(1, edge_per_trans + 1):
                    if i + j < df_len:
                        src.append(sorted_idxs[i])
                        tgt.append(sorted_idxs[i + j])
        if src:
            alls.extend(src)
            allt.extend(tgt)

    if len(alls) == 0:
        # fallback chain by time
        gdf = data.sort_values(by="Time")
        idxs = gdf.index.to_list()
        for i in range(len(idxs) - 1):
            alls.append(idxs[i])
            allt.append(idxs[i + 1])

    return dgl.graph((np.array(alls), np.array(allt)), num_nodes=len(data))


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

        for column in pair:
            src, tgt = [], []
            for _, c_df in data.groupby(column):
                c_df = c_df.sort_values(by="Time")
                sorted_idxs = c_df.index.to_list()
                df_len = len(sorted_idxs)
                for i in range(df_len):
                    for j in range(1, 4):
                        if i + j < df_len:
                            src.append(sorted_idxs[i])
                            tgt.append(sorted_idxs[i + j])
            if src:
                alls.extend(src)
                allt.extend(tgt)

        g = dgl.graph((np.array(alls), np.array(allt)), num_nodes=len(data))
        g = _ensure_self_loops(g)

        for col in ["Source", "Target", "Location", "Type"]:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].apply(str).values)

        feat_data = data.drop("Labels", axis=1)
        labels = data["Labels"]

        index = list(range(len(labels)))
        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

        try:
            dgl.data.utils.save_graphs(prefix + f"graph-{dataset}.bin", [g])
        except Exception:
            pass

        train_idx, test_idx, _, _ = train_test_split(
            index, labels, stratify=labels, test_size=test_size / 2,
            random_state=2, shuffle=True
        )

    elif str(dataset).lower() in ("ieee", "ieee-cis", "ieeecis"):
        # IEEE supports multiple modes:
        #   - raw  : load original train_transaction.csv (and optional identity) and build a raw graph
        #   - norm : load pre-normalized S-FFSD-like CSV and build a norm graph
        #   - v2   : load v2 prebuilt features + prebuilt graph (entity-gated graph builder)
        #
        # v2 is intended to be generated by data-scripts/prep_ieee_v2.py and will not rebuild here.
        if str(ieee_mode).lower() == "v2":
            v2_dir = os.path.join(prefix, "ieee_v2")
            feat_path = os.path.join(v2_dir, "ieee_v2_train.csv")
            graph_path = os.path.join(v2_dir, "graph-IEEE-v2.bin")

            print(f"[IEEE V2] feature_csv={feat_path} exists={os.path.exists(feat_path)}", flush=True)
            print(f"[IEEE V2] graph_bin={graph_path} exists={os.path.exists(graph_path)}", flush=True)

            if not os.path.exists(feat_path):
                raise FileNotFoundError(
                    f"IEEE v2 features not found: {feat_path}. "
                    f"Run: python3 prep_ieee_v2.py --tx-csv <train_transaction.csv> --id-csv <train_identity.csv> "
                    f"--out-dir {v2_dir} --save-dgl-bin"
                )
            if not os.path.exists(graph_path):
                raise FileNotFoundError(
                    f"IEEE v2 graph bin not found: {graph_path}. "
                    f"Re-run prep_ieee_v2.py with --save-dgl-bin to generate graph-IEEE-v2.bin"
                )

            # Load v2 feature table (already engineered + numeric) and labels
            v2_df = pd.read_csv(feat_path, low_memory=False)
            if "Labels" not in v2_df.columns:
                raise ValueError(f"IEEE v2 features CSV must contain a 'Labels' column: {feat_path}")

            # Keep same label semantics as other loaders
            v2_df = v2_df[v2_df["Labels"] <= 2].reset_index(drop=True)

            labels = v2_df["Labels"].astype(int)

            # Build feature matrix: drop identifiers; keep numeric columns only
            drop_cols = {"Labels", "TransactionID", "node_id"}
            feat_cols = [c for c in v2_df.columns if c not in drop_cols]
            feat_data = v2_df[feat_cols]

            # Coerce any non-numeric to numeric (should be rare for v2)
            obj_cols = [c for c in feat_data.columns if feat_data[c].dtype == "object"]
            for col in obj_cols:
                feat_data[col] = feat_data[col].astype(str).fillna("NA")
                feat_data[col] = feat_data[col].astype("category").cat.codes

            feat_data = feat_data.apply(pd.to_numeric, errors="coerce").fillna(0)

            # Load cached v2 graph (prebuilt by v2 builder)
            t0 = time.time()
            graphs, _ = dgl.data.utils.load_graphs(graph_path)
            g = graphs[0]
            g = _ensure_self_loops(g)
            dt = time.time() - t0

            print(f"[IEEE V2] Loaded cached graph in {dt:.2f}s", flush=True)
            try:
                print(f"[IEEE V2] graph stats: nodes={g.number_of_nodes():,} edges={g.number_of_edges():,}", flush=True)
            except Exception:
                pass

            # Validate node alignment (v2 graphs are transaction-node graphs; nodes must match rows)
            if g.number_of_nodes() != len(labels):
                raise ValueError(
                    f"IEEE v2 graph/node mismatch: graph_nodes={g.number_of_nodes():,} "
                    f"df_rows={len(labels):,}. "
                    f"This usually means the v2 graph was built from a different CSV version. "
                    f"Re-run prep_ieee_v2.py to regenerate both ieee_v2_train.csv and graph-IEEE-v2.bin together."
                )

            # Push features/labels into graph
            g.ndata["label"] = torch.from_numpy(labels.to_numpy()).to(torch.long)
            g.ndata["feat"] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

            # Indices split
            index = list(range(len(labels)))
            train_idx, test_idx, _, _ = train_test_split(
                index, labels, stratify=labels, test_size=test_size / 2,
                random_state=2, shuffle=True
            )

            cat_features = []
            return feat_data, labels, train_idx, test_idx, g, cat_features

        # Decide raw vs norm
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

            # graph cache handling (norm)
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

                    # if cached graph nodes don't match, rebuild
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
                except Exception:
                    pass

            for col in ["Source", "Target", "Location", "Type"]:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].apply(str).values)

            feat_data = data.drop("Labels", axis=1)
            labels = data["Labels"]

            # encode any remaining object columns (defensive)
            obj_cols = [c for c in feat_data.columns if feat_data[c].dtype == "object"]
            for col in obj_cols:
                feat_data[col] = feat_data[col].astype(str).fillna("NA")
                feat_data[col] = feat_data[col].astype("category").cat.codes
                if col not in cat_features:
                    cat_features.append(col)

            feat_data = feat_data.apply(pd.to_numeric, errors="coerce").fillna(0)

            index = list(range(len(labels)))
            g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
            g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

            try:
                dgl.data.utils.save_graphs(os.path.join(prefix, "graph-IEEE-norm.bin"), [g])
            except Exception:
                pass

            train_idx, test_idx, _, _ = train_test_split(
                index, labels, stratify=labels, test_size=test_size / 2,
                random_state=2, shuffle=True
            )

        else:
            # --- Raw IEEE ---
            cat_features = []

            train_tx_path, train_id_path = _find_ieee_files(prefix)
            if train_tx_path is None:
                raise FileNotFoundError(
                    f"Could not find IEEE train_transaction.csv under {prefix}. "
                    f"Expected at {prefix}/train_transaction.csv or similar."
                )

            df = _load_ieee_raw_df(train_tx_path, train_id_path)
            # labels column in raw IEEE
            if "isFraud" not in df.columns:
                raise ValueError("Raw IEEE must contain 'isFraud' column.")

            # build a simple raw graph:
            # connect transactions that share same card1 or addr1 if present (fallback)
            data = df.copy()
            data["Labels"] = data["isFraud"].fillna(0).astype(int)
            data = data[data["Labels"] <= 2].reset_index(drop=True)

            # Graph cache for raw
            cache_path = os.path.join(prefix, "graph-IEEE-raw.bin")
            g = None
            if os.path.exists(cache_path):
                try:
                    print(f"[IEEE GRAPH] raw-cache path={cache_path} exists=True size_bytes={os.path.getsize(cache_path)}", flush=True)
                    print("[IEEE GRAPH] Loading cached graph-IEEE-raw.bin ...", flush=True)
                    t0 = time.time()
                    graphs, _ = dgl.data.utils.load_graphs(cache_path)
                    g = graphs[0]
                    print(f"[IEEE GRAPH] Loaded cached graph in {time.time()-t0:.2f}s", flush=True)
                    print(f"[IEEE GRAPH] loaded-raw stats: nodes={g.number_of_nodes():,} edges={g.number_of_edges():,}", flush=True)
                    if g.number_of_nodes() != len(data):
                        print(f"[IEEE GRAPH] WARNING node count mismatch: graph_nodes={g.number_of_nodes():,} df_rows={len(data):,} -> will rebuild", flush=True)
                        g = None
                except Exception as e:
                    print(f"[IEEE GRAPH] WARNING failed to load cached raw graph ({e}); will rebuild", flush=True)
                    g = None

            if g is None:
                # Basic connectivity: by card1 then addr1 else time chain
                alls, allt = [], []
                key_cols = []
                if "card1" in data.columns:
                    key_cols.append("card1")
                if "addr1" in data.columns:
                    key_cols.append("addr1")

                if key_cols:
                    for col in key_cols:
                        for _, c_df in data.groupby(col):
                            # skip NaNs
                            if pd.isna(_) or _ == "":
                                continue
                            c_df = c_df.sort_values(by="TransactionDT" if "TransactionDT" in c_df.columns else c_df.index)
                            idxs = c_df.index.to_list()
                            for i in range(len(idxs) - 1):
                                alls.append(idxs[i])
                                allt.append(idxs[i + 1])
                else:
                    # chain by transaction time
                    sort_col = "TransactionDT" if "TransactionDT" in data.columns else None
                    gdf = data.sort_values(by=sort_col) if sort_col else data
                    idxs = gdf.index.to_list()
                    for i in range(len(idxs) - 1):
                        alls.append(idxs[i])
                        allt.append(idxs[i + 1])

                g = dgl.graph((np.array(alls), np.array(allt)), num_nodes=len(data))
                g = _ensure_self_loops(g)

                try:
                    dgl.data.utils.save_graphs(cache_path, [g])
                except Exception:
                    pass

            # Build raw features: drop Labels/isFraud/TransactionID; coerce numeric
            labels = data["Labels"]
            drop_cols = ["Labels", "isFraud"]
            if "TransactionID" in data.columns:
                drop_cols.append("TransactionID")
            feat_data = data.drop(columns=[c for c in drop_cols if c in data.columns], axis=1)

            # encode object cols
            obj_cols = [c for c in feat_data.columns if feat_data[c].dtype == "object"]
            for col in obj_cols:
                feat_data[col] = feat_data[col].astype(str).fillna("NA")
                feat_data[col] = feat_data[col].astype("category").cat.codes

            feat_data = feat_data.apply(pd.to_numeric, errors="coerce").fillna(0)

            index = list(range(len(labels)))
            g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
            g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

            train_idx, test_idx, _, _ = train_test_split(
                index, labels, stratify=labels, test_size=test_size / 2,
                random_state=2, shuffle=True
            )

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return feat_data, labels, train_idx, test_idx, g, cat_features
