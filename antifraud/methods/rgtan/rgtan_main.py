import os
import zipfile
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import dgl

from tqdm import tqdm
from scipy.io import loadmat

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from dgl.dataloading import MultiLayerFullNeighborSampler
try:
    from dgl.dataloading import NodeDataLoader
except ImportError:
    from dgl.dataloading import DataLoader as NodeDataLoader

from torch.optim.lr_scheduler import MultiStepLR

from . import *
from .rgtan_lpa import load_lpa_subtensor
from .rgtan_model import RGTAN

# Optional Comet (safe if not installed)
try:
    from comet_ml import Experiment
except Exception:
    Experiment = None


# -------------------------
# Utilities
# -------------------------
def _ensure_self_loops(g: dgl.DGLGraph) -> dgl.DGLGraph:
    """
    Option A fix for DGL zero-in-degree nodes:
    remove existing self-loops (if any), then add self-loops for all nodes.
    """
    try:
        g = dgl.remove_self_loop(g)
    except Exception:
        pass
    try:
        g = dgl.add_self_loop(g)
    except Exception:
        pass
    return g


def _safe_auc(y_true, y_score):
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _log_metric(experiment, name: str, value, step=None):
    if experiment is None:
        return
    try:
        if step is None:
            experiment.log_metric(name, float(value))
        else:
            experiment.log_metric(name, float(value), step=int(step))
    except Exception:
        pass


def _log_params_once(experiment, args, nei_att_head):
    if experiment is None:
        return
    try:
        experiment.log_parameters({
            "method": "rgtan",
            "dataset": args.get("dataset"),
            "device": args.get("device"),
            "n_fold": args.get("n_fold"),
            "seed": args.get("seed"),
            "batch_size": args.get("batch_size"),
            "n_layers": args.get("n_layers"),
            "hid_dim": args.get("hid_dim"),
            "dropout": args.get("dropout"),
            "gated": args.get("gated"),
            "lr": args.get("lr"),
            "wd": args.get("wd"),
            "early_stopping": args.get("early_stopping"),
            "max_epochs": args.get("max_epochs"),
            "test_size": args.get("test_size"),
            "nei_att_head": nei_att_head,
        })
    except Exception:
        pass


def _maybe_flush(experiment):
    if experiment is None:
        return
    try:
        if hasattr(experiment, "flush"):
            experiment.flush()
    except Exception:
        pass


# -------------------------
# IEEE helpers
# -------------------------
def _find_ieee_files(prefix_dir: str):
    """
    Locate IEEE-CIS files in common layouts.

    Supported:
      - data/train_transaction.csv + data/train_identity.csv
      - data/IEEE/train_transaction.csv + data/IEEE/train_identity.csv
      - data/ieee/train_transaction.csv + data/ieee/train_identity.csv
      - data/ieee-fraud-detection/train_transaction.csv + train_identity.csv
    """
    candidates = [
        (os.path.join(prefix_dir, "train_transaction.csv"), os.path.join(prefix_dir, "train_identity.csv")),
        (os.path.join(prefix_dir, "IEEE", "train_transaction.csv"), os.path.join(prefix_dir, "IEEE", "train_identity.csv")),
        (os.path.join(prefix_dir, "ieee", "train_transaction.csv"), os.path.join(prefix_dir, "ieee", "train_identity.csv")),
        (os.path.join(prefix_dir, "ieee-fraud-detection", "train_transaction.csv"),
         os.path.join(prefix_dir, "ieee-fraud-detection", "train_identity.csv")),
    ]
    for tx_path, id_path in candidates:
        if os.path.exists(tx_path):
            return tx_path, (id_path if os.path.exists(id_path) else None)
    return None, None


def _read_ieee_from_zip(zip_path: str):
    """
    Read IEEE-CIS CSVs from a zip file without extracting.
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        tx_name = None
        id_name = None
        for n in names:
            if n.endswith("train_transaction.csv"):
                tx_name = n
            elif n.endswith("train_identity.csv"):
                id_name = n

        if tx_name is None:
            raise FileNotFoundError("train_transaction.csv not found inside zip")

        with z.open(tx_name) as f:
            tx = pd.read_csv(f)

        identity = None
        if id_name is not None:
            with z.open(id_name) as f:
                identity = pd.read_csv(f)

    return tx, identity


def _build_ieee_graph(df: pd.DataFrame,
                      time_col: str = "TransactionDT",
                      edge_per_trans: int = 3,
                      max_group_size: int = 5000):
    """
    Build a transaction graph for IEEE by connecting transactions that share an entity value,
    ordered by TransactionDT. Caps large groups to avoid edge explosion.
    """
    if time_col not in df.columns:
        raise ValueError(f"IEEE dataframe missing required time column: {time_col}")

    entity_cols = [
        "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2",
        "P_emaildomain", "R_emaildomain",
        "DeviceType", "DeviceInfo",
    ]
    entity_cols = [c for c in entity_cols if c in df.columns]

    alls, allt = [], []
    df = df.reset_index(drop=True)

    for col in entity_cols:
        src, tgt = [], []
        for _, gdf in df.groupby(col, dropna=True):
            if len(gdf) < 2:
                continue
            if len(gdf) > max_group_size:
                gdf = gdf.sample(n=max_group_size, random_state=42)
            gdf = gdf.sort_values(by=time_col)
            sorted_idxs = gdf.index.to_list()
            n = len(sorted_idxs)
            for i in range(n):
                for j in range(1, edge_per_trans + 1):
                    if i + j < n:
                        src.append(sorted_idxs[i])
                        tgt.append(sorted_idxs[i + j])

        if src:
            alls.extend(src)
            allt.extend(tgt)

    # fallback: chain by time if no edges
    if len(alls) == 0:
        gdf = df.sort_values(by=time_col)
        idxs = gdf.index.to_list()
        for i in range(len(idxs) - 1):
            alls.append(idxs[i])
            allt.append(idxs[i + 1])

    g = dgl.graph((np.array(alls), np.array(allt)), num_nodes=len(df))
    return g, entity_cols


# -------------------------
# Training
# -------------------------
def rgtan_main(
    feat_df,
    graph,
    train_idx,
    test_idx,
    labels,
    args,
    cat_features,
    neigh_features: pd.DataFrame,
    nei_att_head,
    experiment: Experiment = None
):
    """
    RGTAN training loop with GTAN-style console logging and Comet epoch metrics.
    """
    device = args["device"]
    graph = graph.to(device)

    _log_params_once(experiment, args, nei_att_head)

    # Buffers for OOF and test predictions
    oof_predictions = torch.from_numpy(np.zeros([len(feat_df), 2])).float().to(device)
    test_predictions = torch.from_numpy(np.zeros([len(feat_df), 2])).float().to(device)

    # K-fold
    kfold = StratifiedKFold(n_splits=args["n_fold"], shuffle=True, random_state=args["seed"])
    y_target = labels.iloc[train_idx].values

    # Feature tensors
    num_feat = torch.from_numpy(feat_df.values).float().to(device)
    cat_feat = {col: torch.from_numpy(feat_df[col].values).long().to(device) for col in cat_features}

    # Neighbor features (optional)
    neigh_padding_dict = {}
    if isinstance(neigh_features, pd.DataFrame):
        nei_feat = {
            col: torch.from_numpy(neigh_features[col].values).to(torch.float32).to(device)
            for col in neigh_features.columns
        }
    else:
        nei_feat = []

    # Labels tensor
    y_series = labels
    labels = torch.from_numpy(y_series.values).long().to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)

    for fold, (trn_idx, val_idx) in enumerate(kfold.split(feat_df.iloc[train_idx], y_target)):
        fold_no = fold + 1
        print(f"Training fold {fold_no}", flush=True)

        trn_ind = torch.from_numpy(np.array(train_idx)[trn_idx]).long().to(device)
        val_ind = torch.from_numpy(np.array(train_idx)[val_idx]).long().to(device)

        # Dataloaders
        train_sampler = MultiLayerFullNeighborSampler(args["n_layers"])
        train_dataloader = NodeDataLoader(
            graph,
            trn_ind,
            train_sampler,
            device=device,
            use_ddp=False,
            batch_size=args["batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

        val_sampler = MultiLayerFullNeighborSampler(args["n_layers"])
        val_dataloader = NodeDataLoader(
            graph,
            val_ind,
            val_sampler,
            use_ddp=False,
            device=device,
            batch_size=args["batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

        # Model
        model = RGTAN(
            in_feats=feat_df.shape[1],
            hidden_dim=args["hid_dim"] // 4,
            n_classes=2,
            heads=[4] * args["n_layers"],
            activation=nn.PReLU(),
            n_layers=args["n_layers"],
            drop=args["dropout"],
            device=device,
            gated=args["gated"],
            ref_df=feat_df,
            cat_features=cat_feat,
            neigh_features=nei_feat,
            nei_att_head=nei_att_head,
        ).to(device)

        # Optim
        lr = args["lr"] * np.sqrt(args["batch_size"] / 1024)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args["wd"])
        lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=[4000, 12000], gamma=0.3)

        earlystoper = early_stopper(patience=args["early_stopping"], verbose=True)

        # Epoch loop
        for epoch in range(args["max_epochs"]):
            # -----------------------
            # Train
            # -----------------------
            model.train()
            train_loss_list = []

            for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
                    num_feat, cat_feat, nei_feat, neigh_padding_dict, labels,
                    seeds, input_nodes, device, blocks
                )

                blocks = [b.to(device) for b in blocks]
                logits = model(blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)

                mask = batch_labels == 2
                logits_ = logits[~mask]
                batch_labels_ = batch_labels[~mask]

                train_loss = loss_fn(logits_, batch_labels_)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                lr_scheduler.step()

                train_loss_list.append(float(train_loss.detach().cpu().numpy()))

                if step % 10 == 0:
                    try:
                        tr_acc = (
                            torch.sum(torch.argmax(logits_.detach(), dim=1) == batch_labels_) / batch_labels_.shape[0]
                        )
                        score = torch.softmax(logits_.detach(), dim=1)[:, 1].cpu().numpy()
                        yb = batch_labels_.cpu().numpy()
                        print(
                            "In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, train_ap:{:.4f}, train_acc:{:.4f}, train_auc:{:.4f}".format(
                                epoch,
                                step,
                                float(np.mean(train_loss_list)),
                                float(average_precision_score(yb, score)),
                                float(tr_acc.detach().cpu().numpy()),
                                _safe_auc(yb, score),
                            ),
                            flush=True
                        )
                    except Exception:
                        pass

            train_loss_epoch = float(np.mean(train_loss_list)) if train_loss_list else 0.0

            # -----------------------
            # Validation
            # -----------------------
            model.eval()
            val_loss_sum = 0.0
            val_count = 0

            with torch.no_grad():
                for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
                    batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
                        num_feat, cat_feat, nei_feat, neigh_padding_dict, labels,
                        seeds, input_nodes, device, blocks
                    )

                    blocks = [b.to(device) for b in blocks]
                    val_logits = model(blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)

                    oof_predictions[seeds] = val_logits

                    mask = batch_labels == 2
                    val_logits_ = val_logits[~mask]
                    batch_labels_ = batch_labels[~mask]

                    loss_val = loss_fn(val_logits_, batch_labels_)
                    bs = int(batch_labels_.shape[0])

                    val_loss_sum += float(loss_val.detach().cpu().numpy()) * bs
                    val_count += bs

                    if step % 10 == 0:
                        try:
                            val_acc = (
                                torch.sum(torch.argmax(val_logits_.detach(), dim=1) == batch_labels_) / batch_labels_.shape[0]
                            )
                            score = torch.softmax(val_logits_.detach(), dim=1)[:, 1].cpu().numpy()
                            yb = batch_labels_.cpu().numpy()
                            print(
                                "In epoch:{:03d}|batch:{:04d}, val_loss:{:4f}, val_ap:{:.4f}, val_acc:{:.4f}, val_auc:{:.4f}".format(
                                    epoch,
                                    step,
                                    float(val_loss_sum / max(val_count, 1)),
                                    float(average_precision_score(yb, score)),
                                    float(val_acc.detach().cpu().numpy()),
                                    _safe_auc(yb, score),
                                ),
                                flush=True
                            )
                        except Exception:
                            pass

            val_loss_epoch = float(val_loss_sum / max(val_count, 1))

            # Full-val metrics on val_ind
            val_scores = torch.softmax(oof_predictions[val_ind], dim=1)[:, 1].detach().cpu().numpy()
            val_labels_np = labels[val_ind].detach().cpu().numpy()
            m = val_labels_np != 2
            val_scores = val_scores[m]
            val_labels_np = val_labels_np[m]

            val_ap_epoch = float(average_precision_score(val_labels_np, val_scores)) if len(val_labels_np) else float("nan")
            val_auc_epoch = _safe_auc(val_labels_np, val_scores)

            # Comet per epoch
            _log_metric(experiment, "train_loss", train_loss_epoch, step=epoch)
            _log_metric(experiment, "val_loss", val_loss_epoch, step=epoch)
            _log_metric(experiment, "val_ap", val_ap_epoch, step=epoch)
            _log_metric(experiment, "val_auc", val_auc_epoch, step=epoch)
            _maybe_flush(experiment)

            earlystoper.earlystop(val_loss_epoch, model)
            if earlystoper.is_earlystop:
                print("Early Stopping!", flush=True)
                break

        print("Best val_loss is: {:.7f}".format(earlystoper.best_cv), flush=True)

        # -----------------------
        # Test inference (best model)
        # -----------------------
        test_ind = torch.from_numpy(np.array(test_idx)).long().to(device)
        test_sampler = MultiLayerFullNeighborSampler(args["n_layers"])
        test_dataloader = NodeDataLoader(
            graph,
            test_ind,
            test_sampler,
            use_ddp=False,
            device=device,
            batch_size=args["batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

        b_model = earlystoper.best_model.to(device)
        b_model.eval()
        with torch.no_grad():
            for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
                batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
                    num_feat, cat_feat, nei_feat, neigh_padding_dict, labels,
                    seeds, input_nodes, device, blocks
                )

                blocks = [b.to(device) for b in blocks]
                test_logits = b_model(blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)
                test_predictions[seeds] = test_logits

                if step % 10 == 0:
                    print("In test batch:{:04d}".format(step), flush=True)

    # -----------------------
    # Final metrics
    # -----------------------
    y_train = labels[train_idx].detach().cpu().numpy().copy()
    y_train[y_train == 2] = 0
    oof_scores = torch.softmax(oof_predictions, dim=1).detach().cpu().numpy()[train_idx, 1]
    my_ap = float(average_precision_score(y_train, oof_scores))
    print("NN out of fold AP is:", my_ap, flush=True)

    y_test = labels[test_idx].detach().cpu().numpy()
    test_scores = torch.softmax(test_predictions, dim=1).detach().cpu().numpy()[test_idx, 1]
    test_pred = torch.argmax(test_predictions, dim=1).detach().cpu().numpy()[test_idx]

    m = y_test != 2
    y_test = y_test[m]
    test_scores = test_scores[m]
    test_pred = test_pred[m]

    test_auc = float("nan") if len(np.unique(y_test)) < 2 else float(roc_auc_score(y_test, test_scores))
    test_f1 = float(f1_score(y_test, test_pred, average="macro"))
    test_ap = float(average_precision_score(y_test, test_scores))

    print("test AUC:", test_auc, flush=True)
    print("test f1:", test_f1, flush=True)
    print("test AP:", test_ap, flush=True)

    _log_metric(experiment, "oof_ap", my_ap)
    _log_metric(experiment, "test_auc", test_auc)
    _log_metric(experiment, "test_f1", test_f1)
    _log_metric(experiment, "test_ap", test_ap)
    _maybe_flush(experiment)


# -------------------------
# Data loaders
# -------------------------
def loda_rgtan_data(dataset: str, test_size: float):
    """
    Returns:
      feat_data (pd.DataFrame),
      labels (pd.Series),
      train_idx (list[int]),
      test_idx (list[int]),
      g (dgl graph),
      cat_features (list[str]),
      neigh_features (pd.DataFrame or [])
    """
    prefix = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data")) + os.sep

    if dataset == 'S-FFSD':
        cat_features = ["Target", "Location", "Type"]

        df = pd.read_csv(prefix + "S-FFSDneofull.csv")
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]

        data = df[df["Labels"] <= 2].reset_index(drop=True)

        alls, allt = [], []
        pair = ["Source", "Target", "Location", "Type"]

        for column in pair:
            src, tgt = [], []
            edge_per_trans = 3
            for _, c_df in tqdm(data.groupby(column), desc=column):
                c_df = c_df.sort_values(by="Time")
                df_len = len(c_df)
                sorted_idxs = c_df.index
                src.extend([sorted_idxs[i] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
                tgt.extend([sorted_idxs[i + j] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
            alls.extend(src)
            allt.extend(tgt)

        g = dgl.graph((np.array(alls), np.array(allt)))
        g = _ensure_self_loops(g)

        for col in ["Source", "Target", "Location", "Type"]:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].apply(str).values)

        feat_data = data.drop("Labels", axis=1)
        labels = data["Labels"]

        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

        try:
            dgl.data.utils.save_graphs(prefix + f"graph-{dataset}.bin", [g])
        except Exception:
            pass

        index = list(range(len(labels)))
        train_idx, test_idx, _, _ = train_test_split(
            index, labels, stratify=labels, test_size=0.6,
            random_state=2, shuffle=True
        )

        # Neighbor features (as in original S-FFSD flow)
        neigh_features = []
        neigh_path = prefix + "S-FFSD_neigh_feat.csv"
        if os.path.exists(neigh_path):
            neigh_features = pd.read_csv(neigh_path)
            print("neighborhood feature loaded for nn input.", flush=True)

    elif dataset in ("IEEE", "IEEE-CIS", "ieee", "ieeecis"):
        cat_features = []
        neigh_features = []  # optional; keep empty unless you generate a neighbor-feature table

        zip_path = os.path.join(prefix, "ieee-fraud-detection.zip")
        tx_path, id_path = _find_ieee_files(prefix)

        if os.path.exists(zip_path):
            tx, identity = _read_ieee_from_zip(zip_path)
        elif tx_path is not None:
            tx = pd.read_csv(tx_path)
            identity = pd.read_csv(id_path) if id_path is not None else None
        else:
            raise FileNotFoundError(
                "IEEE dataset not found. Place either:\n"
                f"  - {zip_path}\n"
                "  - data/train_transaction.csv (+ train_identity.csv)\n"
                "  - data/IEEE/train_transaction.csv (+ train_identity.csv)\n"
            )

        # merge identity if present
        if identity is not None and "TransactionID" in tx.columns and "TransactionID" in identity.columns:
            df = tx.merge(identity, on="TransactionID", how="left")
        else:
            df = tx.copy()

        if "isFraud" not in df.columns:
            raise ValueError("IEEE train_transaction.csv must include 'isFraud' column")

        labels = df["isFraud"].astype(int)

        # drop obvious non-features
        drop_cols = [c for c in ["TransactionID"] if c in df.columns]
        feat_df = df.drop(columns=drop_cols)

        # Build graph + self loops (Option A)
        g, _ = _build_ieee_graph(
            feat_df, time_col="TransactionDT", edge_per_trans=3, max_group_size=5000
        )
        g = _ensure_self_loops(g)

        # Encode object columns as categorical
        obj_cols = [c for c in feat_df.columns if feat_df[c].dtype == "object"]
        for col in obj_cols:
            le = LabelEncoder()
            feat_df[col] = le.fit_transform(feat_df[col].astype(str).fillna("NA").values)
            cat_features.append(col)

        # Fill numeric NaNs
        for col in feat_df.columns:
            if col in cat_features:
                feat_df[col] = feat_df[col].fillna(0).astype(int)
            else:
                if pd.api.types.is_numeric_dtype(feat_df[col]):
                    med = feat_df[col].median()
                    if pd.isna(med):
                        med = 0.0
                    feat_df[col] = feat_df[col].fillna(med)
                else:
                    feat_df[col] = feat_df[col].fillna(0)

        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(feat_df.to_numpy()).to(torch.float32)

        try:
            dgl.data.utils.save_graphs(os.path.join(prefix, "graph-IEEE.bin"), [g])
        except Exception:
            pass

        index = list(range(len(labels)))
        train_idx, test_idx, _, _ = train_test_split(
            index, labels, stratify=labels, test_size=test_size,
            random_state=2, shuffle=True
        )

        feat_data = feat_df

    elif dataset == 'yelp':
        cat_features = []
        neigh_features = []

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
        for i in homo:
            for j in homo[i]:
                src.append(i)
                tgt.append(j)

        g = dgl.graph((np.array(src), np.array(tgt)))
        g = _ensure_self_loops(g)

        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

        try:
            dgl.data.utils.save_graphs(prefix + f"graph-{dataset}.bin", [g])
        except Exception:
            pass

        neigh_path = prefix + "yelp_neigh_feat.csv"
        if os.path.exists(neigh_path):
            neigh_features = pd.read_csv(neigh_path)
            print("neighborhood feature loaded for nn input.", flush=True)
        else:
            print("no neighbohood feature used.", flush=True)

    elif dataset == 'amazon':
        cat_features = []
        neigh_features = []

        data_file = loadmat(prefix + 'Amazon.mat')
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)

        with open(prefix + 'amz_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)

        index = list(range(3305, len(labels)))
        train_idx, test_idx, _, _ = train_test_split(
            index, labels[3305:], stratify=labels[3305:],
            test_size=test_size, random_state=2, shuffle=True
        )

        src, tgt = [], []
        for i in homo:
            for j in homo[i]:
                src.append(i)
                tgt.append(j)

        g = dgl.graph((np.array(src), np.array(tgt)))
        g = _ensure_self_loops(g)

        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

        try:
            dgl.data.utils.save_graphs(prefix + f"graph-{dataset}.bin", [g])
        except Exception:
            pass

        neigh_path = prefix + "amazon_neigh_feat.csv"
        if os.path.exists(neigh_path):
            neigh_features = pd.read_csv(neigh_path)
            print("neighborhood feature loaded for nn input.", flush=True)
        else:
            print("no neighbohood feature used.", flush=True)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features
