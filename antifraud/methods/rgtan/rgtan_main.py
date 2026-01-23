# rgtan_main.py
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

try:
    from dgl.dataloading import NodeDataLoader
except ImportError:
    from dgl.dataloading import DataLoader as NodeDataLoader

try:
    from dgl.dataloading import MultiLayerFullNeighborSampler
except Exception:
    # Older DGL versions may not have MultiLayerFullNeighborSampler; use NeighborSampler as a fallback
    from dgl.dataloading import MultiLayerNeighborSampler as MultiLayerFullNeighborSampler


from torch.optim.lr_scheduler import MultiStepLR

from . import *
from .rgtan_lpa import load_lpa_subtensor
from .rgtan_model import RGTAN

try:
    from comet_ml import Experiment
except Exception:
    Experiment = None


def _safe_auc(y_true, y_score):
    try:
        if y_true is None or len(y_true) == 0:
            return float("nan")
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _log_metric(experiment, name, value, step=None):
    if experiment is None:
        return
    try:
        if step is None:
            experiment.log_metric(name, value)
        else:
            experiment.log_metric(name, value, step=step)
    except Exception:
        pass


def _maybe_flush(experiment):
    if experiment is None:
        return
    try:
        experiment.flush()
    except Exception:
        pass


def gen_graph(data: pd.DataFrame, edge_per_trans: int = 3):
    """
    Build a directed transaction graph (transaction-as-node).
    Edges connect each transaction to previous transactions within each entity group.
    """
    alls, allt = [], []
    # Ensure required columns exist
    if "Time" not in data.columns:
        # fallback: sequential
        data = data.copy()
        data["Time"] = np.arange(len(data))

    # Prefer Source/Target/Location if present
    group_cols = []
    for c in ["Source", "Target", "Location"]:
        if c in data.columns:
            group_cols.append(c)

    # If we have group cols, connect previous K within each group
    if group_cols:
        for col in group_cols:
            grp = data.groupby(col, sort=False).indices
            for _, idxs in grp.items():
                if len(idxs) <= 1:
                    continue
                # sort within group by time
                sorted_idxs = sorted(idxs, key=lambda i: data.at[i, "Time"])
                # connect each node to up to K previous within group
                for i in range(len(sorted_idxs)):
                    for j in range(1, edge_per_trans + 1):
                        if i - j >= 0:
                            alls.append(sorted_idxs[i])
                            allt.append(sorted_idxs[i - j])
    else:
        # Fallback: global time edges
        gdf = data.sort_values(by="Time")
        idxs = gdf.index.to_list()
        for i in range(len(idxs)):
            for j in range(1, edge_per_trans + 1):
                if i - j >= 0:
                    alls.append(idxs[i])
                    allt.append(idxs[i - j])

    # Safety: ensure at least one edge
    if len(alls) == 0:
        gdf = data.sort_values(by="Time")
        idxs = gdf.index.to_list()
        for i in range(len(idxs) - 1):
            alls.append(idxs[i])
            allt.append(idxs[i + 1])

    return dgl.graph((np.array(alls), np.array(allt)), num_nodes=len(data))


def run_rgtan(
    feat_df: pd.DataFrame,
    labels: pd.Series,
    train_idx,
    test_idx,
    graph,
    args: dict,
    cat_features=None,
    nei_feat: dict = None,
    neigh_padding_dict: dict = None,
    experiment: Experiment = None,
):
    """
    Key patches in this version:
      1) OOF logits are recomputed using earlystoper.best_model (best checkpoint), per fold.
      2) Test logits are averaged across folds (sum/count), not overwritten by the last fold.
    """
    device = args["device"]
    if cat_features is None:
        cat_features = []
    if nei_feat is None:
        nei_feat = {}
    if neigh_padding_dict is None:
        neigh_padding_dict = {}

    labels = torch.from_numpy(labels.values).long().to(device)
    graph = graph.to(device)

    # Split categorical identity features for embedding tables
    cat_feat = {}
    if cat_features:
        ref_df = feat_df[cat_features].copy()
        for c in cat_features:
            col = ref_df[c]
            if not np.issubdtype(col.dtype, np.integer):
                codes, _ = pd.factorize(col, sort=True)
                ref_df[c] = codes.astype("int64")
            if ref_df[c].isna().any():
                ref_df[c] = ref_df[c].fillna(-1).astype("int64")
            minv = int(ref_df[c].min())
            if minv < 0:
                ref_df[c] = (ref_df[c] - minv).astype("int64")
            cat_feat[c] = torch.from_numpy(ref_df[c].values).long().to(device)
    else:
        ref_df = feat_df

    # numeric features exclude categorical columns
    num_df = feat_df.drop(columns=[c for c in cat_features if c in feat_df.columns], errors="ignore")
    num_df = num_df.select_dtypes(include=[np.number]).fillna(0)
    num_feat = torch.from_numpy(num_df.values).float().to(device)

    # predictions tensors
    oof_predictions = torch.zeros((len(feat_df), 2), dtype=torch.float32, device=device)
    # Fold-averaged test logits (we'll average across folds instead of overwriting)
    test_logits_sum = torch.zeros((len(feat_df), 2), dtype=torch.float32, device=device)
    test_logits_cnt = torch.zeros((len(feat_df), 1), dtype=torch.float32, device=device)

    kfold = StratifiedKFold(n_splits=args["n_fold"], shuffle=True, random_state=args["seed"])

    train_labels_np = labels[train_idx].detach().cpu().numpy()
    train_labels_np = np.where(train_labels_np == 2, 0, train_labels_np)

    loss_fn = nn.CrossEntropyLoss()

    for fold, (trn_ind, val_ind) in enumerate(kfold.split(np.array(train_idx), train_labels_np)):
        print(f"Training fold {fold + 1}", flush=True)

        trn_idx = np.array(train_idx)[trn_ind]
        val_idx = np.array(train_idx)[val_ind]

        trn_ind_t = torch.from_numpy(np.array(trn_idx)).long().to(device)
        val_ind_t_epoch = torch.from_numpy(np.array(val_idx)).long().to(device)

        train_sampler = MultiLayerFullNeighborSampler(args["n_layers"])
        train_dataloader = NodeDataLoader(
            graph,
            trn_ind_t,
            train_sampler,
            use_ddp=False,
            device=device,
            batch_size=args["batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

        val_sampler = MultiLayerFullNeighborSampler(args["n_layers"])
        val_dataloader = NodeDataLoader(
            graph,
            val_ind_t_epoch,
            val_sampler,
            use_ddp=False,
            device=device,
            batch_size=args["batch_size"],
            shuffle=True,  # keep as-is for training-time validation
            drop_last=False,
            num_workers=0,
        )

        # ---- Harden args defaults (avoid KeyError when CLI does not supply optional params)
        if isinstance(args.get('dropout', None), (float, int)):
            args['dropout'] = [float(args['dropout']), float(args['dropout'])]
        if args.get('dropout', None) is None:
            args['dropout'] = [0.2, 0.1]
        if isinstance(args['dropout'], (list, tuple)) and len(args['dropout']) == 1:
            args['dropout'] = [float(args['dropout'][0]), float(args['dropout'][0])]
        args.setdefault('hid_dim', 128)
        args.setdefault('n_layers', 2)
        args.setdefault('gated', True)
        args.setdefault('lr', 0.003)
        args.setdefault('wd', 1e-4)
        args.setdefault('batch_size', 1024)
        args.setdefault('max_epochs', 10)
        args.setdefault('early_stopping', 10)
        args.setdefault('seed', 2023)
        args.setdefault('n_fold', 5)
        # Activation: keep existing if it is an nn.Module; otherwise default to ELU
        if 'activation' not in args or args['activation'] is None or not isinstance(args['activation'], nn.Module):
            args['activation'] = nn.ELU()
        # Heads: allow int or list; ensure list length == n_layers
        if 'heads' not in args or args['heads'] is None:
            args['heads'] = [1] * int(args['n_layers'])
        else:
            h = args['heads']
            if isinstance(h, int):
                args['heads'] = [h] * int(args['n_layers'])
            elif isinstance(h, (list, tuple)):
                h = list(h)
                if len(h) < int(args['n_layers']):
                    h = h + [h[-1]] * (int(args['n_layers']) - len(h))
                args['heads'] = h[: int(args['n_layers'])]
            else:
                args['heads'] = [1] * int(args['n_layers'])

        # Initialize model
        # nei_att_head can be passed by args; fallback to 1
        nei_att_head = args.get("nei_att_head", 1)

        model = RGTAN(
            in_feats=num_feat.shape[1],
            hidden_dim=args["hid_dim"],
            n_layers=args["n_layers"],
            n_classes=2,
            heads=args.get("heads"),
            activation=args["activation"],
            drop=args["dropout"],
            device=device,
            gated=args["gated"],
            ref_df=ref_df,
            cat_features=cat_feat,
            neigh_features=nei_feat,
            nei_att_head=nei_att_head,
        ).to(device)

        lr = args["lr"] * np.sqrt(args["batch_size"] / 1024)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args["wd"])
        lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=[4000, 12000], gamma=0.3)

        earlystoper = early_stopper(patience=args["early_stopping"], verbose=True)

        for epoch in range(args["max_epochs"]):
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

                train_loss_list.append(train_loss.detach().cpu().numpy())

                if step % 10 == 0:
                    score = torch.softmax(logits_.detach(), dim=1)[:, 1].cpu().numpy()
                    yb = batch_labels_.cpu().numpy()
                    try:
                        ap_now = float(average_precision_score(yb, score)) if len(np.unique(yb)) > 1 else 0.0
                    except Exception:
                        ap_now = 0.0
                    print(
                        "In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, train_ap:{:.4f}".format(
                            epoch, step, float(np.mean(train_loss_list)), ap_now
                        ),
                        flush=True,
                    )

            # ---- validation epoch (for early stopping)
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

                    # (legacy behavior) write per-epoch logits; will be overwritten by best-checkpoint recompute
                    oof_predictions[seeds] = val_logits

                    mask = batch_labels == 2
                    val_logits_ = val_logits[~mask]
                    batch_labels_ = batch_labels[~mask]

                    val_loss = loss_fn(val_logits_, batch_labels_)
                    val_loss_sum += float(val_loss.detach().cpu().numpy())
                    val_count += 1

                    # batch diagnostics
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
                                    float(average_precision_score(yb, score)) if len(np.unique(yb)) > 1 else 0.0,
                                    float(val_acc.detach().cpu().numpy()),
                                    _safe_auc(yb, score),
                                ),
                                flush=True,
                            )
                        except Exception:
                            pass

            val_loss_epoch = float(val_loss_sum / max(val_count, 1))

            # Epoch-level AP/AUC computed from whatever is currently in oof_predictions for val nodes
            val_scores = torch.softmax(oof_predictions[val_ind_t_epoch], dim=1)[:, 1].detach().cpu().numpy()
            val_labels_np = labels[val_ind_t_epoch].detach().cpu().numpy()
            m = val_labels_np != 2
            val_scores = val_scores[m]
            val_labels_np = val_labels_np[m]

            val_ap_epoch = float(average_precision_score(val_labels_np, val_scores)) if len(val_labels_np) and len(np.unique(val_labels_np)) > 1 else float("nan")
            val_auc_epoch = _safe_auc(val_labels_np, val_scores)

            _log_metric(experiment, "train_loss", float(np.mean(train_loss_list)) if train_loss_list else 0.0, step=epoch)
            _log_metric(experiment, "val_loss", val_loss_epoch, step=epoch)
            _log_metric(experiment, "val_ap", val_ap_epoch, step=epoch)
            _log_metric(experiment, "val_auc", val_auc_epoch, step=epoch)
            _maybe_flush(experiment)

            earlystoper.earlystop(val_loss_epoch, model)
            if earlystoper.is_earlystop:
                print("Early Stopping!", flush=True)
                break

        print("Best val_loss is: {:.7f}".format(earlystoper.best_cv), flush=True)

        # Recompute out-of-fold (OOF) logits for this fold using the BEST checkpoint,
        # instead of whatever happened to be written during the last validation epoch.
        val_ind_t = torch.from_numpy(np.array(val_idx)).long().to(device)
        oof_sampler = MultiLayerFullNeighborSampler(args["n_layers"])
        oof_dataloader = NodeDataLoader(
            graph,
            val_ind_t,
            oof_sampler,
            use_ddp=False,
            device=device,
            batch_size=args["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        b_model = earlystoper.best_model.to(device)
        b_model.eval()
        with torch.no_grad():
            for step, (input_nodes, seeds, blocks) in enumerate(oof_dataloader):
                batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
                    num_feat, cat_feat, nei_feat, neigh_padding_dict, labels,
                    seeds, input_nodes, device, blocks
                )
                blocks = [b.to(device) for b in blocks]
                val_logits = b_model(blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)
                oof_predictions[seeds] = val_logits
                if step % 50 == 0:
                    print("OOF recompute batch:{:04d}".format(step), flush=True)

        # Test inference for this fold (accumulate logits for fold-averaging)
        test_ind = torch.from_numpy(np.array(test_idx)).long().to(device)
        test_sampler = MultiLayerFullNeighborSampler(args["n_layers"])
        test_dataloader = NodeDataLoader(
            graph,
            test_ind,
            test_sampler,
            use_ddp=False,
            device=device,
            batch_size=args["batch_size"],
            shuffle=False,
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
                test_logits_sum[seeds] += test_logits
                test_logits_cnt[seeds] += 1.0

                if step % 10 == 0:
                    print("In test batch:{:04d}".format(step), flush=True)

    # ---- Final metrics
    # OOF AP on train_idx
    y_train = labels[train_idx].detach().cpu().numpy().copy()
    y_train[y_train == 2] = 0
    oof_scores = torch.softmax(oof_predictions, dim=1).detach().cpu().numpy()[train_idx, 1]
    my_ap = float(average_precision_score(y_train, oof_scores)) if len(np.unique(y_train)) > 1 else float("nan")
    print("NN out of fold AP is:", my_ap, flush=True)

    # Fold-averaged test logits
    test_predictions = test_logits_sum / torch.clamp(test_logits_cnt, min=1.0)

    y_test = labels[test_idx].detach().cpu().numpy()
    test_scores = torch.softmax(test_predictions, dim=1).detach().cpu().numpy()[test_idx, 1]
    test_pred = torch.argmax(test_predictions, dim=1).detach().cpu().numpy()[test_idx]

    # sanitize test labels if 2 used as masked class
    y_test_s = y_test.copy()
    y_test_s[y_test_s == 2] = 0

    test_auc = _safe_auc(y_test_s, test_scores)
    try:
        test_f1 = float(f1_score(y_test_s, test_pred))
    except Exception:
        test_f1 = float("nan")
    try:
        test_ap = float(average_precision_score(y_test_s, test_scores)) if len(np.unique(y_test_s)) > 1 else float("nan")
    except Exception:
        test_ap = float("nan")

    print("test AUC:", test_auc, flush=True)
    print("test f1:", test_f1, flush=True)
    print("test AP:", test_ap, flush=True)

    _log_metric(experiment, "oof_ap", my_ap)
    _log_metric(experiment, "test_auc", test_auc)
    _log_metric(experiment, "test_f1", test_f1)
    _log_metric(experiment, "test_ap", test_ap)
    _maybe_flush(experiment)

    return my_ap, test_auc, test_f1, test_ap


def loda_rgtan_data(dataset: str, test_size: float, ieee_mode: str = "auto"):
    """
    Returns:
      feat_data (pd.DataFrame),
      labels (pd.Series),
      train_idx (list[int]),
      test_idx (list[int]),
      g (dgl graph),
      cat_features (list[str]),
      neigh_features (dict/df)
    """
    # --- S-FFSD
    if dataset == "S-FFSD":
        mat = loadmat("./data/S-FFSD/S-FFSD.mat")
        data = mat["data"]
        labels = data[:, -1].astype(int)
        feat_data = data[:, :-1]
        feat_data = pd.DataFrame(feat_data, columns=["Source", "Target", "Amount", "Location", "Time", "Type"])

        # label encode categorical columns
        le = LabelEncoder()
        for c in ["Source", "Target", "Location", "Type"]:
            feat_data[c] = le.fit_transform(feat_data[c].astype(str))

        labels = pd.Series(labels)
        train_idx, test_idx = train_test_split(
            np.arange(len(feat_data)),
            test_size=test_size,
            random_state=2023,
            stratify=labels.values,
        )
        g = gen_graph(feat_data, edge_per_trans=3)
        cat_features = ["Target", "Location", "Type"]
        neigh_features = {}

        return feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features

    # --- IEEE
    if dataset == "IEEE":

        # Load pre-processed IEEE already mapped to S-FFSD + reuse features
        data = pd.read_csv(
            "./data/ieee_modified/ieee_sffsd_with_reuse.csv"
        )

        # Labels
        labels = data["Labels"].astype(int)

        # Features (everything except label)
        feat_data = data.drop(columns=["Labels"])

        # Train / test split (stratified)
        train_idx, test_idx = train_test_split(
            np.arange(len(feat_data)),
            test_size=test_size,
            random_state=2023,
            stratify=labels.values,
        )

        # Build reuse-aware graph
        # gen_graph already connects within Source / Target / Location by Time
        g = gen_graph(feat_data, edge_per_trans=3)

        # Categorical features for embeddings (CRITICAL for RGTAN)
        cat_features = ["Source", "Target", "Location", "Type"]

        neigh_features = {}

        return (
            feat_data,
            labels,
            train_idx,
            test_idx,
            g,
            cat_features,
            neigh_features,
        )


    raise ValueError(f"Unknown dataset: {dataset}")


def rgtan_main(
    feat_data,
    g,
    train_idx,
    test_idx,
    labels,
    args,
    cat_features,
    neigh_features,
    nei_att_head,
    experiment=None,
):
    """
    Entry point expected by antifraud/main.py.

    main.py calls:
        rgtan_main(feat_data, g, train_idx, test_idx, labels, args, cat_features, neigh_features, nei_att_head, experiment=...)

    This function forwards into run_rgtan(), which implements:
      - OOF logits recomputed using the best checkpoint per fold
      - Test logits averaged across folds
    """
    # Ensure nei_att_head is available to model builder inside run_rgtan()
    if isinstance(args, dict):
        args["nei_att_head"] = nei_att_head

    neigh_padding_dict = {}

    return run_rgtan(
        feat_df=feat_data,
        labels=labels,
        train_idx=train_idx,
        test_idx=test_idx,
        graph=g,
        args=args,
        cat_features=cat_features,
        nei_feat=neigh_features,
        neigh_padding_dict=neigh_padding_dict,
        experiment=experiment,
    )
