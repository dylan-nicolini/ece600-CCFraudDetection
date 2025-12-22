import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import pickle

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


def _comet_log_metric(experiment, name: str, value, step=None, *, silent: bool = False):
    """Log a numeric metric to Comet with optional step.

    We do NOT fully swallow failures by default, because silent failures are
    exactly what makes RGTAN appear to have 'blank' metrics in Comet.
    """
    if experiment is None:
        return
    try:
        if step is None:
            experiment.log_metric(name, float(value))
        else:
            experiment.log_metric(name, float(value), step=int(step))
    except Exception as e:
        if not silent:
            print(f"[RGTAN] Comet log_metric failed: {name}={value} step={step} err={repr(e)}", flush=True)


def _comet_log_other(experiment, name: str, value, *, silent: bool = True):
    if experiment is None:
        return
    try:
        experiment.log_other(name, value)
    except Exception as e:
        if not silent:
            print(f"[RGTAN] Comet log_other failed: {name}={value} err={repr(e)}", flush=True)


def _comet_log_parameters_once(experiment, args, nei_att_head):
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
    except Exception as e:
        # Parameters are non-critical; keep silent unless debugging.
        print(f"[RGTAN] Comet log_parameters failed: {repr(e)}", flush=True)


def _safe_auc(y_true, y_score):
    # AUC only valid if both classes exist
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


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
    print(f"[RGTAN] Comet experiment is None? {experiment is None}", flush=True)

    if experiment is not None:
        try:
            print(f"[RGTAN] Comet experiment disabled? {getattr(experiment, 'disabled', 'unknown')}", flush=True)
            experiment.log_other("rgtan_alive", True)
            experiment.log_metric("rgtan_heartbeat", 1.0)
            print("[RGTAN] Wrote rgtan_alive + rgtan_heartbeat to Comet", flush=True)
        except Exception as e:
            print(f"[RGTAN] Comet logging ERROR: {repr(e)}", flush=True)

    """
    RGTAN with GTAN-standard logging:

    Console (every 10 batches):
      - In epoch|batch ... train_loss, train_ap, train_acc, train_auc
      - In epoch|batch ... val_loss, val_ap, val_acc, val_auc

    Comet per epoch (step=epoch):
      - train_loss
      - val_loss
      - val_ap
      - val_auc

    Comet final:
      - oof_ap
      - test_auc
      - test_f1
      - test_ap
    """
    device = args["device"]
    graph = graph.to(device)

    _comet_log_parameters_once(experiment, args, nei_att_head)

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
        print(f"Training fold {fold + 1}", flush=True)
        _comet_log_other(experiment, "fold", fold + 1, silent=False)

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
        _comet_log_other(experiment, "lr_scaled", float(lr), silent=False)
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

                # GTAN-style console output
                if step % 10 == 0:
                    try:
                        tr_batch_pred = (
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
                                float(tr_batch_pred.detach().cpu().numpy()),
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

                    # store for full val metrics
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
                            val_batch_pred = (
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
                                    float(val_batch_pred.detach().cpu().numpy()),
                                    _safe_auc(yb, score),
                                ),
                                flush=True
                            )
                        except Exception:
                            pass

            val_loss_epoch = float(val_loss_sum / max(val_count, 1))

            # Full-val metrics (GTAN standard)
            val_scores = torch.softmax(oof_predictions[val_ind], dim=1)[:, 1].detach().cpu().numpy()
            val_labels_np = labels[val_ind].detach().cpu().numpy()
            mask = val_labels_np != 2
            val_scores = val_scores[mask]
            val_labels_np = val_labels_np[mask]

            val_ap_epoch = float(average_precision_score(val_labels_np, val_scores)) if len(val_labels_np) else float("nan")
            val_auc_epoch = _safe_auc(val_labels_np, val_scores)

            # -----------------------
            # Comet: GTAN-standard per-epoch logging (step=epoch)
            # (Direct + non-silent failure reporting)
            # -----------------------
            _comet_log_metric(experiment, "train_loss", train_loss_epoch, step=epoch, silent=False)
            _comet_log_metric(experiment, "val_loss", val_loss_epoch, step=epoch, silent=False)
            _comet_log_metric(experiment, "val_ap", val_ap_epoch, step=epoch, silent=False)
            _comet_log_metric(experiment, "val_auc", val_auc_epoch, step=epoch, silent=False)
            _comet_log_metric(experiment, "epoch_alive", 1.0, step=epoch, silent=True)

            # Early stopping uses val_loss (same idea as original / GTAN)
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
    # Final metrics (GTAN standard keys)
    # -----------------------
    y_train = labels[train_idx].detach().cpu().numpy().copy()
    y_train[y_train == 2] = 0
    oof_scores = torch.softmax(oof_predictions, dim=1).detach().cpu().numpy()[train_idx, 1]
    my_ap = float(average_precision_score(y_train, oof_scores))
    print("NN out of fold AP is:", my_ap, flush=True)

    y_test = labels[test_idx].detach().cpu().numpy()
    test_scores = torch.softmax(test_predictions, dim=1).detach().cpu().numpy()[test_idx, 1]
    test_pred = torch.argmax(test_predictions, dim=1).detach().cpu().numpy()[test_idx]

    mask = y_test != 2
    y_test = y_test[mask]
    test_scores = test_scores[mask]
    test_pred = test_pred[mask]

    test_auc = float("nan") if len(np.unique(y_test)) < 2 else float(roc_auc_score(y_test, test_scores))
    test_f1 = float(f1_score(y_test, test_pred, average="macro"))
    test_ap = float(average_precision_score(y_test, test_scores))

    print("test AUC:", test_auc, flush=True)
    print("test f1:", test_f1, flush=True)
    print("test AP:", test_ap, flush=True)

    _comet_log_metric(experiment, "oof_ap", my_ap, silent=False)
    _comet_log_metric(experiment, "test_auc", test_auc, silent=False)
    _comet_log_metric(experiment, "test_f1", test_f1, silent=False)
    _comet_log_metric(experiment, "test_ap", test_ap, silent=False)


def loda_rgtan_data(dataset: str, test_size: float):
    prefix = "data/"

    if dataset == 'S-FFSD':
        cat_features = ["Target", "Location", "Type"]

        df = pd.read_csv(prefix + "S-FFSDneofull.csv")
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]

        neigh_features = []
        data = df[df["Labels"] <= 2]
        data = data.reset_index(drop=True)

        alls = []
        allt = []
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

        cal_list = ["Source", "Target", "Location", "Type"]
        for col in cal_list:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].apply(str).values)

        feat_data = data.drop("Labels", axis=1)
        labels = data["Labels"]

        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])

        index = list(range(len(labels)))
        train_idx, test_idx, _, _ = train_test_split(
            index, labels, stratify=labels, test_size=0.6,
            random_state=2, shuffle=True
        )

        feat_neigh = pd.read_csv(prefix + "S-FFSD_neigh_feat.csv")
        print("neighborhood feature loaded for nn input.", flush=True)
        neigh_features = feat_neigh

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
        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])

        try:
            feat_neigh = pd.read_csv(prefix + "yelp_neigh_feat.csv")
            print("neighborhood feature loaded for nn input.", flush=True)
            neigh_features = feat_neigh
        except Exception:
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
        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])

        try:
            feat_neigh = pd.read_csv(prefix + "amazon_neigh_feat.csv")
            print("neighborhood feature loaded for nn input.", flush=True)
            neigh_features = feat_neigh
        except Exception:
            print("no neighbohood feature used.", flush=True)

    return feat_data, labels, train_idx, test_idx, g, cat_features, neigh_features
