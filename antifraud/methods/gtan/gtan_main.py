import numpy as np
import dgl
import torch
import os
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

# ✅ Correct GTAN imports (NOT RGTAN)
from .gtan_model import GraphAttnModel
from .gtan_lpa import load_lpa_subtensor

# pulls in early_stopper and any other shared utils you already had
from . import *

# Optional Comet (safe if not installed)
try:
    from comet_ml import Experiment
except Exception:
    Experiment = None


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

    # Comet: log static run params once
    if experiment is not None:
        try:
            experiment.log_parameters({
                "method": "gtan",
                "dataset": args.get("dataset"),
                "device": args.get("device"),
                "n_fold": args.get("n_fold"),
                "seed": args.get("seed"),
                "batch_size": args.get("batch_size"),
                "n_layers": args.get("n_layers"),
                "hid_dim": args.get("hid_dim"),
                "dropout": args.get("dropout"),
                "gated": args.get("gated"),
                "lr_base": args.get("lr"),
                "wd": args.get("wd"),
                "early_stopping": args.get("early_stopping"),
                "max_epochs": args.get("max_epochs"),
                "test_size": args.get("test_size"),
            })
        except Exception:
            pass

    for fold, (trn_idx, val_idx) in enumerate(kfold.split(feat_df.iloc[train_idx], y_target)):
        print(f'Training fold {fold + 1}')

        if experiment is not None:
            try:
                experiment.log_other("fold", fold + 1)
            except Exception:
                pass

        trn_ind = torch.from_numpy(np.array(train_idx)[trn_idx]).long().to(device)
        val_ind = torch.from_numpy(np.array(train_idx)[val_idx]).long().to(device)

        train_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        train_dataloader = NodeDataLoader(
            graph,
            trn_ind,
            train_sampler,
            device=device,
            use_ddp=False,
            batch_size=args['batch_size'],
            shuffle=True,
            drop_last=False,
            num_workers=0
        )

        val_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        val_dataloader = NodeDataLoader(
            graph,
            val_ind,
            val_sampler,
            use_ddp=False,
            device=device,
            batch_size=args['batch_size'],
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

        model = GraphAttnModel(
            in_feats=feat_df.shape[1],
            hidden_dim=args['hid_dim'] // 4,
            n_classes=2,
            heads=[4] * args['n_layers'],
            activation=nn.PReLU(),
            n_layers=args['n_layers'],
            drop=args['dropout'],
            device=device,
            gated=args['gated'],
            ref_df=feat_df,
            cat_features=cat_feat
        ).to(device)

        lr = args['lr'] * np.sqrt(args['batch_size'] / 1024)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args['wd'])
        lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=[4000, 12000], gamma=0.3)

        if experiment is not None:
            try:
                experiment.log_other("lr_scaled", float(lr))
            except Exception:
                pass

        earlystoper = early_stopper(patience=args['early_stopping'], verbose=True)

        for epoch in range(args['max_epochs']):
            # ---- Train
            train_loss_list = []
            model.train()

            for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
                    num_feat, cat_feat, labels, seeds, input_nodes, device
                )

                blocks = [block.to(device) for block in blocks]
                train_batch_logits = model(blocks, batch_inputs, lpa_labels, batch_work_inputs)

                mask = batch_labels == 2
                train_batch_logits = train_batch_logits[~mask]
                batch_labels = batch_labels[~mask]

                train_loss = loss_fn(train_batch_logits, batch_labels)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                lr_scheduler.step()

                train_loss_list.append(float(train_loss.detach().cpu().numpy()))

                if step % 10 == 0:
                    tr_batch_pred = (
                        torch.sum(torch.argmax(train_batch_logits.detach(), dim=1) == batch_labels)
                        / batch_labels.shape[0]
                    )
                    score = torch.softmax(train_batch_logits.detach(), dim=1)[:, 1].cpu().numpy()

                    try:
                        print(
                            'In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, '
                            'train_ap:{:.4f}, train_acc:{:.4f}, train_auc:{:.4f}'.format(
                                epoch, step,
                                float(np.mean(train_loss_list)),
                                average_precision_score(batch_labels.cpu().numpy(), score),
                                tr_batch_pred.detach(),
                                roc_auc_score(batch_labels.cpu().numpy(), score)
                            )
                        )
                    except Exception:
                        pass

            # Comet epoch-level train loss
            if experiment is not None:
                try:
                    experiment.log_metric("train_loss", float(np.mean(train_loss_list)), step=epoch)
                except Exception:
                    pass

            # ---- Validation
            val_loss_sum = 0.0
            val_count = 0
            model.eval()

            with torch.no_grad():
                for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
                    batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
                        num_feat, cat_feat, labels, seeds, input_nodes, device
                    )

                    blocks = [block.to(device) for block in blocks]
                    val_batch_logits = model(blocks, batch_inputs, lpa_labels, batch_work_inputs)

                    oof_predictions[seeds] = val_batch_logits

                    mask = batch_labels == 2
                    val_batch_logits = val_batch_logits[~mask]
                    batch_labels = batch_labels[~mask]

                    loss_val = loss_fn(val_batch_logits, batch_labels)
                    bs = int(batch_labels.shape[0])

                    val_loss_sum += float(loss_val.detach().cpu().numpy()) * bs
                    val_count += bs

                    if step % 10 == 0:
                        score = torch.softmax(val_batch_logits.detach(), dim=1)[:, 1].cpu().numpy()
                        try:
                            print(
                                'In epoch:{:03d}|batch:{:04d}, val_loss:{:4f}, val_ap:{:.4f}, '
                                'val_auc:{:.4f}'.format(
                                    epoch, step,
                                    (val_loss_sum / max(val_count, 1)),
                                    average_precision_score(batch_labels.cpu().numpy(), score),
                                    roc_auc_score(batch_labels.cpu().numpy(), score)
                                )
                            )
                        except Exception:
                            pass

            val_loss_epoch = val_loss_sum / max(val_count, 1)

            # Comet epoch-level val loss + epoch-level val AUC/AP (using full val_ind)
            if experiment is not None:
                try:
                    experiment.log_metric("val_loss", float(val_loss_epoch), step=epoch)

                    val_scores = torch.softmax(oof_predictions[val_ind], dim=1)[:, 1].detach().cpu().numpy()
                    val_labels_np = labels[val_ind].detach().cpu().numpy()

                    mask = val_labels_np != 2
                    val_scores = val_scores[mask]
                    val_labels_np = val_labels_np[mask]

                    if len(np.unique(val_labels_np)) > 1:
                        experiment.log_metric("val_auc", float(roc_auc_score(val_labels_np, val_scores)), step=epoch)
                    experiment.log_metric("val_ap", float(average_precision_score(val_labels_np, val_scores)), step=epoch)
                except Exception:
                    pass

            earlystoper.earlystop(val_loss_epoch, model)
            if earlystoper.is_earlystop:
                print("Early Stopping!")
                if experiment is not None:
                    try:
                        experiment.log_other(f"early_stop_epoch_fold_{fold+1}", epoch)
                    except Exception:
                        pass
                break

        print("Best val_loss is: {:.7f}".format(earlystoper.best_cv))

        # ---- Test (per fold)
        test_ind = torch.from_numpy(np.array(test_idx)).long().to(device)
        test_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        test_dataloader = NodeDataLoader(
            graph,
            test_ind,
            test_sampler,
            use_ddp=False,
            device=device,
            batch_size=args['batch_size'],
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

        b_model = earlystoper.best_model.to(device)
        b_model.eval()
        with torch.no_grad():
            for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
                batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
                    num_feat, cat_feat, labels, seeds, input_nodes, device
                )

                blocks = [block.to(device) for block in blocks]
                test_batch_logits = b_model(blocks, batch_inputs, lpa_labels, batch_work_inputs)
                test_predictions[seeds] = test_batch_logits

                if step % 10 == 0:
                    print('In test batch:{:04d}'.format(step))

    # ---- Final metrics (overall)
    mask = y_target == 2
    y_target[mask] = 0

    my_ap = average_precision_score(
        y_target,
        torch.softmax(oof_predictions, dim=1).detach().cpu().numpy()[train_idx, 1]
    )
    print("NN out of fold AP is:", my_ap)

    if experiment is not None:
        try:
            experiment.log_metric("oof_ap", float(my_ap))
        except Exception:
            pass

    test_score = torch.softmax(test_predictions, dim=1)[test_idx, 1].detach().cpu().numpy()
    y_test = labels[test_idx].detach().cpu().numpy()
    test_pred = torch.argmax(test_predictions, dim=1)[test_idx].detach().cpu().numpy()

    mask = y_test != 2
    test_score = test_score[mask]
    y_test = y_test[mask]
    test_pred = test_pred[mask]

    test_auc = roc_auc_score(y_test, test_score)
    test_f1 = f1_score(y_test, test_pred, average="macro")
    test_ap = average_precision_score(y_test, test_score)

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


def load_gtan_data(dataset: str, test_size: float):
    """
    Load graph, feature, and label given dataset name
    :param dataset: the dataset name
    :param test_size: the size of test set
    :returns: feature, label, graph, category features
    """
    prefix = os.path.join(os.path.dirname(__file__), "..", "..", "data/")

    if dataset == "S-FFSD":
        cat_features = ["Target", "Location", "Type"]

        df = pd.read_csv(prefix + "S-FFSDneofull.csv")
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        data = df[df["Labels"] <= 2]
        data = data.reset_index(drop=True)

        alls = []
        allt = []
        pair = ["Source", "Target", "Location", "Type"]
        for column in pair:
            src, tgt = [], []
            edge_per_trans = 3
            for _, c_df in data.groupby(column):
                c_df = c_df.sort_values(by="Time")
                df_len = len(c_df)
                sorted_idxs = c_df.index
                src.extend([sorted_idxs[i] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
                tgt.extend([sorted_idxs[i + j] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
            alls.extend(src)
            allt.extend(tgt)

        alls = np.array(alls)
        allt = np.array(allt)
        g = dgl.graph((alls, allt))

        cal_list = ["Source", "Target", "Location", "Type"]
        for col in cal_list:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].apply(str).values)

        feat_data = data.drop("Labels", axis=1)
        labels = data["Labels"]

        feat_data.to_csv(prefix + "S-FFSD_feat_data.csv", index=None)
        labels.to_csv(prefix + "S-FFSD_label_data.csv", index=None)

        index = list(range(len(labels)))
        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])

        train_idx, test_idx, _, _ = train_test_split(
            index, labels, stratify=labels, test_size=test_size / 2,
            random_state=2, shuffle=True
        )

    elif dataset == "yelp":
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
        for i in homo:
            for j in homo[i]:
                src.append(i)
                tgt.append(j)

        g = dgl.graph((np.array(src), np.array(tgt)))
        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

        graph_path = prefix + "graph-{}.bin".format(dataset)
        dgl.data.utils.save_graphs(graph_path, [g])

    elif dataset == "amazon":
        cat_features = []
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

    return feat_data, labels, train_idx, test_idx, g, cat_features
