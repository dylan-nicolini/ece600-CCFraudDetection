#!/usr/bin/env python3
"""
ieee_option4_hgt_comet_sampling.py

IEEE-CIS Option 4 hetero graph + HGT + neighbor sampling + Comet logging.

ZIP input must contain:
  - train_transaction.csv
  - train_identity.csv

Graph Option 4:
  Node types: transaction, card, address, device, browser, os, screen, product
  Edge types (entity -> transaction):
    card-made->transaction
    address-at->transaction
    device-used->transaction
    browser-ua->transaction
    os-os->transaction
    screen-screen->transaction
    product-type->transaction

Split:
  50/50 time-based split on TransactionDT

Training:
  - Neighbor sampling with MultiLayerNeighborSampler
  - Seed nodes are transaction nodes
  - Predict labels for seed transaction nodes
  - Logs AP, AUC, F1 and confusion matrix to Comet
"""

import argparse
import json
import os
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.dataloading import MultiLayerNeighborSampler, DataLoader

from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, confusion_matrix


# -----------------------------
# Optional Comet import
# -----------------------------
def try_create_comet_experiment(enabled: bool, project: str, workspace: str, tags: list, name: str):
    if not enabled:
        return None, "comet disabled"
    try:
        from comet_ml import Experiment  # type: ignore
    except Exception as e:
        return None, f"comet_ml not available: {e}"

    api_key = os.getenv("COMET_API_KEY", "").strip()
    if not api_key:
        return None, "COMET_API_KEY env var not set"

    try:
        exp = Experiment(
            api_key=api_key,
            project_name=project,
            workspace=workspace if workspace else None,
            auto_output_logging="simple",
            auto_metric_logging=False,
            auto_param_logging=False,
        )
        if name:
            exp.set_name(name)
        if tags:
            exp.add_tags(tags)
        return exp, "comet enabled"
    except Exception as e:
        return None, f"failed to create comet experiment: {e}"


# -----------------------------
# Helpers: incremental mapping
# -----------------------------
class IdMapper:
    def __init__(self):
        self.map = {}
        self.next_id = 0

    def get(self, key: str) -> int:
        if key in self.map:
            return self.map[key]
        i = self.next_id
        self.map[key] = i
        self.next_id += 1
        return i

    def __len__(self):
        return self.next_id


def safe_str(x) -> str:
    if pd.isna(x):
        return "__MISSING__"
    return str(x)


def zread_csv(zf: zipfile.ZipFile, name: str, usecols=None, chunksize=None):
    with zf.open(name) as f:
        return pd.read_csv(f, usecols=usecols, chunksize=chunksize)


# -----------------------------
# Model: HGT for blocks (neighbor sampling)
# -----------------------------
class HGTBlockClassifier(nn.Module):
    """
    HGTConv over sampled blocks.

    - transaction nodes: numeric features projected to hidden_dim
    - other node types: embedding lookup by node ID
    """
    def __init__(self, full_g: dgl.DGLHeteroGraph, tx_in_dim: int, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Canonical ntypes/etypes based on full graph
        self.ntypes = full_g.ntypes
        self.etypes = full_g.canonical_etypes

        if not hasattr(dgl.nn, "HGTConv"):
            raise RuntimeError("dgl.nn.HGTConv not found. Upgrade DGL or use R-GCN.")

        self.ntype2id = {n: i for i, n in enumerate(self.ntypes)}
        self.etype2id = {e: i for i, e in enumerate(self.etypes)}

        self.tx_proj = nn.Linear(tx_in_dim, hidden_dim)

        # Embeddings for entity node types
        self.emb = nn.ModuleDict()
        for ntype in self.ntypes:
            if ntype == "transaction":
                continue
            self.emb[ntype] = nn.Embedding(full_g.num_nodes(ntype), hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                dgl.nn.HGTConv(
                    in_size=hidden_dim,
                    head_size=hidden_dim // num_heads,
                    num_heads=num_heads,
                    num_ntypes=len(self.ntypes),
                    num_etypes=len(self.etypes),
                    dropout=dropout,
                    use_norm=True,
                )
            )

        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def _input_features_for_block(self, block: dgl.DGLHeteroGraph, full_g: dgl.DGLHeteroGraph) -> dict:
        """
        Build input feature dict for a block's SRC nodes.
        DGL blocks store original node IDs in block.srcnodes[ntype].data[dgl.NID]
        """
        h = {}
        device = block.device

        for ntype in block.srctypes:
            src_nids = block.srcnodes[ntype].data[dgl.NID]  # original node IDs
            if ntype == "transaction":
                tx_feat = full_g.nodes["transaction"].data["feat"][src_nids].to(device)
                h["transaction"] = self.tx_proj(tx_feat)
            else:
                h[ntype] = self.emb[ntype](src_nids.to(device))

        return h

    def forward(self, blocks, full_g: dgl.DGLHeteroGraph):
        """
        blocks: list of hetero blocks, one per layer
        returns logits for DST transaction nodes of the last block (seed nodes)
        """
        h = self._input_features_for_block(blocks[0], full_g)

        for l, layer in enumerate(self.layers):
            block = blocks[l]
            h = layer(block, h, ntype_id_map=self.ntype2id, etype_id_map=self.etype2id)
            for ntype in h:
                h[ntype] = self.dropout(F.relu(h[ntype]))

        # On the final block, h contains features for DST nodes (the seeds) of that block
        tx_dst_h = h["transaction"]
        logits = self.cls(tx_dst_h)
        return logits


# -----------------------------
# Graph build (Option 4)
# -----------------------------
def build_graph(zip_path: str, chunksize: int = 200000):
    zf = zipfile.ZipFile(zip_path)

    tx_cols = [
        "TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
        "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2", "dist1", "dist2"
    ]
    id_cols = ["TransactionID", "id_30", "id_31", "id_33", "DeviceType", "DeviceInfo"]

    # identity table fits in memory (~144k rows)
    id_df = zread_csv(zf, "train_identity.csv", usecols=id_cols, chunksize=None)
    id_df = id_df.set_index("TransactionID")
    id_dict = id_df.to_dict(orient="index")

    # mappers
    card_map = IdMapper()
    addr_map = IdMapper()
    device_map = IdMapper()
    browser_map = IdMapper()
    os_map = IdMapper()
    screen_map = IdMapper()
    product_map = IdMapper()

    edges = {
        ("card", "made", "transaction"): ([], []),
        ("address", "at", "transaction"): ([], []),
        ("device", "used", "transaction"): ([], []),
        ("browser", "ua", "transaction"): ([], []),
        ("os", "os", "transaction"): ([], []),
        ("screen", "screen", "transaction"): ([], []),
        ("product", "type", "transaction"): ([], []),
    }

    tx_time, tx_amt, tx_dist1, tx_dist2, tx_labels = [], [], [], [], []
    tx_index = 0

    for chunk in zread_csv(zf, "train_transaction.csv", usecols=tx_cols, chunksize=chunksize):
        for row in chunk.itertuples(index=False):
            tid = int(row[0])
            is_fraud = int(row[1])
            tdt = float(row[2])
            amt = float(row[3])
            prod = safe_str(row[4])

            card_key = "|".join(safe_str(x) for x in row[5:11])
            addr_key = f"{safe_str(row[11])}|{safe_str(row[12])}"

            d1 = 0.0 if pd.isna(row[13]) else float(row[13])
            d2 = 0.0 if pd.isna(row[14]) else float(row[14])

            ident = id_dict.get(tid, None)
            if ident is None:
                device_key = "__NO_IDENTITY__"
                browser_key = "__NO_IDENTITY__"
                os_key = "__NO_IDENTITY__"
                screen_key = "__NO_IDENTITY__"
            else:
                di = safe_str(ident.get("DeviceInfo", "__MISSING__"))
                dt = safe_str(ident.get("DeviceType", "__MISSING__"))
                device_key = di if di != "__MISSING__" else f"DeviceType={dt}"

                browser_key = safe_str(ident.get("id_31", "__MISSING__"))
                os_key = safe_str(ident.get("id_30", "__MISSING__"))
                screen_key = safe_str(ident.get("id_33", "__MISSING__"))

            # map to IDs
            card_id = card_map.get(card_key)
            addr_id = addr_map.get(addr_key)
            dev_id = device_map.get(device_key)
            br_id = browser_map.get(browser_key)
            os_id = os_map.get(os_key)
            sc_id = screen_map.get(screen_key)
            pr_id = product_map.get(prod)

            # add edges entity -> tx
            edges[("card", "made", "transaction")][0].append(card_id)
            edges[("card", "made", "transaction")][1].append(tx_index)

            edges[("address", "at", "transaction")][0].append(addr_id)
            edges[("address", "at", "transaction")][1].append(tx_index)

            edges[("device", "used", "transaction")][0].append(dev_id)
            edges[("device", "used", "transaction")][1].append(tx_index)

            edges[("browser", "ua", "transaction")][0].append(br_id)
            edges[("browser", "ua", "transaction")][1].append(tx_index)

            edges[("os", "os", "transaction")][0].append(os_id)
            edges[("os", "os", "transaction")][1].append(tx_index)

            edges[("screen", "screen", "transaction")][0].append(sc_id)
            edges[("screen", "screen", "transaction")][1].append(tx_index)

            edges[("product", "type", "transaction")][0].append(pr_id)
            edges[("product", "type", "transaction")][1].append(tx_index)

            tx_time.append(tdt)
            tx_amt.append(amt)
            tx_dist1.append(d1)
            tx_dist2.append(d2)
            tx_labels.append(is_fraud)

            tx_index += 1

    num_tx = tx_index

    data_dict = {}
    for etype, (srcs, dsts) in edges.items():
        data_dict[etype] = (torch.tensor(srcs, dtype=torch.int64),
                            torch.tensor(dsts, dtype=torch.int64))

    g = dgl.heterograph(
        data_dict,
        num_nodes_dict={
            "transaction": num_tx,
            "card": len(card_map),
            "address": len(addr_map),
            "device": len(device_map),
            "browser": len(browser_map),
            "os": len(os_map),
            "screen": len(screen_map),
            "product": len(product_map),
        }
    )

    # transaction features (simple starter set)
    tx_time = np.asarray(tx_time, dtype=np.float32)
    tx_amt = np.asarray(tx_amt, dtype=np.float32)
    tx_dist1 = np.asarray(tx_dist1, dtype=np.float32)
    tx_dist2 = np.asarray(tx_dist2, dtype=np.float32)

    def norm(x):
        return (x - x.mean()) / (x.std() + 1e-6)

    feat = np.stack([norm(tx_time), norm(tx_amt), norm(tx_dist1), norm(tx_dist2)], axis=1).astype(np.float32)
    labels = np.asarray(tx_labels, dtype=np.int64)

    g.nodes["transaction"].data["feat"] = torch.tensor(feat, dtype=torch.float32)
    g.nodes["transaction"].data["label"] = torch.tensor(labels, dtype=torch.long)
    g.nodes["transaction"].data["time_raw"] = torch.tensor(tx_time, dtype=torch.float32)

    # 50/50 time split
    order = torch.argsort(g.nodes["transaction"].data["time_raw"])
    split = num_tx // 2
    train_ids = order[:split]
    val_ids = order[split:]

    train_mask = torch.zeros(num_tx, dtype=torch.bool)
    val_mask = torch.zeros(num_tx, dtype=torch.bool)
    train_mask[train_ids] = True
    val_mask[val_ids] = True

    meta = {
        "graph_option": "option4",
        "num_tx": int(num_tx),
        "num_nodes": {ntype: int(g.num_nodes(ntype)) for ntype in g.ntypes},
        "edge_counts": {str(etype): int(g.num_edges(etype)) for etype in g.canonical_etypes},
        "tx_feat_dim": int(g.nodes["transaction"].data["feat"].shape[1]),
        "split": "50/50_time",
    }

    return g, train_mask, val_mask, meta


# -----------------------------
# Dataloaders (neighbor sampling)
# -----------------------------
def make_dataloaders(g, train_mask, val_mask, fanouts, batch_size, num_workers=0):
    # Seed nodes are transaction nodes only
    train_nids = {"transaction": torch.nonzero(train_mask, as_tuple=False).squeeze(1)}
    val_nids = {"transaction": torch.nonzero(val_mask, as_tuple=False).squeeze(1)}

    sampler = MultiLayerNeighborSampler(fanouts)

    train_loader = DataLoader(
        g,
        train_nids,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        device="cpu",  # keep sampling on CPU, move blocks to GPU in training loop
    )

    val_loader = DataLoader(
        g,
        val_nids,
        sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        device="cpu",
    )

    return train_loader, val_loader


# -----------------------------
# Train / Eval with metrics
# -----------------------------
@torch.no_grad()
def evaluate_epoch(model, full_g, loader, device):
    model.eval()

    y_true_all = []
    y_prob_all = []
    y_pred_all = []

    for input_nodes, output_nodes, blocks in loader:
        blocks = [b.to(device) for b in blocks]

        # seed transaction node IDs (original graph IDs)
        seed_tx = output_nodes["transaction"]
        labels = full_g.nodes["transaction"].data["label"][seed_tx].to(device)

        logits = model(blocks, full_g.to(device))
        probs = F.softmax(logits, dim=1)[:, 1]

        y_true = labels.detach().cpu().numpy()
        y_prob = probs.detach().cpu().numpy()
        y_pred = (y_prob >= 0.5).astype(np.int64)

        y_true_all.append(y_true)
        y_prob_all.append(y_prob)
        y_pred_all.append(y_pred)

    y_true_all = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.int64)
    y_prob_all = np.concatenate(y_prob_all) if y_prob_all else np.array([], dtype=np.float64)
    y_pred_all = np.concatenate(y_pred_all) if y_pred_all else np.array([], dtype=np.int64)

    # Robust guards for degenerate splits (rare)
    if len(np.unique(y_true_all)) < 2:
        auc = float("nan")
        ap = float("nan")
    else:
        auc = float(roc_auc_score(y_true_all, y_prob_all))
        ap = float(average_precision_score(y_true_all, y_prob_all))

    f1 = float(f1_score(y_true_all, y_pred_all, zero_division=0))
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1]).tolist()

    return {"ap": ap, "auc": auc, "f1": f1, "cm": cm, "y_true": y_true_all, "y_pred": y_pred_all}


def train(
    g,
    train_mask,
    val_mask,
    fanouts,
    batch_size,
    device="cpu",
    epochs=10,
    hidden_dim=64,
    lr=2e-3,
    comet_exp=None,
):
    # Full graph stays as source of features/labels; blocks carry structure for each batch
    full_g = g  # keep on CPU for sampler; we'll move tensors as needed

    train_loader, val_loader = make_dataloaders(full_g, train_mask, val_mask, fanouts, batch_size)

    tx_feat_dim = full_g.nodes["transaction"].data["feat"].shape[1]
    model = HGTBlockClassifier(full_g, tx_in_dim=tx_feat_dim, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Imbalance-aware class weights from train set
    y_train = full_g.nodes["transaction"].data["label"][train_mask]
    pos = int((y_train == 1).sum().item())
    neg = int((y_train == 0).sum().item())
    w1 = (neg / (pos + 1e-6)) if pos > 0 else 1.0
    class_w = torch.tensor([1.0, float(w1)], dtype=torch.float32, device=device)

    if comet_exp is not None:
        comet_exp.log_parameters({
            "graph_option": "option4",
            "split": "50/50_time",
            "fanouts": str(fanouts),
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "lr": lr,
            "device": device,
            "tx_feat_dim": int(tx_feat_dim),
            "pos_train": pos,
            "neg_train": neg,
            "pos_weight_class1": float(w1),
        })

    best_val_ap = -1.0
    best_state = None
    best_epoch = -1

    # Move full_g feature tensors to device once (for fast indexing)
    # Note: DGL graph structure used by sampler remains on CPU.
    full_g_device = full_g.to(device)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for input_nodes, output_nodes, blocks in train_loader:
            blocks = [b.to(device) for b in blocks]
            seed_tx = output_nodes["transaction"]
            labels = full_g_device.nodes["transaction"].data["label"][seed_tx]

            logits = model(blocks, full_g_device)
            loss = F.cross_entropy(logits, labels, weight=class_w)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.item())
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Evaluate train/val (metrics + confusion matrix)
        train_metrics = evaluate_epoch(model, full_g, train_loader, device)
        val_metrics = evaluate_epoch(model, full_g, val_loader, device)

        # Track best by val AP
        if not np.isnan(val_metrics["ap"]) and val_metrics["ap"] > best_val_ap:
            best_val_ap = val_metrics["ap"]
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Console output
        print(
            f"epoch {ep:03d} | loss={avg_loss:.4f} "
            f"| train AP={train_metrics['ap']:.4f} AUC={train_metrics['auc']:.4f} F1={train_metrics['f1']:.4f} "
            f"| val AP={val_metrics['ap']:.4f} AUC={val_metrics['auc']:.4f} F1={val_metrics['f1']:.4f} "
            f"| val CM={val_metrics['cm']}"
        )

        # Comet logging
        if comet_exp is not None:
            comet_exp.log_metrics({
                "loss": avg_loss,
                "train_ap": train_metrics["ap"],
                "train_auc": train_metrics["auc"],
                "train_f1": train_metrics["f1"],
                "val_ap": val_metrics["ap"],
                "val_auc": val_metrics["auc"],
                "val_f1": val_metrics["f1"],
            }, step=ep)

            # Log confusion matrix (Comet-native if available)
            try:
                comet_exp.log_confusion_matrix(
                    y_true=val_metrics["y_true"].tolist(),
                    y_predicted=val_metrics["y_pred"].tolist(),
                    labels=["legit(0)", "fraud(1)"],
                    step=ep
                )
            except Exception:
                # fallback: log as text
                comet_exp.log_text(json.dumps({"val_confusion_matrix": val_metrics["cm"]}), step=ep)

    # Reload best
    if best_state is not None:
        model.load_state_dict(best_state)

    final_val = evaluate_epoch(model, full_g, val_loader, device)
    print(f"\nBEST val AP={best_val_ap:.4f} at epoch {best_epoch}")
    print(f"FINAL val (reloaded): AP={final_val['ap']:.4f} AUC={final_val['auc']:.4f} F1={final_val['f1']:.4f} CM={final_val['cm']}")

    if comet_exp is not None:
        comet_exp.log_metrics({
            "best_val_ap": float(best_val_ap),
            "best_epoch": int(best_epoch),
            "final_val_ap": float(final_val["ap"]),
            "final_val_auc": float(final_val["auc"]),
            "final_val_f1": float(final_val["f1"]),
        })
        try:
            comet_exp.log_text(json.dumps({"final_val_confusion_matrix": final_val["cm"]}, indent=2))
        except Exception:
            pass


def parse_fanouts(s: str):
    # e.g. "15,10" -> [15,10]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    fanouts = [int(p) for p in parts]
    if len(fanouts) < 1:
        raise ValueError("fanouts must have at least 1 layer")
    return fanouts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-path", required=True, help="Path to ieee_cis_tran_and_identity.zip")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--chunksize", type=int, default=200000)

    # Neighbor sampling
    ap.add_argument("--fanouts", default="15,10", help="Comma-separated fanouts per layer, e.g. '15,10'")
    ap.add_argument("--batch-size", type=int, default=4096)

    ap.add_argument("--dump-meta", default="", help="Optional JSON path to write graph metadata")

    # Comet
    ap.add_argument("--comet", action="store_true", help="Enable Comet logging")
    ap.add_argument("--comet-project", default=os.getenv("COMET_PROJECT_NAME", "fraud-gnn"))
    ap.add_argument("--comet-workspace", default=os.getenv("COMET_WORKSPACE", ""))
    ap.add_argument("--comet-name", default="", help="Experiment name (optional)")
    ap.add_argument("--comet-tags", default="ieee,option4,hgt,neighbor_sampling", help="Comma-separated tags")

    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; using CPU.")
        device = "cpu"

    fanouts = parse_fanouts(args.fanouts)

    tags = [t.strip() for t in args.comet_tags.split(",") if t.strip()]
    comet_exp, comet_reason = try_create_comet_experiment(
        enabled=args.comet,
        project=args.comet_project,
        workspace=args.comet_workspace,
        tags=tags,
        name=args.comet_name
    )
    if args.comet:
        print(f"[Comet] {comet_reason}")

    g, train_mask, val_mask, meta = build_graph(args.zip_path, chunksize=args.chunksize)

    print("Graph built.")
    print("Node counts:", meta["num_nodes"])
    print("Transactions:", meta["num_tx"])
    print("Train tx:", int(train_mask.sum().item()), "Val tx:", int(val_mask.sum().item()))
    print("Fanouts:", fanouts, "Batch size:", args.batch_size)

    if args.dump_meta:
        with open(args.dump_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Metadata written to: {args.dump_meta}")

    if comet_exp is not None:
        comet_exp.log_parameters({
            "zip_path": os.path.basename(args.zip_path),
            "chunksize": args.chunksize,
        })
        comet_exp.log_text(json.dumps(meta, indent=2), metadata={"type": "graph_meta"})

    train(
        g=g,
        train_mask=train_mask,
        val_mask=val_mask,
        fanouts=fanouts,
        batch_size=args.batch_size,
        device=device,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        comet_exp=comet_exp
    )

    if comet_exp is not None:
        try:
            comet_exp.end()
        except Exception:
            pass


if __name__ == "__main__":
    main()
