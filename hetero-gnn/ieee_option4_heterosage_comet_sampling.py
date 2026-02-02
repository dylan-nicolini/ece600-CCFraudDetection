#!/usr/bin/env python3
"""
ieee_option4_heterosage_comet_sampling.py

IEEE-CIS Option 4 heterogeneous graph + neighbor sampling + Comet logging.

Why this version?
- Your DGL build's HGTConv requires (ntype, etype) tensors (older API),
  which is awkward for true heterographs + blocks.
- This implementation uses heterograph-native message passing:
    dgl.nn.HeteroGraphConv with per-relation SAGEConv modules
  which is stable across DGL versions and works with neighbor sampling.

ZIP input must contain:
  - train_transaction.csv
  - train_identity.csv

Graph Option 4:
  Node types:
    - transaction (labeled)
    - card       (card1..card6 combined)
    - address    (addr1|addr2)
    - device     (DeviceInfo preferred; fallback DeviceType)
    - browser    (id_31)
    - os         (id_30)
    - screen     (id_33)
    - product    (ProductCD)

Edge types (entity -> transaction):
  ("card",    "made",   "transaction")
  ("address", "at",     "transaction")
  ("device",  "used",   "transaction")
  ("browser", "ua",     "transaction")
  ("os",      "os",     "transaction")
  ("screen",  "screen", "transaction")
  ("product", "type",   "transaction")

Split:
  50/50 time-based split on TransactionDT (sorted ascending):
    first half -> train, second half -> val

Training:
  - Neighbor sampling via MultiLayerNeighborSampler
  - Seed nodes are transaction nodes
  - Predict labels for seed transaction nodes
  - Log AP, AUC, F1 + confusion matrix; Comet optional

Chunked ZIP reading FIX:
  - zread_csv() keeps the ZIP member stream open for iterator lifetime.
"""

import argparse
import io
import json
import os
import zipfile

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
    """
    Safe ZIP CSV reader.

    - chunksize=None: returns a DataFrame and closes stream
    - chunksize set: returns an iterator that yields chunks and closes stream at end

    Prevents: ValueError: I/O operation on closed file
    """
    zstream = zf.open(name)  # keep open for iterator life
    text = io.TextIOWrapper(zstream, encoding="utf-8", newline="")

    if chunksize is None:
        try:
            return pd.read_csv(text, usecols=usecols)
        finally:
            text.close()

    def _iter():
        try:
            for chunk in pd.read_csv(text, usecols=usecols, chunksize=chunksize):
                yield chunk
        finally:
            text.close()

    return _iter()


# -----------------------------
# Model: HeteroGraphConv + SAGEConv (block-friendly)
# -----------------------------
class HeteroSAGEBlockClassifier(nn.Module):
    """
    Neighbor-sampling friendly hetero GNN:

    - transaction: numeric features -> linear proj -> hidden
    - entities: ID embedding -> hidden
    - message passing: HeteroGraphConv with per-relation SAGEConv
    - classifier: MLP on transaction embeddings at final (seed) layer
    """
    def __init__(self, full_g: dgl.DGLHeteroGraph, tx_in_dim: int, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.ntypes = full_g.ntypes
        # NOTE: older DGL expects string keys in HeteroGraphConv mods dict
        self.etypes = full_g.etypes  # e.g. ["made","at","used","ua","os","screen","type"]


        # Transaction feature projection
        self.tx_proj = nn.Linear(tx_in_dim, hidden_dim)

        # Entity embeddings
        self.emb = nn.ModuleDict()
        for ntype in self.ntypes:
            if ntype == "transaction":
                continue
            self.emb[ntype] = nn.Embedding(full_g.num_nodes(ntype), hidden_dim)

        # Per-layer hetero conv
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            rel_convs = {}
            for etype in self.etypes:
                rel_convs[etype] = dgl.nn.SAGEConv(
                    in_feats=hidden_dim,
                    out_feats=hidden_dim,
                    aggregator_type="mean"
                )
            self.layers.append(dgl.nn.HeteroGraphConv(rel_convs, aggregate="sum"))

        self.cls = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def _input_features_for_block(self, block: dgl.DGLHeteroGraph, full_g: dgl.DGLHeteroGraph) -> dict:
        """
        Build input features for SRC nodes of the first block.

        In blocks: block.srcnodes[ntype].data[dgl.NID] gives original IDs in full graph.
        """
        h = {}
        device = block.device

        for ntype in block.srctypes:
            src_nids = block.srcnodes[ntype].data[dgl.NID]
            if ntype == "transaction":
                tx_feat = full_g.nodes["transaction"].data["feat"][src_nids].to(device)
                h["transaction"] = self.tx_proj(tx_feat)
            else:
                h[ntype] = self.emb[ntype](src_nids.to(device))
        return h

    def forward(self, blocks, full_g: dgl.DGLHeteroGraph) -> torch.Tensor:
        """
        blocks: list of sampled blocks (len == num_layers)
        returns logits for DST transaction nodes of last block (seed nodes)
        """
        h = self._input_features_for_block(blocks[0], full_g)

        for l in range(self.num_layers):
            block = blocks[l]
            h = self.layers[l](block, h)  # returns features for DST nodes of this block
            for ntype in h:
                h[ntype] = self.dropout(F.relu(h[ntype]))

        # After final layer, h["transaction"] corresponds to DST transaction nodes (seeds)
        return self.cls(h["transaction"])


# -----------------------------
# Build graph (Option 4)
# -----------------------------
def build_graph(zip_path: str, chunksize: int = 200000):
    zf = zipfile.ZipFile(zip_path)

    tx_cols = [
        "TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
        "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2", "dist1", "dist2"
    ]
    id_cols = ["TransactionID", "id_30", "id_31", "id_33", "DeviceType", "DeviceInfo"]

    # Load identity into dict (fits in memory)
    id_df = zread_csv(zf, "train_identity.csv", usecols=id_cols, chunksize=None)
    id_df = id_df.set_index("TransactionID")
    id_dict = id_df.to_dict(orient="index")

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

            card_id = card_map.get(card_key)
            addr_id = addr_map.get(addr_key)
            dev_id = device_map.get(device_key)
            br_id = browser_map.get(browser_key)
            os_id = os_map.get(os_key)
            sc_id = screen_map.get(screen_key)
            pr_id = product_map.get(prod)

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

    # Tx features (starter)
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
        "model": "hetero_sage",
        "split": "50/50_time",
        "num_tx": int(num_tx),
        "num_nodes": {ntype: int(g.num_nodes(ntype)) for ntype in g.ntypes},
        "edge_counts": {str(etype): int(g.num_edges(etype)) for etype in g.canonical_etypes},
        "tx_feat_dim": int(g.nodes["transaction"].data["feat"].shape[1]),
        "chunksize": int(chunksize),
        "zip_path": os.path.basename(zip_path),
    }

    return g, train_mask, val_mask, meta


# -----------------------------
# Neighbor sampling loaders
# -----------------------------
def make_dataloaders(g_cpu, train_mask, val_mask, fanouts, batch_size, num_workers=0):
    train_nids = {"transaction": torch.nonzero(train_mask, as_tuple=False).squeeze(1)}
    val_nids = {"transaction": torch.nonzero(val_mask, as_tuple=False).squeeze(1)}

    sampler = MultiLayerNeighborSampler(fanouts)

    train_loader = DataLoader(
        g_cpu,
        train_nids,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        device="cpu",
    )

    val_loader = DataLoader(
        g_cpu,
        val_nids,
        sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        device="cpu",
    )

    return train_loader, val_loader


@torch.no_grad()
def evaluate_epoch(model, g_device, loader, device):
    model.eval()

    y_true_all = []
    y_prob_all = []
    y_pred_all = []

    for _, output_nodes, blocks in loader:
        blocks = [b.to(device) for b in blocks]
        seed_tx = output_nodes["transaction"]

        labels = g_device.nodes["transaction"].data["label"][seed_tx]
        logits = model(blocks, g_device)
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
    g_cpu,
    train_mask,
    val_mask,
    fanouts,
    batch_size,
    device="cpu",
    epochs=10,
    hidden_dim=64,
    lr=2e-3,
    num_workers=0,
    comet_exp=None,
):
    train_loader, val_loader = make_dataloaders(g_cpu, train_mask, val_mask, fanouts, batch_size, num_workers=num_workers)

    tx_feat_dim = int(g_cpu.nodes["transaction"].data["feat"].shape[1])
    model = HeteroSAGEBlockClassifier(g_cpu, tx_in_dim=tx_feat_dim, hidden_dim=hidden_dim, num_layers=len(fanouts)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # class weights
    y_train = g_cpu.nodes["transaction"].data["label"][train_mask]
    pos = int((y_train == 1).sum().item())
    neg = int((y_train == 0).sum().item())
    w1 = (neg / (pos + 1e-6)) if pos > 0 else 1.0
    class_w = torch.tensor([1.0, float(w1)], dtype=torch.float32, device=device)

    # Move full graph tensors to device once
    g_device = g_cpu.to(device)

    if comet_exp is not None:
        comet_exp.log_parameters({
            "graph_option": "option4",
            "model": "hetero_sage",
            "split": "50/50_time",
            "fanouts": str(list(fanouts)),
            "batch_size": int(batch_size),
            "hidden_dim": int(hidden_dim),
            "epochs": int(epochs),
            "lr": float(lr),
            "device": device,
            "tx_feat_dim": int(tx_feat_dim),
            "pos_train": int(pos),
            "neg_train": int(neg),
            "pos_weight_class1": float(w1),
            "num_workers": int(num_workers),
        })

    best_val_ap = -1.0
    best_state = None
    best_epoch = -1

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for _, output_nodes, blocks in train_loader:
            blocks = [b.to(device) for b in blocks]
            seed_tx = output_nodes["transaction"]
            labels = g_device.nodes["transaction"].data["label"][seed_tx]

            logits = model(blocks, g_device)
            loss = F.cross_entropy(logits, labels, weight=class_w)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.item())
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        train_metrics = evaluate_epoch(model, g_device, train_loader, device)
        val_metrics = evaluate_epoch(model, g_device, val_loader, device)

        if not np.isnan(val_metrics["ap"]) and val_metrics["ap"] > best_val_ap:
            best_val_ap = val_metrics["ap"]
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"epoch {ep:03d} | loss={avg_loss:.4f} "
            f"| train AP={train_metrics['ap']:.4f} AUC={train_metrics['auc']:.4f} F1={train_metrics['f1']:.4f} "
            f"| val AP={val_metrics['ap']:.4f} AUC={val_metrics['auc']:.4f} F1={val_metrics['f1']:.4f} "
            f"| val CM={val_metrics['cm']}"
        )

        if comet_exp is not None:
            comet_exp.log_metrics({
                "loss": avg_loss,
                "train_ap": float(train_metrics["ap"]),
                "train_auc": float(train_metrics["auc"]),
                "train_f1": float(train_metrics["f1"]),
                "val_ap": float(val_metrics["ap"]),
                "val_auc": float(val_metrics["auc"]),
                "val_f1": float(val_metrics["f1"]),
            }, step=ep)

            # Confusion matrix
            try:
                comet_exp.log_confusion_matrix(
                    y_true=val_metrics["y_true"].tolist(),
                    y_predicted=val_metrics["y_pred"].tolist(),
                    labels=["legit(0)", "fraud(1)"],
                    step=ep
                )
            except Exception:
                comet_exp.log_text(json.dumps({"val_confusion_matrix": val_metrics["cm"]}), step=ep)

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val = evaluate_epoch(model, g_device, val_loader, device)
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
    parts = [p.strip() for p in s.split(",") if p.strip()]
    fanouts = [int(p) for p in parts]
    if len(fanouts) < 1:
        raise ValueError("fanouts must contain at least one integer, e.g. '15,10'")
    return fanouts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-path", required=True, help="Path to ieee_cis_tran_and_identity.zip")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--chunksize", type=int, default=200000)

    ap.add_argument("--fanouts", default="15,10", help="Comma-separated fanouts per layer, e.g. '15,10'")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--num-workers", type=int, default=0)

    ap.add_argument("--dump-meta", default="", help="Optional JSON path to write graph metadata")

    # Comet
    ap.add_argument("--comet", action="store_true", help="Enable Comet logging")
    ap.add_argument("--comet-project", default=os.getenv("COMET_PROJECT_NAME", "fraud-gnn"))
    ap.add_argument("--comet-workspace", default=os.getenv("COMET_WORKSPACE", ""))
    ap.add_argument("--comet-name", default="", help="Experiment name (optional)")
    ap.add_argument("--comet-tags", default="ieee,option4,heterosage,neighbor_sampling", help="Comma-separated tags")

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
            "chunksize": int(args.chunksize),
        })
        try:
            comet_exp.log_text(json.dumps(meta, indent=2), metadata={"type": "graph_meta"})
        except Exception:
            pass

    train(
        g_cpu=g,
        train_mask=train_mask,
        val_mask=val_mask,
        fanouts=fanouts,
        batch_size=args.batch_size,
        device=device,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        num_workers=args.num_workers,
        comet_exp=comet_exp
    )

    if comet_exp is not None:
        try:
            comet_exp.end()
        except Exception:
            pass


if __name__ == "__main__":
    main()
