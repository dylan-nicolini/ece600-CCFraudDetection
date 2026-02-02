#!/usr/bin/env python3
"""
sffsd_option4_manual_heterosage_comet_sampling.py

Run S-FFSD through the SAME style pipeline as IEEE Option 4:
- Build a heterogeneous graph where entities point to transactions
- Train a manual hetero GraphSAGE over sampled DGL blocks
- Neighbor sampling (IN-neighbors) for directed entity->transaction edges
- Time-based split (default 70/30) to avoid leakage
- Logs AP, AUC, F1 and confusion matrix to console + (optionally) Comet

S-FFSD expected columns (case-insensitive supported):
  time, source, target, amount, location, type, labels

Notes:
- We rename the node type for "type" to "ttype" internally to avoid
  collisions with older PyTorch/DGL ModuleDict attributes.
- We also avoid using an edge relation name "type" for the same reason.
"""

import argparse
import json
import os

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
# Force IN-neighbor sampling
# -----------------------------
class InNeighborSampler(MultiLayerNeighborSampler):
    """
    Forces IN-neighbor sampling.

    Our graph edges are entity -> transaction.
    If the sampler used out-neighbors from transaction seeds, blocks would be empty.
    """
    def sample_frontier(self, block_id, g, seed_nodes):
        fanout = self.fanouts[block_id]
        return dgl.sampling.sample_neighbors(g, seed_nodes, fanout, edge_dir="in")


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


def parse_fanouts(s: str):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    fanouts = [int(p) for p in parts]
    if len(fanouts) < 1:
        raise ValueError("fanouts must contain at least one integer, e.g. '15,10'")
    return fanouts


def resolve_cols_case_insensitive(df_cols, required_map):
    """
    required_map: dict {canonical_name: [aliases...]}
    returns dict {canonical_name: actual_df_col_name}
    """
    lower = {c.lower(): c for c in df_cols}
    resolved = {}
    for canon, aliases in required_map.items():
        found = None
        for a in aliases:
            if a.lower() in lower:
                found = lower[a.lower()]
                break
        if found is None:
            raise KeyError(f"Missing required column for '{canon}'. Tried aliases: {aliases}. Available: {list(df_cols)[:50]} ...")
        resolved[canon] = found
    return resolved


# -----------------------------
# Model: Manual hetero SAGE over blocks
# -----------------------------
class HeteroSAGEBlockClassifier(nn.Module):
    """
    Manual hetero message passing over blocks (robust across older DGL).

    For each layer:
      For each canonical etype (src, rel, dst) present in the sampled block:
        out[dst] += SAGEConv(block[(src,rel,dst)], (h_src, h_dst))
      h := relu(dropout(out))

    Returns logits for DST transaction nodes of final block (seed nodes).
    """
    def __init__(self, full_g: dgl.DGLHeteroGraph, tx_in_dim: int, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.ntypes = full_g.ntypes
        self.canonical_etypes = list(full_g.canonical_etypes)

        self.tx_proj = nn.Linear(tx_in_dim, hidden_dim)

        self.emb = nn.ModuleDict()
        for ntype in self.ntypes:
            if ntype == "transaction":
                continue
            self.emb[ntype] = nn.Embedding(full_g.num_nodes(ntype), hidden_dim)

        # Per-layer, per-relation conv modules keyed by safe string "src__rel__dst"
        self.rel_convs = nn.ModuleList()
        for _ in range(num_layers):
            md = nn.ModuleDict()
            for (s, r, d) in self.canonical_etypes:
                key = f"{s}__{r}__{d}"
                md[key] = dgl.nn.SAGEConv(hidden_dim, hidden_dim, "mean")
            self.rel_convs.append(md)

        self.cls = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def _input_features_for_block(self, block: dgl.DGLHeteroGraph, full_g: dgl.DGLHeteroGraph) -> dict:
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
        h = self._input_features_for_block(blocks[0], full_g)

        for l in range(self.num_layers):
            block = blocks[l]
            out = {}

            for (s, r, d) in block.canonical_etypes:
                if block.num_edges((s, r, d)) == 0:
                    continue

                key = f"{s}__{r}__{d}"
                conv = self.rel_convs[l][key]

                dst_count = block.num_dst_nodes(d)

                if s not in h or d not in h:
                    continue

                h_src = h[s]
                h_dst = h[d][:dst_count]

                rel_g = block[(s, r, d)]
                msg = conv(rel_g, (h_src, h_dst))
                out[d] = msg if d not in out else (out[d] + msg)

            # Carry forward dst types that got no messages
            for d in block.dsttypes:
                if d not in out and d in h:
                    dst_count = block.num_dst_nodes(d)
                    out[d] = h[d][:dst_count]

            for ntype in out:
                out[ntype] = self.dropout(F.relu(out[ntype]))

            h = out

        # Fallback if something odd happens
        if "transaction" not in h:
            last_block = blocks[-1]
            dst_tx_nids = last_block.dstnodes["transaction"].data[dgl.NID]
            tx_feat = full_g.nodes["transaction"].data["feat"][dst_tx_nids].to(last_block.device)
            h["transaction"] = self.tx_proj(tx_feat)

        return self.cls(h["transaction"])


# -----------------------------
# Graph build: S-FFSD -> heterograph
# -----------------------------
def build_graph_sffsd(csv_path: str, train_ratio: float = 0.7):
    df = pd.read_csv(csv_path)

    # Resolve columns case-insensitively
    colmap = resolve_cols_case_insensitive(
        df.columns,
        {
            "time": ["time", "Time"],
            "source": ["source", "Source"],
            "target": ["target", "Target"],
            "amount": ["amount", "Amount", "TransactionAmt"],
            "location": ["location", "Location"],
            "type": ["type", "Type"],
            "labels": ["labels", "Labels", "isFraud"],
        },
    )

    # Mappers
    src_map = IdMapper()
    tgt_map = IdMapper()
    loc_map = IdMapper()
    ttype_map = IdMapper()  # rename "type" node type -> "ttype"

    # Edge lists entity -> transaction
    edges = {
        ("source", "src_of", "transaction"): ([], []),
        ("target", "tgt_of", "transaction"): ([], []),
        ("location", "loc_of", "transaction"): ([], []),
        ("ttype", "kind_of", "transaction"): ([], []),  # avoid rel name "type"
    }

    # Tx arrays
    tx_time = df[colmap["time"]].to_numpy(dtype=np.float32)
    tx_amt = df[colmap["amount"]].to_numpy(dtype=np.float32)
    tx_lbl = df[colmap["labels"]].to_numpy(dtype=np.int64)

    # Build entities + edges
    for tx_id, (s, t, loc, ty) in enumerate(
        zip(
            df[colmap["source"]].map(safe_str),
            df[colmap["target"]].map(safe_str),
            df[colmap["location"]].map(safe_str),
            df[colmap["type"]].map(safe_str),
        )
    ):
        s_id = src_map.get(s)
        t_id = tgt_map.get(t)
        l_id = loc_map.get(loc)
        ty_id = ttype_map.get(ty)

        edges[("source", "src_of", "transaction")][0].append(s_id)
        edges[("source", "src_of", "transaction")][1].append(tx_id)

        edges[("target", "tgt_of", "transaction")][0].append(t_id)
        edges[("target", "tgt_of", "transaction")][1].append(tx_id)

        edges[("location", "loc_of", "transaction")][0].append(l_id)
        edges[("location", "loc_of", "transaction")][1].append(tx_id)

        edges[("ttype", "kind_of", "transaction")][0].append(ty_id)
        edges[("ttype", "kind_of", "transaction")][1].append(tx_id)

    num_tx = df.shape[0]

    data_dict = {}
    for etype, (srcs, dsts) in edges.items():
        data_dict[etype] = (torch.tensor(srcs, dtype=torch.int64), torch.tensor(dsts, dtype=torch.int64))

    g = dgl.heterograph(
        data_dict,
        num_nodes_dict={
            "transaction": int(num_tx),
            "source": len(src_map),
            "target": len(tgt_map),
            "location": len(loc_map),
            "ttype": len(ttype_map),
        },
    )

    # Normalize features
    def norm(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / (x.std() + 1e-6)

    feat = np.stack([norm(tx_time), norm(tx_amt)], axis=1).astype(np.float32)

    g.nodes["transaction"].data["feat"] = torch.tensor(feat, dtype=torch.float32)
    g.nodes["transaction"].data["label"] = torch.tensor(tx_lbl, dtype=torch.long)
    g.nodes["transaction"].data["time_raw"] = torch.tensor(tx_time, dtype=torch.float32)

    # Time-based split
    order = torch.argsort(g.nodes["transaction"].data["time_raw"])
    split = int(num_tx * train_ratio)
    train_ids = order[:split]
    val_ids = order[split:]

    train_mask = torch.zeros(num_tx, dtype=torch.bool)
    val_mask = torch.zeros(num_tx, dtype=torch.bool)
    train_mask[train_ids] = True
    val_mask[val_ids] = True

    meta = {
        "dataset": "S-FFSD",
        "graph_style": "option4_like",
        "model": "manual_hetero_sage",
        "split": f"{int(train_ratio*100)}/{int((1-train_ratio)*100)}_time",
        "num_tx": int(num_tx),
        "num_nodes": {ntype: int(g.num_nodes(ntype)) for ntype in g.ntypes},
        "edge_counts": {str(etype): int(g.num_edges(etype)) for etype in g.canonical_etypes},
        "tx_feat_dim": int(g.nodes["transaction"].data["feat"].shape[1]),
        "csv_path": os.path.basename(csv_path),
    }

    return g, train_mask, val_mask, meta


# -----------------------------
# DataLoaders + Eval + Train
# -----------------------------
def make_dataloaders(g_cpu, train_mask, val_mask, fanouts, batch_size, num_workers=0):
    train_nids = {"transaction": torch.nonzero(train_mask, as_tuple=False).squeeze(1)}
    val_nids = {"transaction": torch.nonzero(val_mask, as_tuple=False).squeeze(1)}

    sampler = InNeighborSampler(fanouts)

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
    y_true_all, y_prob_all, y_pred_all = [], [], []

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
    model = HeteroSAGEBlockClassifier(
        g_cpu, tx_in_dim=tx_feat_dim, hidden_dim=hidden_dim, num_layers=len(fanouts)
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Class weights (helps imbalance)
    y_train = g_cpu.nodes["transaction"].data["label"][train_mask]
    pos = int((y_train == 1).sum().item())
    neg = int((y_train == 0).sum().item())
    w1 = (neg / (pos + 1e-6)) if pos > 0 else 1.0
    class_w = torch.tensor([1.0, float(w1)], dtype=torch.float32, device=device)

    g_device = g_cpu.to(device)

    if comet_exp is not None:
        comet_exp.log_parameters({
            "dataset": "S-FFSD",
            "model": "manual_hetero_sage",
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


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", required=True, help="Path to S-FFSD.csv")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--train-ratio", type=float, default=0.7, help="Time-based train ratio, e.g. 0.7 for 70/30")

    ap.add_argument("--fanouts", default="15,10", help="Comma-separated fanouts per layer, e.g. '15,10'")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--num-workers", type=int, default=0)

    ap.add_argument("--dump-meta", default="", help="Optional JSON path to write graph metadata")

    # Comet
    ap.add_argument("--comet", action="store_true", help="Enable Comet logging")
    ap.add_argument("--comet-project", default=os.getenv("COMET_PROJECT_NAME", "fraud-gnn"))
    ap.add_argument("--comet-workspace", default=os.getenv("COMET_WORKSPACE", ""))
    ap.add_argument("--comet-name", default="", help="Experiment name (optional)")
    ap.add_argument("--comet-tags", default="s-ffsd,option4_like,manual_heterosage,neighbor_sampling", help="Comma-separated tags")

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
        print(f"[Comet] project={args.comet_project} workspace={args.comet_workspace}")

    if not (0.1 <= args.train_ratio <= 0.95):
        raise ValueError("--train-ratio must be between 0.1 and 0.95")

    g, train_mask, val_mask, meta = build_graph_sffsd(args.csv_path, train_ratio=args.train_ratio)

    print("Graph built.")
    print("Node counts:", meta["num_nodes"])
    print("Transactions:", meta["num_tx"])
    print("Train tx:", int(train_mask.sum().item()), "Val tx:", int(val_mask.sum().item()))
    print("Fanouts:", fanouts, "Batch size:", args.batch_size)

    # Batches per epoch for sanity
    train_loader, val_loader = make_dataloaders(g, train_mask, val_mask, fanouts, args.batch_size, num_workers=args.num_workers)
    print("Train batches per epoch:", len(train_loader))
    print("Val batches:", len(val_loader))

    if args.dump_meta:
        with open(args.dump_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Metadata written to: {args.dump_meta}")

    if comet_exp is not None:
        comet_exp.log_parameters({
            "csv_path": os.path.basename(args.csv_path),
            "train_ratio": float(args.train_ratio),
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
