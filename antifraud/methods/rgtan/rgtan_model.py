import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dgl
from dgl.utils import expand_as_pair
from dgl import function as fn
from dgl.base import DGLError
from dgl.nn.functional import edge_softmax


# -----------------------------
# Positional encoding (kept for compatibility)
# -----------------------------
class PosEncoding(nn.Module):
    def __init__(self, dim, device, base=10000, bias=0):
        super().__init__()
        self.dim = dim
        self.device = device
        self.base = base
        self.bias = bias

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        # pos: (B,)
        pos = pos + self.bias
        div = torch.exp(
            torch.arange(0, self.dim, 2, device=self.device, dtype=torch.float32)
            * (-math.log(self.base) / self.dim)
        )
        pe = torch.zeros((pos.shape[0], self.dim), device=self.device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos.unsqueeze(1) * div)
        pe[:, 1::2] = torch.cos(pos.unsqueeze(1) * div)
        return pe


# -----------------------------
# TransformerConv (block-safe)
# -----------------------------
class TransformerConv(nn.Module):
    """
    Attention conv that is DGLBlock-safe.

    KEY FIX:
      If graph is a DGLBlock, src and dst node sets differ.
      We must use:
        h_src for src nodes (num_src)
        h_dst for dst nodes (num_dst), where dst nodes are the first num_dst nodes in src.
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=True,
        bias=True,
    ):
        super().__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._num_heads = num_heads
        self._allow_zero_in_degree = allow_zero_in_degree

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.residual = residual
        self.activation = activation

        self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=bias)
        self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=bias)
        self.fc_value = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=bias)

        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))

        if residual and (self._in_dst_feats != out_feats * num_heads):
            self.res_fc = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=bias)
        else:
            self.res_fc = None

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_value.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.res_fc is not None:
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph. "
                        "Add self-loops or set allow_zero_in_degree=True."
                    )

            # ---- Block-safe src/dst handling ----
            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
            else:
                h_src = self.feat_drop(feat)
                if getattr(graph, "is_block", False):
                    h_dst = h_src[: graph.num_dst_nodes()]
                else:
                    h_dst = h_src

            feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            feat_v = self.fc_value(h_src).view(-1, self._num_heads, self._out_feats)

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)  # (Ns, H, 1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)  # (Nd, H, 1)

            graph.srcdata.update({"ft": feat_src, "el": el, "fv": feat_v})
            graph.dstdata.update({"er": er})

            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))
            graph.edata["a"] = edge_softmax(graph, e)
            graph.edata["a"] = self.attn_drop(graph.edata["a"])

            graph.update_all(fn.u_mul_e("fv", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]  # (Nd, H, F)

            # residual
            if self.residual:
                if self.res_fc is not None:
                    resval = self.res_fc(h_dst).view(-1, self._num_heads, self._out_feats)
                    rst = rst + resval
                else:
                    # only safe if in-dst == out_feats*num_heads
                    rst = rst + h_dst.view(-1, self._num_heads, self._out_feats)

            rst = rst.flatten(1)  # (Nd, H*F)

            if self.activation is not None:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata["a"]
            return rst


# -----------------------------
# Neighbor-stat encoder (IEEE-safe)
# -----------------------------
class Tabular1DCNN2(nn.Module):
    """
    Encodes neighbor-stat features.

    IEEE FIX:
      If input_dim <= 0, disable and return an empty embedding.
    """

    def __init__(self, input_dim: int, embed_dim: int, dropout: float = 0.2):
        super().__init__()
        self.input_dim = int(input_dim) if input_dim is not None else 0
        self.embed_dim = int(embed_dim)
        self.disabled = self.input_dim <= 0

        if self.disabled:
            return

        # Simple projection: (B, input_dim) -> (B, input_dim, embed_dim)
        self.bn = nn.BatchNorm1d(self.input_dim)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(self.input_dim, self.input_dim * self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.disabled:
            if x is None:
                return torch.zeros((1, 0, self.embed_dim), device="cpu")
            return x.new_zeros((x.shape[0], 0, self.embed_dim))

        x = self.drop(self.bn(x))
        x = torch.relu(self.proj(x))
        x = x.view(x.shape[0], self.input_dim, self.embed_dim)
        return x


# -----------------------------
# Categorical + optional neighbor-stat embedding
# -----------------------------
class TransEmbedding(nn.Module):
    """
    Embeds categorical columns + optional neighbor-stat dict.

    Fixes:
      - forward_mlp always exists and matches cat_cols (no NoneType crash)
      - neighbor-stat branch safely skipped when empty for IEEE
    """

    def __init__(
        self,
        df: pd.DataFrame,
        device: str,
        dropout: float,
        in_feats_dim: int,
        cat_features,
        neigh_features,
        att_head_num: int = 4,
    ):
        super().__init__()
        self.device = device
        self.in_feats_dim = in_feats_dim
        self.dropout = nn.Dropout(dropout)

        if isinstance(cat_features, dict):
            cat_cols = list(cat_features.keys())
        else:
            cat_cols = list(cat_features) if cat_features is not None else []

        self.cat_cols = [c for c in cat_cols if c not in {"Labels", "Time"}]

        # Embedding tables for categorical cols
        self.cat_table = nn.ModuleDict()
        for col in self.cat_cols:
            max_id = int(df[col].max()) if (df is not None and col in df.columns) else 0
            self.cat_table[col] = nn.Embedding(max_id + 1, in_feats_dim).to(device)

        # Per-cat MLP layers sized to cat_cols (never None)
        self.forward_mlp = nn.ModuleList([nn.Linear(in_feats_dim, in_feats_dim) for _ in range(len(self.cat_cols))])

        # Neighbor-stat encoder only if dict non-empty
        self.nei_table = None
        if isinstance(neigh_features, dict) and len(neigh_features) > 0:
            self.nei_table = Tabular1DCNN2(input_dim=len(neigh_features), embed_dim=in_feats_dim)

        # Attention to combine neighbor-stat tokens
        self.att_head_num = max(1, int(att_head_num))
        self.att_head_size = max(1, int(in_feats_dim / self.att_head_num))
        self.total_head_size = in_feats_dim

        self.lin_q = nn.Linear(in_feats_dim, self.total_head_size)
        self.lin_k = nn.Linear(in_feats_dim, self.total_head_size)
        self.lin_v = nn.Linear(in_feats_dim, self.total_head_size)
        self.lin_final = nn.Linear(in_feats_dim, in_feats_dim)
        self.layer_norm = nn.LayerNorm(in_feats_dim, eps=1e-8)

        self.neigh_mlp = nn.Linear(in_feats_dim, 1)

    def forward_emb(self, cat_feat: dict) -> dict:
        support = {}
        for col in self.cat_cols:
            support[col] = self.cat_table[col](cat_feat[col])
        return support

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        H = self.att_head_num
        Hd = int(D / H) if (D % H == 0) else self.att_head_size
        x = x.view(B, T, H, Hd)
        return x.permute(0, 2, 1, 3)  # (B, H, T, Hd)

    def forward_neigh_emb(self, neighstat_feat: dict):
        if (neighstat_feat is None) or (not isinstance(neighstat_feat, dict)) or (len(neighstat_feat) == 0) or (self.nei_table is None):
            return None

        cols = list(neighstat_feat.keys())
        vals = [neighstat_feat[c] for c in cols]  # each (B,)
        x = torch.stack(vals, dim=1)  # (B, C)

        tokens = self.nei_table(x)  # (B, C, D)

        q = self.lin_q(tokens)
        k = self.lin_k(tokens)
        v = self.lin_v(tokens)

        qh = self._transpose_for_scores(q)
        kh = self._transpose_for_scores(k)
        vh = self._transpose_for_scores(v)

        att = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(qh.shape[-1])
        att = torch.softmax(att, dim=-1)

        ctx = torch.matmul(att, vh)  # (B, H, T, Hd)
        ctx = ctx.permute(0, 2, 1, 3).contiguous()  # (B, T, H, Hd)
        ctx = ctx.view(ctx.shape[0], ctx.shape[1], -1)  # (B, T, D)

        out = self.lin_final(ctx)
        out = self.layer_norm(out)

        # Aggregate over tokens -> (B, D)
        out = out.mean(dim=1)
        return out

    def forward(self, cat_feat: dict, neighstat_feat: dict):
        support = self.forward_emb(cat_feat)

        cat_output = 0
        for i, col in enumerate(self.cat_cols):
            x = self.dropout(support[col])
            x = self.forward_mlp[i](x)
            cat_output = cat_output + x

        nei_output = 0
        nei_vec = self.forward_neigh_emb(neighstat_feat)
        if nei_vec is not None:
            nei_output = self.neigh_mlp(nei_vec).squeeze(-1)

        return cat_output, nei_output


# -----------------------------
# RGTAN model
# -----------------------------
class RGTAN(nn.Module):
    def __init__(
        self,
        in_feats,
        hidden_dim,
        n_classes,
        heads,
        activation,
        n_layers,
        drop,
        device,
        gated,
        ref_df,
        cat_features,
        neigh_features,
        nei_att_head,
        **kwargs
    ):
        super().__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.heads = heads
        self.activation = activation
        self.n_layers = n_layers
        self.drop = drop
        self.device = device
        self.gated = gated

        self.layers = nn.ModuleList()

        feat_drop = drop[0] if isinstance(drop, list) and len(drop) > 0 else float(drop)
        attn_drop = drop[1] if isinstance(drop, list) and len(drop) > 1 else feat_drop

        # first layer
        self.layers.append(
            TransformerConv(
                in_feats=in_feats,
                out_feats=hidden_dim,
                num_heads=heads[0],
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                residual=False,
                activation=activation,
                allow_zero_in_degree=True,
            )
        )

        # middle layers
        for l in range(1, n_layers):
            self.layers.append(
                TransformerConv(
                    in_feats=hidden_dim * heads[l - 1],
                    out_feats=hidden_dim,
                    num_heads=heads[l],
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    residual=True,
                    activation=activation,
                    allow_zero_in_degree=True,
                )
            )

        self.dropout = nn.Dropout(feat_drop)
        self.classifier = nn.Linear(hidden_dim * heads[-1], n_classes)

        self.trans_emb = TransEmbedding(
            df=ref_df,
            device=device,
            dropout=feat_drop,
            in_feats_dim=in_feats,
            cat_features=cat_features,
            neigh_features=neigh_features,
            att_head_num=nei_att_head,
        )

        # label embedding (labels expected: {0,1} plus padding idx 2)
        self.label_emb = nn.Embedding(3, hidden_dim * heads[-1]).to(device)

    def forward(self, blocks, x, y, x1, neigh_input=None):
        # Add categorical embedding to numeric features for src nodes
        if isinstance(x1, dict):
            x_cat, _ = self.trans_emb(x1, neigh_input)
            x = x + x_cat

        h = x

        # Each block maps src -> dst; output h is always sized to dst nodes of that block
        for l in range(self.n_layers):
            h = self.layers[l](blocks[l], h)
            h = self.dropout(h)

        # -----------------------------
        # Backwards-compatible label alignment
        # -----------------------------
        # h is for dst nodes of the LAST block
        # y must match that same dst node set length
        if y is not None and isinstance(blocks, (list, tuple)) and len(blocks) > 0:
            ndst = blocks[-1].num_dst_nodes()
            if y.shape[0] != ndst:
                # DGLBlock convention: dst nodes correspond to the first ndst nodes
                y = y[:ndst]

        label_embed = self.label_emb(y)
        h = h + label_embed

        logits = self.classifier(h)
        return logits
