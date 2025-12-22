import torch
import torch.nn as nn
import torch.optim as opt
import dgl
import numpy as np
from math import sqrt
from dgl.base import DGLError
from dgl.nn.functional import edge_softmax
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.utils import Identity
import torch.nn.functional as F


class PosEncoding(nn.Module):

    def __init__(self, dim, device, base=10000, bias=0):
        """
        Initialize the position encoding component
        :param dim: the encoding dimension
        :param device: where to train
        :param base: the base for angle calculation
        :param bias: the bias
        """
        super(PosEncoding, self).__init__()
        self.dim = dim
        self.device = device
        self.base = base
        self.bias = bias

    def forward(self, pos):
        pos = pos + self.bias
        div = torch.exp(torch.arange(0, self.dim, 2) *
                        (-np.log(self.base) / self.dim)).to(self.device)
        pe = torch.zeros(pos.shape[0], self.dim).to(self.device)
        pe[:, 0::2] = torch.sin(pos.unsqueeze(1) * div)
        pe[:, 1::2] = torch.cos(pos.unsqueeze(1) * div)
        return pe


class TransformerConv(nn.Module):

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        bias=True,
        allow_zero_in_degree=False,
        # feat dropout
        feat_drop=0.0,
        # attention dropout
        attn_drop=0.0,
        # edge feat dropout
        edge_drop=0.0,
        # negative slope of LeakyReLU
        negative_slope=0.2,
        # residual connection
        residual=False,
        # activation
        activation=None,
        # output attention
        output_attn=False,
        # layer normalization
        norm=None,
        # share weights
        share_weights=False,
        # scale
        scale=None,
        # shortcut
        shortcut=False,
        # edge feature
        edge_feat=False,
        # use edge residual
        edge_res=False,
        # use edge normalization
        edge_norm=False,
        # use edge attention
        edge_attn=False,
        # use edge attention
        edge_attn_drop=0.0,
        # use edge attention
        edge_attn_bias=True,
        # use edge attention
        edge_attn_agg="mul",
    ):
        super().__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._num_heads = num_heads
        self._allow_zero_in_degree = allow_zero_in_degree
        self._feat_drop = nn.Dropout(feat_drop)
        self._attn_drop = nn.Dropout(attn_drop)
        self._edge_drop = nn.Dropout(edge_drop)
        self._edge_attn_drop = nn.Dropout(edge_attn_drop)
        self._negative_slope = negative_slope
        self._residual = residual
        self._output_attn = output_attn
        self._norm = norm
        self._share_weights = share_weights
        self._scale = scale
        self._shortcut = shortcut
        self._edge_feat = edge_feat
        self._edge_res = edge_res
        self._edge_norm = edge_norm
        self._edge_attn = edge_attn
        self._edge_attn_bias = edge_attn_bias
        self._edge_attn_agg = edge_attn_agg

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=bias
            )
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=bias
            )
        else:
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=bias
            )
            if share_weights:
                self.fc_dst = self.fc_src
            else:
                self.fc_dst = nn.Linear(
                    self._in_src_feats, out_feats * num_heads, bias=bias
                )

        self.fc_value = nn.Linear(
            self._in_src_feats, out_feats * num_heads, bias=bias
        )

        if shortcut:
            self.fc_shortcut = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=bias
            )

        if edge_feat:
            self.fc_edge = nn.Linear(out_feats, out_feats * num_heads, bias=bias)

        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=bias
                )
            else:
                self.res_fc = Identity()

        if norm is not None:
            self.norm = norm(out_feats * num_heads)

        if edge_norm:
            self.edge_norm = nn.BatchNorm1d(out_feats * num_heads)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_value.weight, gain=gain)
        if self._shortcut:
            nn.init.xavier_normal_(self.fc_shortcut.weight, gain=gain)
        if self._edge_feat:
            nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        if self._residual and isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        if self._norm is not None:
            self.norm.reset_parameters()
        if self._edge_norm:
            self.edge_norm.reset_parameters()

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False, edge_feat=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, output for those nodes will be invalid. "
                        "This is harmful for some applications, causing silent performance regression. "
                        "Adding self-loop on the input graph by calling `g = dgl.add_self_loop(g)` will resolve the issue. "
                        "Setting ``allow_zero_in_degree`` to be `True` when constructing this module will suppress the check and let the code run."
                    )

            if isinstance(feat, tuple):
                h_src = self._feat_drop(feat[0])
                h_dst = self._feat_drop(feat[1])
            else:
                h_src = h_dst = self._feat_drop(feat)

            feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            feat_v = self.fc_value(h_src).view(-1, self._num_heads, self._out_feats)

            if self._shortcut:
                feat_s = self.fc_shortcut(h_dst).view(
                    -1, self._num_heads, self._out_feats
                )

            if self._scale is not None:
                feat_src = feat_src * self._scale

            graph.srcdata.update({"ft": feat_src, "fv": feat_v})
            graph.dstdata.update({"ft": feat_dst})

            if edge_feat is not None:
                ef = self._edge_drop(edge_feat)
                ef = self.fc_edge(ef).view(-1, self._num_heads, self._out_feats)
                graph.edata["ef"] = ef

            graph.apply_edges(dgl.function.u_add_v("ft", "ft", "a"))

            if self._edge_feat and edge_feat is not None:
                if self._edge_attn:
                    if self._edge_attn_agg == "mul":
                        graph.edata["a"] = graph.edata["a"] * graph.edata["ef"]
                    elif self._edge_attn_agg == "add":
                        graph.edata["a"] = graph.edata["a"] + graph.edata["ef"]
                else:
                    graph.edata["a"] = graph.edata["a"] + graph.edata["ef"]

            e = nn.functional.leaky_relu(graph.edata["a"], self._negative_slope)
            graph.edata["sa"] = edge_softmax(graph, e)
            graph.edata["sa"] = self._attn_drop(graph.edata["sa"])

            if self._edge_attn and edge_feat is not None:
                graph.edata["sa"] = self._edge_attn_drop(graph.edata["sa"])

            graph.update_all(dgl.function.u_mul_e("fv", "sa", "m"), dgl.function.sum("m", "rst"))
            rst = graph.dstdata["rst"]

            if self._shortcut:
                rst = rst + feat_s

            if self._residual:
                resval = self.res_fc(h_dst).view(-1, self._num_heads, self._out_feats)
                rst = rst + resval

            rst = rst.flatten(1)

            if self._norm is not None:
                rst = self.norm(rst)

            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata["sa"]
            else:
                return rst


class Tabular1DCNN2(nn.Module):
    """Tabular 1D CNN encoder used for neighbor-stat features.

    NOTE (IEEE patch):
    For some datasets (e.g., IEEE-CIS), neighbor-stat features may be unavailable,
    resulting in input_dim == 0. The original implementation used depthwise Conv1d
    with groups=input_dim, which crashes when input_dim==0.

    This patched version disables the CNN path when input_dim <= 0 and provides
    a safe forward() that returns an empty embedding tensor of shape (B, 0, embed_dim).
    Downstream code should skip using neighbor embeddings when no neighbor stats exist.
    """
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        K: int = 4,  # K*input_dim -> hidden dim
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = int(input_dim) if input_dim is not None else 0
        self.embed_dim = embed_dim
        self.K = K
        self.disabled = self.input_dim <= 0

        # If disabled, do NOT construct any Conv layers (would be invalid).
        if self.disabled:
            # Keep these for callers that might inspect attributes.
            self.hid_dim = 0
            self.cha_input = self.cha_output = 0
            self.cha_hidden = 0
            self.sign_size1 = 2 * embed_dim
            self.sign_size2 = embed_dim
            return

        # ---- Original architecture (unchanged) ----
        self.hid_dim = self.input_dim * embed_dim * 2
        self.cha_input = self.cha_output = self.input_dim
        self.cha_hidden = (self.input_dim * K) // 2
        self.sign_size1 = 2 * embed_dim
        self.sign_size2 = embed_dim

        self.bn1 = nn.BatchNorm1d(self.input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(self.input_dim, self.hid_dim)

        self.bn_cv1 = nn.BatchNorm1d(self.cha_input)
        self.conv1 = nn.Conv1d(
            in_channels=self.cha_input,
            out_channels=self.cha_input * self.K,
            kernel_size=5,
            padding=2,
            groups=self.cha_input,
            bias=False
        )

        self.ave_pool1 = nn.AdaptiveAvgPool1d(self.sign_size2)

        self.bn_cv2 = nn.BatchNorm1d(self.cha_input * self.K)
        self.dropout2 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            in_channels=self.cha_input * self.K,
            out_channels=self.cha_input * self.K,
            kernel_size=3,
            padding=1,
            groups=self.cha_input * self.K,
            bias=False
        )

        self.bn_cv3 = nn.BatchNorm1d(self.cha_input * self.K)
        self.conv3 = nn.Conv1d(
            in_channels=self.cha_input * self.K,
            out_channels=self.cha_input * (self.K // 2),
            kernel_size=3,
            padding=1,
            # groups=self.cha_hidden,
            bias=True
        )

        self.bn_cvs = nn.ModuleList()
        self.convs = nn.ModuleList()
        for _ in range(6):
            self.bn_cvs.append(nn.BatchNorm1d(self.cha_input * (self.K // 2)))
            self.convs.append(nn.Conv1d(
                in_channels=self.cha_input * (self.K // 2),
                out_channels=self.cha_input * (self.K // 2),
                kernel_size=3,
                padding=1,
                # groups=self.cha_hidden,
                bias=True
            ))

        self.bn_cv10 = nn.BatchNorm1d(self.cha_input * (self.K // 2))
        self.conv10 = nn.Conv1d(
            in_channels=self.cha_input * (self.K // 2),
            out_channels=self.cha_output,
            kernel_size=3,
            padding=1,
            # groups=self.cha_hidden,
            bias=True
        )

    def forward(self, x):
        # Disabled path: return empty embedding of shape (B, 0, embed_dim)
        if getattr(self, "disabled", False):
            if x is None:
                # best-effort fallback
                return torch.zeros((1, 0, self.embed_dim), device="cpu")
            b = x.shape[0]
            return x.new_zeros((b, 0, self.embed_dim))

        x = self.dropout1(self.bn1(x))
        x = nn.functional.celu(self.dense1(x))
        x = x.reshape(x.shape[0], self.cha_input, self.sign_size1)

        x = self.bn_cv1(x)
        x = nn.functional.relu(self.conv1(x))
        x = self.ave_pool1(x)

        x_input = x
        x = self.dropout2(self.bn_cv2(x))
        x = nn.functional.relu(self.conv2(x))  # -> (|b|,24,32)
        x = x + x_input

        x = self.bn_cv3(x)
        x = nn.functional.relu(self.conv3(x))  # -> (|b|,6,32)

        for i in range(6):
            x_input = x
            x = self.bn_cvs[i](x)
            x = nn.functional.relu(self.convs[i](x))
            x = x + x_input

        x = self.bn_cv10(x)
        x = nn.functional.relu(self.conv10(x))

        return x


class TransEmbedding(nn.Module):

    def __init__(
        self,
        df=None,
        device='cpu',
        dropout=0.2,
        in_feats_dim=82,
        cat_features=None,
        neigh_features: dict = None,
        att_head_num: int = 4,  # yelp 4 amazon 5 S-FFSD 9
        neighstat_uni_dim=64
    ):
        """
        Initialize the attribute embedding and feature learning compoent
        :param df: the pandas dataframe
        :param device: where to train model
        :param dropout: the dropout rate
        :param in_feats_dim: the shape of input feature in dimension 1
        :param cat_features: category features
        :param neigh_features: neighbor riskstat features
        :param att_head_num: attention head number for riskstat embeddings
        """
        super(TransEmbedding, self).__init__()
        self.time_pe = PosEncoding(dim=in_feats_dim, device=device, base=100)

        self.cat_table = nn.ModuleDict({col: nn.Embedding(max(df[col].unique(
        ))+1, in_feats_dim).to(device) for col in cat_features if col not in {"Labels", "Time"}})

        self.nei_table = None
        if isinstance(neigh_features, dict) and len(neigh_features) > 0:
            self.nei_table = Tabular1DCNN2(input_dim=len(neigh_features), embed_dim=in_feats_dim)

        self.att_head_num = att_head_num
        self.att_head_size = int(in_feats_dim / att_head_num)
        self.total_head_size = in_feats_dim
        self.lin_q = nn.Linear(in_feats_dim, self.total_head_size)
        self.lin_k = nn.Linear(in_feats_dim, self.total_head_size)
        self.lin_v = nn.Linear(in_feats_dim, self.total_head_size)

        self.lin_final = nn.Linear(in_feats_dim, in_feats_dim)
        self.layer_norm = nn.LayerNorm(in_feats_dim, eps=1e-8)

        self.neigh_mlp = nn.Linear(in_feats_dim, 1)

        self.neigh_add_mlp = nn.ModuleList([nn.Linear(in_feats_dim, in_feats_dim) for _ in range(neighstat_uni_dim)]) if isinstance(neigh_features, dict) else None

        self.dropout = nn.Dropout(dropout)
        self.forward_mlp = nn.ModuleList([nn.Linear(in_feats_dim, in_feats_dim) for _ in range(
            len(cat_features))]) if isinstance(cat_features, list) else None

    def forward_emb(self, cat_feat):
        support = {}
        for col in cat_feat.keys():
            if col not in {"Labels", "Time"}:
                support[col] = self.cat_table[col](cat_feat[col])
            else:
                if col == "Time":
                    support[col] = self.time_pe(cat_feat[col].to(torch.float32))
        return support

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.att_head_num, self.att_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward_neigh_emb(self, neighstat_feat):
        # Guard: IEEE (or other datasets) may not provide neighbor-stat features.
        if (neighstat_feat is None) or (not isinstance(neighstat_feat, dict)) or (len(neighstat_feat) == 0) or (self.nei_table is None):
            return None, []

        cols = neighstat_feat.keys()
        tensor_list = []
        for col in cols:
            tensor_list.append(neighstat_feat[col])
        neis = torch.stack(tensor_list).T
        input_tensor = self.nei_table(neis)

        mixed_q_layer = self.lin_q(input_tensor)
        mixed_k_layer = self.lin_k(input_tensor)
        mixed_v_layer = self.lin_v(input_tensor)

        q_layer = self.transpose_for_scores(mixed_q_layer)
        k_layer = self.transpose_for_scores(mixed_k_layer)
        v_layer = self.transpose_for_scores(mixed_v_layer)

        att_scores = torch.matmul(q_layer, k_layer.transpose(-1, -2))
        att_scores = att_scores / sqrt(self.att_head_size)

        att_probs = nn.Softmax(dim=-1)(att_scores)
        context_layer = torch.matmul(att_probs, v_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.total_head_size,)
        context_layer = context_layer.view(*new_context_shape)
        hidden_states = self.lin_final(context_layer)
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states, cols

    def forward(self, cat_feat: dict, neighstat_feat: dict):
        support = self.forward_emb(cat_feat)
        cat_output = 0
        nei_output = 0
        for i, k in enumerate(support.keys()):
            support[k] = self.dropout(support[k])
            support[k] = self.forward_mlp[i](support[k])
            cat_output = cat_output + support[k]

        if (neighstat_feat is not None) and isinstance(neighstat_feat, dict) and (len(neighstat_feat) > 0) and (self.nei_table is not None):
            nei_embs, cols_list = self.forward_neigh_emb(neighstat_feat)
            nei_output = self.neigh_mlp(nei_embs).squeeze(-1)

        return cat_output, nei_output


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
        neighbor_data=None,
    ):
        super(RGTAN, self).__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.heads = heads
        self.activation = activation
        self.n_layers = n_layers
        self.drop = drop
        self.device = device
        self.gated = gated

        self.ref_df = ref_df
        self.cat_features = cat_features
        self.neigh_features = neigh_features
        self.nei_att_head = nei_att_head
        self.neighbor_data = neighbor_data

        self.layers = nn.ModuleList()
        self.layers.append(
            TransformerConv(
                in_feats=in_feats,
                out_feats=hidden_dim,
                num_heads=heads[0],
                feat_drop=drop[0] if isinstance(drop, list) else drop,
                attn_drop=drop[1] if isinstance(drop, list) else drop,
                residual=False,
                activation=activation,
                allow_zero_in_degree=True,
            )
        )

        for l in range(1, n_layers):
            self.layers.append(
                TransformerConv(
                    in_feats=hidden_dim * heads[l - 1],
                    out_feats=hidden_dim,
                    num_heads=heads[l],
                    feat_drop=drop[0] if isinstance(drop, list) else drop,
                    attn_drop=drop[1] if isinstance(drop, list) else drop,
                    residual=True,
                    activation=activation,
                    allow_zero_in_degree=True,
                )
            )

        self.layers.append(
            nn.Linear(hidden_dim * heads[-1], n_classes)
        )

        self.dropout = nn.Dropout(drop[0] if isinstance(drop, list) else drop)

        self.trans_emb = TransEmbedding(
            df=ref_df,
            device=device,
            dropout=drop[0] if isinstance(drop, list) else drop,
            in_feats_dim=in_feats,
            cat_features=cat_features,
            neigh_features=neigh_features,
            att_head_num=nei_att_head,
        )

        self.label_emb = nn.Embedding(3, hidden_dim * heads[-1]).to(device)

    def forward(self, blocks, x, y, x1, neigh_input=None):
        if isinstance(x1, dict):
            x_cat, x_nei = self.trans_emb(x1, neigh_input)
            x = x + x_cat
        else:
            x_nei = 0

        h = x

        for l in range(self.n_layers):
            h = self.layers[l](blocks[l], h)
            h = self.dropout(h)

        label_embed = self.label_emb(y)
        h = h + label_embed

        logits = self.layers[-1](h)

        return logits
