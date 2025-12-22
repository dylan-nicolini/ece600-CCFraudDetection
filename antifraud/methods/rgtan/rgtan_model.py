import torch
import torch.nn as nn
import torch.optim as optim
from dgl.utils import expand_as_pair
from dgl import function as fn
from dgl.base import DGLError
from dgl.nn.functional import edge_softmax
import numpy as np
import pandas as pd
from math import sqrt


class PosEncoding(nn.Module):

    def __init__(self, dim, device, base=10000, bias=0):

        super(PosEncoding, self).__init__()
        """
        Initialize the posencoding component
        :param dim: the encoding dimension
        :param device: where to train model
        :param base: base for sin and cos
        :param bias: bias for sin and cos
        """
        p = []
        sft = []
        for i in range(dim):
            b = (i - i % 2) / dim
            p.append(base ** -b)
            if i % 2:
                sft.append(np.pi / 2.0 + bias)
            else:
                sft.append(bias)
        self.device = device
        self.sft = torch.tensor(
            sft, dtype=torch.float32).view(1, -1).to(device)
        self.base = torch.tensor(p, dtype=torch.float32).view(1, -1).to(device)

    def forward(self, pos):
        with torch.no_grad():
            pos = pos.view(-1, 1)
            x = pos / self.base + self.sft
            return torch.sin(x)


class TransformerConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):

        super(TransformerConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._feat_drop = nn.Dropout(feat_drop)
        self._attn_drop = nn.Dropout(attn_drop)
        self._negative_slope = negative_slope
        self._residual = residual
        self._activation = activation

        self.fc_src = nn.Linear(
            self._in_src_feats, out_feats * num_heads, bias=bias)
        self.fc_dst = nn.Linear(
            self._in_dst_feats, out_feats * num_heads, bias=bias)

        self.fc_value = nn.Linear(
            self._in_src_feats, out_feats * num_heads, bias=bias)

        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))

        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=bias)
            else:
                self.res_fc = None

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_value.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self._residual and self.res_fc is not None:
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will '
                                   'resolve the issue. Setting '
                                   '``allow_zero_in_degree`` to be `True` when '
                                   'constructing this module will suppress the '
                                   'check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self._feat_drop(feat[0])
                h_dst = self._feat_drop(feat[1])
            else:
                h_src = h_dst = self._feat_drop(feat)

            feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            feat_v = self.fc_value(h_src).view(-1, self._num_heads, self._out_feats)

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el, 'fv': feat_v})
            graph.dstdata.update({'er': er})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = nn.functional.leaky_relu(graph.edata.pop('e'), self._negative_slope)
            graph.edata['a'] = edge_softmax(graph, e)
            graph.edata['a'] = self._attn_drop(graph.edata['a'])
            graph.update_all(fn.u_mul_e('fv', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            if self._residual:
                if self.res_fc is not None:
                    resval = self.res_fc(h_dst).view(-1, self._num_heads, self._out_feats)
                    rst = rst + resval
                else:
                    rst = rst + feat_dst

            rst = rst.flatten(1)

            if self._activation is not None:
                rst = self._activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class Tabular1DCNN2(nn.Module):
    """Tabular 1D CNN encoder used for neighbor-stat features.

    IEEE patch:
      - Some datasets (e.g., IEEE-CIS) may have *no* neighbor-stat columns.
      - In that case input_dim == 0, and the original depthwise Conv1d setup
        (groups=input_dim) crashes because groups must be a positive integer.

    This patched version disables itself when input_dim <= 0 and returns an
    empty embedding tensor in forward().
    """

    def __init__(self, input_dim, embed_dim, K=4, dropout=0.2):
        super(Tabular1DCNN2, self).__init__()
        self.input_dim = int(input_dim) if input_dim is not None else 0
        self.embed_dim = int(embed_dim)
        self.K = K

        # ✅ Disable neighbor-stat CNN when there are no neighbor-stat columns.
        self.disabled = self.input_dim <= 0
        if self.disabled:
            return

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
            bias=False,
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
            bias=False,
        )

        self.bn_cv3 = nn.BatchNorm1d(self.cha_input * self.K)
        self.conv3 = nn.Conv1d(
            in_channels=self.cha_input * self.K,
            out_channels=self.cha_input * (self.K // 2),
            kernel_size=3,
            padding=1,
            bias=True,
        )

        self.bn_cvs = nn.ModuleList()
        self.convs = nn.ModuleList()
        for _ in range(6):
            self.bn_cvs.append(nn.BatchNorm1d(self.cha_input * (self.K // 2)))
            self.convs.append(
                nn.Conv1d(
                    in_channels=self.cha_input * (self.K // 2),
                    out_channels=self.cha_input * (self.K // 2),
                    kernel_size=3,
                    padding=1,
                    bias=True,
                )
            )

        self.bn_cv10 = nn.BatchNorm1d(self.cha_input * (self.K // 2))
        self.conv10 = nn.Conv1d(
            in_channels=self.cha_input * (self.K // 2),
            out_channels=self.cha_output,
            kernel_size=3,
            padding=1,
            bias=True,
        )

    def forward(self, x):
        # Disabled path: return an empty embedding of shape (B, 0, embed_dim)
        if getattr(self, "disabled", False):
            if x is None:
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
        x = nn.functional.relu(self.conv2(x))
        x = x + x_input

        x = self.bn_cv3(x)
        x = nn.functional.relu(self.conv3(x))

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
        neigh_features=None,
        att_head_num: int = 4,
        neighstat_uni_dim=64
    ):
        """
        Initialize the attribute embedding and feature learning component
        :param df: the pandas dataframe
        :param device: where to train model
        :param dropout: the dropout rate
        :param in_feats_dim: embedding dim
        :param cat_features: categorical feature dict/list
        :param neigh_features: neighbor stat dict (may be empty for IEEE)
        :param att_head_num: attention head number for riskstat embeddings
        """
        super(TransEmbedding, self).__init__()
        self.time_pe = PosEncoding(dim=in_feats_dim, device=device, base=100)

        self.cat_table = nn.ModuleDict({
            col: nn.Embedding(max(df[col].unique()) + 1, in_feats_dim).to(device)
            for col in cat_features if col not in {"Labels", "Time"}
        })

        # ✅ IEEE patch: only build neighbor-stat CNN if dict is non-empty
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

        self.neigh_add_mlp = nn.ModuleList([nn.Linear(in_feats_dim, in_feats_dim) for _ in range(
            len(neigh_features.columns))]) if isinstance(neigh_features, pd.DataFrame) else None

        self.label_table = nn.Embedding(3, in_feats_dim, padding_idx=2).to(device)
        self.time_emb = None
        self.emb_dict = None
        self.label_emb = None
        self.cat_features = cat_features
        self.neigh_features = neigh_features

        # keep the existing behavior, but guard at call site
        self.forward_mlp = nn.ModuleList(
            [nn.Linear(in_feats_dim, in_feats_dim) for _ in range(len(cat_features))]
        )
        self.dropout = nn.Dropout(dropout)

    def forward_emb(self, cat_feat):
        if self.emb_dict is None:
            self.emb_dict = self.cat_table
        support = {
            col: self.emb_dict[col](cat_feat[col])
            for col in self.cat_features if col not in {"Labels", "Time"}
        }
        return support

    def transpose_for_scores(self, input_tensor):
        new_x_shape = input_tensor.size()[:-1] + (self.att_head_num, self.att_head_size)
        input_tensor = input_tensor.view(*new_x_shape)
        return input_tensor.permute(0, 2, 1, 3)

    def forward_neigh_emb(self, neighstat_feat):
        # ✅ IEEE patch: neighbor-stat features may be missing/empty
        if (neighstat_feat is None) or (not isinstance(neighstat_feat, dict)) or (len(neighstat_feat) == 0) or (self.nei_table is None):
            return None, []

        cols = neighstat_feat.keys()
        tensor_list = []
        for col in cols:
            tensor_list.append(neighstat_feat[col])

        # (B, num_cols)
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
            if self.forward_mlp is not None and i < len(self.forward_mlp):
                support[k] = self.forward_mlp[i](support[k])
            cat_output = cat_output + support[k]

        # ✅ IEEE patch: only use neighbor branch if dict non-empty + nei_table exists
        if (neighstat_feat is not None) and isinstance(neighstat_feat, dict) and (len(neighstat_feat) > 0) and (self.nei_table is not None):
            nei_embs, cols_list = self.forward_neigh_emb(neighstat_feat)
            if nei_embs is not None:
                nei_output = self.neigh_mlp(nei_embs).squeeze(-1)

        return cat_output, nei_output


class RGTAN(nn.Module):

    def __init__(self,
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
                 **kwargs):
        """
        Initialize the RGTAN model
        """
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

        self.layers = nn.ModuleList()
        self.layers.append(TransformerConv(
            in_feats=in_feats,
            out_feats=hidden_dim,
            num_heads=heads[0],
            feat_drop=drop[0] if isinstance(drop, list) else drop,
            attn_drop=drop[1] if isinstance(drop, list) else drop,
            residual=False,
            activation=activation,
            allow_zero_in_degree=True,
        ))

        for l in range(1, n_layers):
            self.layers.append(TransformerConv(
                in_feats=hidden_dim * heads[l - 1],
                out_feats=hidden_dim,
                num_heads=heads[l],
                feat_drop=drop[0] if isinstance(drop, list) else drop,
                attn_drop=drop[1] if isinstance(drop, list) else drop,
                residual=True,
                activation=activation,
                allow_zero_in_degree=True,
            ))

        self.classifier = nn.Linear(hidden_dim * heads[-1], n_classes)

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

        h = x
        for l in range(self.n_layers):
            h = self.layers[l](blocks[l], h)
            h = self.dropout(h)

        label_embed = self.label_emb(y)
        h = h + label_embed

        logits = self.classifier(h)
        return logits