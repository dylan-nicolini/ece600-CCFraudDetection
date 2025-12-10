# %%
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.io import loadmat
import torch
import dgl
import random
import os
import time
import argparse
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
# from . import *
DATADIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "data/")


def featmap_gen(tmp_df=None):
    """
    Handle S-FFSD dataset and do some feature engineering
    :param tmp_df: the feature of input dataset
    """
    # time_span = [2, 5, 12, 20, 60, 120, 300, 600, 1500, 3600, 10800, 32400, 64800, 129600,
    #              259200]  # Increase in the number of time windows to increase the characteristics.
    time_span = [2, 3, 5, 15, 20, 50, 100, 150,
                 200, 300, 864, 2590, 5100, 10000, 24000]
    time_name = [str(i) for i in time_span]
    time_list = tmp_df['Time']
    post_fe = []
    for trans_idx, trans_feat in tqdm(tmp_df.iterrows()):
        new_df = pd.Series(trans_feat)
        temp_time = new_df.Time
        temp_amt = new_df.Amount
        for length, tname in zip(time_span, time_name):
            lowbound = (time_list >= temp_time - length)
            upbound = (time_list <= temp_time)
            correct_data = tmp_df[lowbound & upbound]
            new_df['trans_at_avg_{}'.format(
                tname)] = correct_data['Amount'].mean()
            new_df['trans_at_totl_{}'.format(
                tname)] = correct_data['Amount'].sum()
            new_df['trans_at_std_{}'.format(
                tname)] = correct_data['Amount'].std()
            new_df['trans_at_bias_{}'.format(
                tname)] = temp_amt - correct_data['Amount'].mean()
            new_df['trans_at_num_{}'.format(tname)] = len(correct_data)
            new_df['trans_target_num_{}'.format(tname)] = len(
                correct_data.Target.unique())
            new_df['trans_location_num_{}'.format(tname)] = len(
                correct_data.Location.unique())
            new_df['trans_type_num_{}'.format(tname)] = len(
                correct_data.Type.unique())
        post_fe.append(new_df)
    return pd.DataFrame(post_fe)


def sparse_to_adjlist(sp_matrix, filename):
    """
    Transfer sparse matrix to adjacency list
    :param sp_matrix: the sparse matrix
    :param filename: the filename of adjlist
    """
    # add self loop
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    # create adj_list
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    with open(filename, 'wb') as file:
        pickle.dump(adj_lists, file)
    file.close()


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def MinMaxScaling(data):
    mind, maxd = data.min(), data.max()
    # return mind + (data - mind) / (maxd - mind)
    return (data - mind) / (maxd - mind)


def k_neighs(
    graph: dgl.DGLGraph,
    center_idx: int,
    k: int,
    where: str,
    choose_risk: bool = False,
    risk_label: int = 1
) -> torch.Tensor:
    """return indices of risk k-hop neighbors

    Args:
        graph (dgl.DGLGraph): dgl graph dataset
        center_idx (int): center node idx
        k (int): k-hop neighs
        where (str): {"predecessor", "successor"}
        risk_label (int, optional): value of fruad label. Defaults to 1.
    """
    target_idxs: torch.Tensor
    if k == 1:
        if where == "in":
            neigh_idxs = graph.predecessors(center_idx)
        elif where == "out":
            neigh_idxs = graph.successors(center_idx)

    elif k == 2:
        if where == "in":
            subg_in = dgl.khop_in_subgraph(
                graph, center_idx, 2, store_ids=True)[0]
            neigh_idxs = subg_in.ndata[dgl.NID][subg_in.ndata[dgl.NID] != center_idx]
            # delete center node itself
            neigh1s = graph.predecessors(center_idx)
            neigh_idxs = neigh_idxs[~torch.isin(neigh_idxs, neigh1s)]
        elif where == "out":
            subg_out = dgl.khop_out_subgraph(
                graph, center_idx, 2, store_ids=True)[0]
            neigh_idxs = subg_out.ndata[dgl.NID][subg_out.ndata[dgl.NID] != center_idx]
            neigh1s = graph.successors(center_idx)
            neigh_idxs = neigh_idxs[~torch.isin(neigh_idxs, neigh1s)]

    neigh_labels = graph.ndata['label'][neigh_idxs]
    if choose_risk:
        target_idxs = neigh_idxs[neigh_labels == risk_label]
    else:
        target_idxs = neigh_idxs

    return target_idxs


def count_risk_neighs(
    graph: dgl.DGLGraph,
    risk_label: int = 1
) -> torch.Tensor:

    ret = []
    for center_idx in graph.nodes():
        neigh_idxs = graph.successors(center_idx)
        neigh_labels = graph.ndata['label'][neigh_idxs]
        risk_neigh_num = (neigh_labels == risk_label).sum()
        ret.append(risk_neigh_num)

    return torch.Tensor(ret)


def feat_map():
    tensor_list = []
    feat_names = []
    for idx in tqdm(range(graph.num_nodes())):
        neighs_1_of_center = k_neighs(graph, idx, 1, "in")
        neighs_2_of_center = k_neighs(graph, idx, 2, "in")

        tensor = torch.FloatTensor([
            edge_feat[neighs_1_of_center, 0].sum().item(),
            # edge_feat[neighs_1_of_center, 0].std().item(),
            edge_feat[neighs_2_of_center, 0].sum().item(),
            # edge_feat[neighs_2_of_center, 0].std().item(),
            edge_feat[neighs_1_of_center, 1].sum().item(),
            # edge_feat[neighs_1_of_center, 1].std().item(),
            edge_feat[neighs_2_of_center, 1].sum().item(),
            # edge_feat[neighs_2_of_center, 1].std().item(),
        ])
        tensor_list.append(tensor)

    feat_names = ["1hop_degree", "2hop_degree",
                  "1hop_riskstat", "2hop_riskstat"]

    tensor_list = torch.stack(tensor_list)
    return tensor_list, feat_names
if __name__ == "__main__":

    set_seed(42)

    # -------------------------------------------------------------
    # S-FFSD ONLY: load, feature-engineer, and build graph
    # -------------------------------------------------------------
    print("processing S-FFSD data...")

    sffsd_csv = os.path.join(DATADIR, "S-FFSD.csv")
    if not os.path.exists(sffsd_csv):
        raise FileNotFoundError(f"Expected S-FFSD.csv at {sffsd_csv}")

    # Load base S-FFSD transactions
    data = pd.read_csv(sffsd_csv)

    # Feature engineering
    data = featmap_gen(data.reset_index(drop=True))
    data.replace(np.nan, 0, inplace=True)

    # Save and reload (keeps behavior close to original)
    sffsd_neo_csv = os.path.join(DATADIR, "S-FFSDneofull.csv")
    data.to_csv(sffsd_neo_csv, index=None)
    data = pd.read_csv(sffsd_neo_csv).reset_index(drop=True)

    # -------------------------------------------------------------
    # Optimized graph construction for S-FFSD
    #   - preserves original semantics:
    #       group by column, sort by Time,
    #       connect each txn to the next 3 txns in that group
    #   - avoids slow Python nested loops over rows
    # -------------------------------------------------------------
    print("Building S-FFSD graph (optimized)...")

    pair = ["Source", "Target", "Location", "Type"]
    edge_per_trans = 3

    all_src = []
    all_dst = []

    # Pre-sort once by Time so each group is time-ordered
    data_sorted_time = data.sort_values("Time")

    for col in pair:
        print(f"  processing edges for group: {col}")
        # group on the column, but keep within-group order by Time
        for _, gdf in tqdm(data_sorted_time.groupby(col), desc=col):
            idx = gdf.index.to_numpy()
            if len(idx) <= 1:
                continue
            # connect idx[i] -> idx[i+1], idx[i+2], ... up to edge_per_trans
            for j in range(1, edge_per_trans + 1):
                if len(idx) > j:
                    src = idx[:-j]
                    dst = idx[j:]
                    all_src.append(src)
                    all_dst.append(dst)

    if len(all_src) == 0:
        raise RuntimeError("No edges were created for S-FFSD graph; check your data distribution.")

    all_src = np.concatenate(all_src)
    all_dst = np.concatenate(all_dst)

    g = dgl.graph((all_src, all_dst))

    # Encode categorical columns and attach node features / labels
    cal_list = ["Source", "Target", "Location", "Type"]
    for col in cal_list:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].apply(str).values)

    feat_data = data.drop("Labels", axis=1)
    labels = data["Labels"]

    g.ndata["label"] = torch.from_numpy(labels.to_numpy()).to(torch.long)
    g.ndata["feat"] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

    graph_bin_path = os.path.join(DATADIR, "graph-S-FFSD.bin")
    dgl.data.utils.save_graphs(graph_bin_path, [g])

    # -------------------------------------------------------------
    # Neighbor risk-aware feature generation for S-FFSD ONLY
    # -------------------------------------------------------------
    print("Generating neighbor risk-aware features for S-FFSD dataset...")

    graph = dgl.load_graphs(graph_bin_path)[0][0]
    graph: dgl.DGLGraph
    print(f"graph info: {graph}")

    # Base edge features: in-degree + count of risky neighbors
    degree_feat = graph.in_degrees().unsqueeze_(1).float()
    risk_feat = count_risk_neighs(graph).unsqueeze_(1).float()

    origin_feat_name = ["degree", "riskstat"]
    edge_feat = torch.cat([degree_feat, risk_feat], dim=1)

    # expose to feat_map() via globals (as in original code)
    globals()["graph"] = graph
    globals()["edge_feat"] = edge_feat

    # Higher-order neighbor statistics
    features_neigh, feat_names = feat_map()
    features_neigh = torch.cat((edge_feat, features_neigh), dim=1).numpy()
    feat_names = origin_feat_name + feat_names
    features_neigh[np.isnan(features_neigh)] = 0.0

    # Scale and save neighbor features
    output_path = os.path.join(DATADIR, "S-FFSD_neigh_feat.csv")
    features_neigh_df = pd.DataFrame(features_neigh, columns=feat_names)
    scaler = StandardScaler()
    features_neigh_df = pd.DataFrame(
        scaler.fit_transform(features_neigh_df),
        columns=features_neigh_df.columns,
    )
    features_neigh_df.to_csv(output_path, index=False)

    print(f"Done. Neighbor features written to: {output_path}")

