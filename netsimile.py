import torch
from torch_scatter import scatter_mean
import torch_geometric.utils.degree


def get_topological_node_features(data):

    simile_node_feat = torch.empty((data.num_nodes, 7))

    row, col = data.edge_index
    # 0 degrees_
    simile_node_feat[:, 0] = torch_geometric.utils.degree(index=row, num_nodes=data.num_nodes)
    # 2 mean_degrees_
    simile_node_feat[:, 2] = scatter_mean(src=simile_node_feat[col, 0], index=row)

    for node in range(data.num_nodes):
        # init
        node_mask = torch.full((data.num_nodes,), dtype=torch.bool, fill_value=False)
        edge_mask = torch.full((data.edge_index.shape[1],), dtype=torch.bool, fill_value=False)
        # select central node
        node_mask[node] = True
        # select 1-hop neighborhood
        torch.index_select(node_mask, 0, data.edge_index[0], out=edge_mask)
        node_mask[col[edge_mask]] = True
        # 4: ego_edge_: count egonet edges
        simile_node_feat[node, 4] = torch.sum(node_mask[row] & node_mask[col])
        # select 1-hop and 2-hop neighborhood
        torch.index_select(node_mask, 0, data.edge_index[0], out=edge_mask)
        node_mask[col[edge_mask]] = True
        # count 1- and 2-hop edges minus egonet edges ...continues below
        simile_node_feat[node, 5] = torch.sum(node_mask[row]) - simile_node_feat[node, 4]
        # count 1- and 2-hop neighbors ...continues below
        node_mask[col[edge_mask]] = True
        simile_node_feat[node, 6] = torch.sum(node_mask)

    # 5: ego_edge_out_: count out edges egonet: 1- and 2-hop edges minus egonet edges
    simile_node_feat[:, 5] = simile_node_feat[:, 5] - simile_node_feat[:, 4]
    # 6: ego_node_out_: count out nodes egonet: 1- and 2-hop neighbors minus 1-hop neighbors
    simile_node_feat[:, 6] = simile_node_feat[:, 6] - simile_node_feat[:, 0] - 1

    # 1: clus_
    simile_node_feat[:, 1] = (simile_node_feat[:, 4] - simile_node_feat[:, 0]) / (simile_node_feat[:, 0] * (simile_node_feat[:, 0] - 1.0))
    if torch.any(simile_node_feat[:, 0] <= 1):  # sets to zero anytime it can't divide
        simile_node_feat[torch.where(simile_node_feat[:, 0] <= 1)[0], 1] = 0.0
    if data.is_undirected():
        simile_node_feat[:, 1] = simile_node_feat[:, 1] * 2
        simile_node_feat[:, 4] = simile_node_feat[:, 4] / 2

    # 3: mean_clus_
    simile_node_feat[:, 3] = scatter_mean(src=simile_node_feat[col, 1], index=data.edge_index[0])

    return simile_node_feat


class NetSimile(torch.nn.Module):
    """
    https://arxiv.org/pdf/1209.2684.pdf
    """
    def __init__(self, with_node_features=False):
        super().__init__()
        self.with_node_features = with_node_features

    def forward(self, data):

        topo_feat = get_topological_node_features(data)
        if data.x is None or not self.with_node_features:
            all_features = topo_feat
        else:
            # add node features if present
            all_features = torch.hstack([
                data.x,
                get_topological_node_features(data)
            ])

        # num_nodes = data.num_nodes
        common_args = dict(index=data.batch, dim=0)
        mean = scatter_mean(src=all_features, **common_args)
        # std = scatter_std(src=all_features, **common_args)
        # std_ = torch.max(input=std, other=torch.tensor(1e-4))**4
        centred_val = all_features - mean[data.batch]
        m2 = scatter_mean(src=centred_val ** 2, **common_args)
        m3 = scatter_mean(src=centred_val ** 3, **common_args)
        m4 = scatter_mean(src=centred_val ** 4, **common_args)
        median = torch.vstack(
            [torch.median(all_features[torch.where(data.batch == i)], dim=0)[0] for i in range(data.num_graphs)])
        signature = torch.hstack([
            mean,
            median,
            torch.sqrt(m2),
            m3 / torch.max(input=m2 ** 1.5, other=torch.tensor(1e-4)),
            m4 / torch.max(input=m2 ** 2, other=torch.tensor(1e-4)),
        ])
        return signature


if __name__ == "__main__":
    ## debug
    import datasets

    dataset = datasets.get_dataset(name="delaunay", verbose=True)
    from torch_geometric.data import DataLoader

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    import networkx as nx

    for data in loader:
        node_feat = get_topological_node_features(data)
