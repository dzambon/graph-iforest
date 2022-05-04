import torch


def get_model(name, dataset, **kwargs):

    out_channels = kwargs.pop("channels", 256)

    if name == "grnf":
        from grnf.torch import GRNF
        model = GRNF(channels=out_channels, in_node_channels=dataset.num_node_features, in_edge_channels=dataset.num_edge_features)
        model.extension_level = 0

    elif name == "arma":
        if dataset.num_edge_features > 1:
            raise NotImplementedError
        else:
            model = GenericConv(layer="arma", num_node_features=dataset.num_node_features, out_channels=out_channels)
        model.extension_level = 0

    elif name == "gcn":
        model = GenericConv(layer="gcn", num_node_features=dataset.num_node_features, out_channels=out_channels)
        model.extension_level = 0

    elif name == "generalconv": # TODO not working
        model = GenericConv(layer="generalconv", num_node_features=dataset.num_node_features, out_channels=out_channels)
        model.extension_level = 0

    elif name == "netsimile-nf":
        from netsimile import NetSimile
        model = NetSimile(with_node_features=True)
        model.extension_level = 1

    elif name == "netsimile-bare":
        from netsimile import NetSimile
        model = NetSimile(with_node_features=False)
        model.extension_level = 1

    else:
        raise ValueError(f"Model {name} not recognized.")

    if (dataset[0].x is not None and dataset[0].x.dtype == torch.float64) or \
        (dataset[0].edge_attr is not None and dataset[0].edge_attr.dtype == torch.float64):
        model.double()

    model.name = name
    return model



def get_eccconv(dataset):

    from torch.nn import Linear, ReLU
    from torch.nn import Sequential as torch_Sequantial
    from torch_geometric.nn.conv import ECConv
    from torch_geometric.nn.glob import global_add_pool
    from torch_geometric.nn import Sequential as pyg_Sequantial
    hidden_dim = 32
    out_dim = 128
    h_edge1 = torch_Sequantial(Linear(in_features=dataset.num_edge_features, out_features=hidden_dim),
                         ReLU(inplace=True),
                         Linear(in_features=hidden_dim, out_features=hidden_dim * dataset.num_edge_features))
    h_edge2 = torch_Sequantial(Linear(in_features=dataset.num_edge_features, out_features=32),
                         ReLU(inplace=True),
                         Linear(in_features=out_dim, out_features=out_dim * hidden_dim))
    model = pyg_Sequantial('x, edge_index, edge_attr, batch', [
        (ECConv(in_channels=dataset.num_node_features, out_channels=hidden_dim, nn=h_edge1), 'x, edge_index, edge_attr -> x1'),
        ReLU(inplace=True),
        (ECConv(in_channels=hidden_dim, out_channels=out_dim, nn=h_edge2), 'x1, edge_index, edge_attr -> x2'),
        (global_add_pool, 'x2, batch -> x2'),
    ])
    return model


class GenericConv(torch.nn.Module):
    def __init__(self, layer, num_node_features, out_channels=128):
        super().__init__()
        num_node_features = max([num_node_features, 1]) # because I pass features equal to 1.0 if no feature is given
        if layer == "arma":
            from torch_geometric.nn import ARMAConv
            self.convs = torch.nn.ModuleList([
                            ARMAConv(in_channels=num_node_features, out_channels=out_channels, num_stacks=2, num_layers=2)
                        ])
        elif layer == "gcn":
            from torch_geometric.nn import GCNConv
            self.convs = torch.nn.ModuleList([
                            GCNConv(in_channels=num_node_features, out_channels=128),
                            GCNConv(in_channels=128, out_channels=out_channels)
                        ])
        elif layer == "generalconv":
            from torch_geometric.nn import GeneralConv
            self.convs = torch.nn.ModuleList([
                            GeneralConv(in_channels=num_node_features, out_channels=128),
                            GeneralConv(in_channels=128, out_channels=out_channels)
                        ])
        self.activation = torch.nn.ReLU(inplace=True)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, data):
        from torch_geometric.nn.glob import global_add_pool
        if data.x is not None:
            x_ = data.x
        else:
            x_ = torch.ones((data.num_nodes, 1), device=data.edge_index.device, dtype=self.convs[0].bias.dtype)
        for i, conv in enumerate(self.convs):
            x_ = conv(x=x_, edge_index=data.edge_index, edge_weight=data.edge_weight)
            self.activation(x_)
        return global_add_pool(x_, batch=data.batch)
