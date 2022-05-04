from tqdm import tqdm
import os
import glob
from distutils.dir_util import copy_tree

import numpy as np
import networkx as nx
import pickle

import torch
from torch_geometric.data import InMemoryDataset, Data

from cdg.graph.delaunay import DelaunayGraphs
from cdg.graph.sbm import StochasticBlockModel

import logging

def pre_process_to_float32(data):
    if data.x is not None:
        data.x = data.x.type(torch.float)
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr.type(torch.float)
    return data

def get_dataset(name, verbose=False):

    if name == "delaunay":
        dataset = DelaunayPyG(root='data/Delaunay/', name="Del513",
                              classes=[0, 5, 7, 9, 11, 13], no_graphs=500,
                              pre_transform=pre_process_to_float32)  # yes

    elif name == "sbm":
        dataset = SBMPyG(root='data/SBM/', name="SBM6",
                         classes=[0, 10, 20, 40, 70, 100], no_graphs=500,
                         pre_transform=pre_process_to_float32)  # yes

    elif name == "mnist":
        from torch_geometric.datasets import GNNBenchmarkDataset
        dataset = GNNBenchmarkDataset(root='data/GNNBenchmarkDataset', name='MNIST',
                                      pre_transform=pre_process_to_float32)  # qualcosa

    elif name in ["proteins", "imdb-binary"]:
        from torch_geometric.datasets import TUDataset
        # dataset = TUDataset(root='data/TUDataset', name='MUTAG') # naa
        dataset = TUDataset(root='data/TUDataset', name=name.upper(),
                            pre_transform=pre_process_to_float32)  # mezzo
        # dataset = TUDataset(root='data/TUDataset', name='ENZYMES') # zero proprio

    else:
        raise NotImplementedError(f"Unknown dataset {name}" )

    max_num_graphs = 8000
    if len(dataset) > max_num_graphs:
        logging.warning(f"Maximum number of graphs ({max_num_graphs}) exceeded. Dataset {dataset.name} has {len(dataset)}. Random downsampling applied.")
        dataset = dataset.copy(np.random.permutation(len(dataset))[:8000])

    dataset.classes = [int(c) for c in dataset.data.y.unique()]

    if verbose:

        print()
        logging.info(f'Dataset: {dataset}:')
        logging.info('====================')
        logging.info(f'Number of graphs: {len(dataset)}')
        logging.info(f'Number of features: {dataset.num_features}')
        logging.info(f'Number of node features: {dataset.num_node_features}')
        logging.info(f'Number of edge features: {dataset.num_edge_features}')
        logging.info(f'Number of classes: {dataset.num_classes}')
        for c in dataset.classes:
            logging.info(f' - Number of graphs in class {c}: {len(torch.where(dataset.data.y == c)[0].tolist())}')

        logging.info(f'Avg. num. of nodes: {dataset.data.num_nodes / len(dataset)}')
        logging.info(f'Avg. num. of edges: {dataset.data.num_edges / len(dataset)}')
        print(dataset.data.num_nodes)

        data = dataset[0]  # Get the first graph object.
        print()
        print(data)
        logging.info('=============================================================')
        # Gather some statistics about the first graph.
        logging.info(f'Number of nodes: {data.num_nodes}')
        logging.info(f'Number of edges: {data.num_edges}')
        logging.info(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        logging.info(f'Has isolated nodes: {data.has_isolated_nodes()}')
        logging.info(f'Has self-loops: {data.has_self_loops()}')
        logging.info(f'Is undirected: {data.is_undirected()}')


    return dataset


class DelaunayPyG(InMemoryDataset):

    def __init__(self, root, name="Del", transform=None, pre_transform=None, pre_filter=None,
                 classes=[0, 5, 10, 20], **kwargs):
        self.classes = classes
        self.name = name
        self.del_args = kwargs
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, "processed")

    def raw_file_names_fun(self, cl=None):
        return f'data_cl{cl}.pkl'

    @property
    def raw_file_names(self):
        return [self.raw_file_names_fun(cl=c) for c in self.classes]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        graphs_ = DelaunayGraphs().get(no_graphs=self.del_args.get("no_graphs", 200),
                                       classes=self.classes)
        for cl, graphs in graphs_.items():
            with open(os.path.join(self.raw_dir, self.raw_file_names_fun(cl=cl)), "wb") as f:
                pickle.dump(graphs, f)

    def process(self):
        # Read data into huge `Data` list.
        graph_list = []
        targets = []
        for cl in self.classes:
            with open(os.path.join(self.raw_dir, self.raw_file_names_fun(cl=cl)), "rb") as f:
                graphs = pickle.load(f)

            graph_list += graphs
            targets += [cl]*len(graphs)

        data_list = []
        for g, y in zip(graph_list, targets):
            import numpy as np
            data = Data(x=torch.tensor(g.nf),
                        edge_index=torch.tensor(np.where(g.adj), dtype=torch.long),
                        y=torch.tensor(y, dtype=torch.float))
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])
        torch.save((data, slices), os.path.join(self.processed_dir, self.processed_file_names[0]))


class SBMPyG(InMemoryDataset):

    def __init__(self, root, name="SBM", transform=None, pre_transform=None, pre_filter=None,
                 classes=[0, 20, 50, 100], **kwargs):
        self.classes = classes
        self.name = name
        self.sbm_args = kwargs
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, "processed")

    def raw_file_names_fun(self, cl=None):
        return f'data_cl{cl}.pkl'

    @property
    def raw_file_names(self):
        return [self.raw_file_names_fun(cl=c) for c in self.classes]

    @property
    def processed_file_names(self):
        return ['data.pt']

    # def generate_graphs(no_graphs=self.del_args.get("no_graphs", 200),
    #                                classes=self.classes)

    def download(self):
        # Download to `self.raw_dir`.
        graphs_ = {}
        communities = self.sbm_args.get("communities", [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]])
        prob_matrix_base = np.array(self.sbm_args.get("prob_matrix", [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]))
        prob_matrix_null = np.array(self.sbm_args.get("prob_matrix_null", 1-prob_matrix_base))
        prob_matrix_null = np.array(self.sbm_args.get("prob_matrix_null", np.ones((3,3))/3.0))

        for cl in self.classes:
            factor = 1.0 - float(cl) * 0.01
            assert factor >= 0.0
            assert factor <= 1.0
            sbm = StochasticBlockModel(communities=communities,
                                       prob_matrix=prob_matrix_base * factor + prob_matrix_null * (1.0 - factor))
            graphs_[cl] = sbm.get(no_graphs=self.sbm_args.get("no_graphs", 200),
                                  format="npy")[0]

        for cl, graphs in graphs_.items():
            with open(os.path.join(self.raw_dir, self.raw_file_names_fun(cl=cl)), "wb") as f:
                pickle.dump(graphs, f)

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        import numpy as np
        for cl in self.classes:
            with open(os.path.join(self.raw_dir, self.raw_file_names_fun(cl=cl)), "rb") as f:
                adjs = pickle.load(f)
                # graph_list += graphs
                # targets += [cl]*len(graphs)
                for a in adjs:
                    data = Data(edge_index=torch.tensor(np.where(a), dtype=torch.long),
                                y=torch.tensor(cl, dtype=torch.float))
                    data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])
        torch.save((data, slices), os.path.join(self.processed_dir, self.processed_file_names[0]))
