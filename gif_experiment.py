from tqdm import tqdm
import logging
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib
colors = list(matplotlib.cm.get_cmap("Set1").colors + matplotlib.cm.get_cmap("Pastel1").colors)
colors[:2] = colors[:2][::-1]  # otherwise the red is used for nominal class
from datetime import datetime
t0 = datetime.now()
time_str = t0.strftime("%y%m%d_%H%M")

import pandas as pd
import torch
from torch_geometric.loader import DataLoader
import numpy as np
from scipy.stats import gaussian_kde, mannwhitneyu

import eif
import models
import datasets


def predict_dataloader(model, loader, to_numpy_f64=False, desc=None,  verbose=False):
    z = []
    y_true = []
    for step, data in tqdm(enumerate(loader), disable=not verbose, total=len(loader),
                           desc=f"{model.name} psi(g;w) for all w and g" if desc is None else desc):
        z.append(model(data))
        y_true.append(data.y)
    z = torch.cat(z, dim=0)
    y_true = torch.cat(y_true, dim=0)
    if to_numpy_f64:
        return z.numpy().astype("float64"), \
               y_true.numpy().astype("float64")
    else:
        return z, y_true

def run_single_simulation(model, dataset, contamination, fpr=0.05, verbose=False, num_training_data=300):

    cl0_idx = torch.where(dataset.data.y == 0)[0].tolist()
    cl0_train = np.random.permutation(cl0_idx)[:num_training_data]

    if contamination > 0:
        num_cont = np.round(len(cl0_train) * 0.01 * contamination).astype(int)
        cl_anom_idx = torch.where(dataset.data.y != 0)[0].tolist()
        cl_anon_train = np.random.permutation(cl_anom_idx)[:num_cont]
        train_idx = np.concatenate([cl0_train[:-num_cont], cl_anon_train])
        np.random.shuffle(train_idx)

    else:
        train_idx = cl0_train

    train_dataset = dataset[train_idx]
    test_dataset = dataset.shuffle()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    if verbose:
        logging.info(f"Dataset {dataset.name} (contamination={contamination}) with:")
        for c in dataset.classes:
            logging.info(f" - Class {c} has {len(torch.where(dataset.data.y == c)[0].tolist())} graphs")

        logging.info(f'The experiment considers:')
        logging.info(f' - {len(train_dataset)} training graphs')
        logging.info(f' - {len(test_dataset)} test graphs in total')

    z_train, y_train = predict_dataloader(model, train_loader, to_numpy_f64=True, verbose=verbose,
                                          desc="psi(g;w) for all w and g in train set")
    # assert np.all(y_train == 0)
    z, y = predict_dataloader(model, test_loader, to_numpy_f64=True, verbose=verbose,
                                          desc="psi(g;w) for all w and g in test set")

    if verbose:
        logging.info("Running iForest....")

    seed_iforset_ = np.random.randint(0, 2**16-1)
    logging.info(f"Seed used in eif.iForest = {seed_iforset_}")
    iForest = eif.iForest(z_train, ntrees=200, sample_size=min([256, len(train_dataset)]),
                          ExtensionLevel=model.extension_level, seed=seed_iforset_)

    scores_train = iForest.compute_paths(X_in=z_train)
    scores_train.sort()
    scores = iForest.compute_paths(X_in=z)

    threshold = np.percentile(scores_train, q= int(100*(1-fpr)))

    s0 = scores[y == 0]
    si = scores[y != 0]
    u, pval = mannwhitneyu(si, s0, alternative="greater")
    u = u / len(s0) / len(si) 

    results = dict(auc={}, pval={}, fpr={})
    results["auc"]["joint"] = u
    for i, c_ in enumerate(np.unique(y)):
        si = scores[y == c_]
        u, pval = mannwhitneyu(si, s0, alternative="greater")
        u = u / len(s0) / len(si)
        results["auc"][c_] = u
        results["pval"][c_] = pval
        results["fpr"][c_] = np.sum(si > threshold) / si.shape[0]



    intermediate_data = dict(scores=scores, z_train=z_train, z=z, y=y, threshold=threshold)
    return iForest, results, intermediate_data


def main(raw_args=None):

    args = parser.parse_args(raw_args)

    seed_ = np.random.randint(0, 2**32-1) if args.seed < 0 else args.seed
    np.random.seed(seed_)
    torch.manual_seed(seed_)
    logging.info(f"Seed (numpy and torch) set to = {seed_}")

    results_folder = os.path.abspath(args.base_folder)

    if args.figures or (args.repetitions == 1 and not args.histogram and not args.scatterplot and not args.path_lengths):
        args.histogram = True
        args.scatterplot = True
        args.path_lengths = True

    if args.histogram or args.scatterplot or args.path_lengths:
        assert args.repetitions == 1

    params = dict(out_channels=args.out_channels,
                  contamination=args.contamination)

    dataset = datasets.get_dataset(name=args.dataset, verbose=args.verbose)
    model = models.get_model(name=args.model, dataset=dataset, **params)

    results_list = []
    for r in tqdm(range(args.repetitions), desc="Run repeated simulations"):
        logging.info(f"Run {r+1}/{args.repetitions} of {model} on {dataset}")
        iForest, results, intermediate_data = run_single_simulation(dataset=dataset, model=model,
                                                                    contamination=args.contamination,
                                                                    verbose=args.verbose)
        results_list.append(results)
        logging.info(results)

    if args.repetitions > 1:
        if not os.path.isdir(results_folder):
            os.mkdir(results_folder)
        df_res = {stat: pd.DataFrame([r_[stat] for r_ in results_list]) for stat in results.keys()}
        logging.info("Saving files: ")
        for stat, df in df_res.items():
            fname = os.path.join(results_folder, f"r{args.repetitions}_{dataset.name}_c{args.contamination}_M{args.out_channels}_{model.name}_{stat}_{time_str}.csv")
            logging.info(f" - {fname}")
            df.to_csv(fname, float_format='%.3f')

    fig_base_fname = f"{dataset.name}_{model.name}_{time_str}.pdf"

    if args.histogram:
        logging.info("Figure score distribution....")
        xs = np.linspace(min(intermediate_data["scores"]), max(intermediate_data["scores"]), 200)
        plt.figure(figsize=(4, 3))
        s0_ = intermediate_data["scores"][intermediate_data["y"] == 0]
        for i, c_ in enumerate(np.unique(intermediate_data["y"])):
            si_ = intermediate_data["scores"][intermediate_data["y"] == c_]
            density = gaussian_kde(si_)
            density.covariance_factor = lambda : .25
            density._compute_covariance()
            # plt.plot(xs, density(xs), label=f'cl[{int(c_)}] AUC:{results["auc"][c_]:.3f} p:{results["pval"][c_]:.3f}', color=colors[i])
            plt.plot(xs, density(xs), label=f'Cl. {int(c_)} (AUC:{results["auc"][c_]:.3f})', color=colors[i])
        plt.axvline(x=intermediate_data["threshold"],  color='k', linestyle='--', label="95% threshold")
        # plt.title(f"hist_{dataset.name}_{model.name}")
        plt.title(f"{dataset.name}")
        plt.legend()
        filename = os.path.join(results_folder, "hist_" + fig_base_fname)
        plt.savefig(filename)
        logging.info(f"saving {filename}")
        plt.show()


    #################################################################################################
    if args.scatterplot:
        logging.info("TSNE....")
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, init="pca", learning_rate="auto")
        z_tsne = tsne.fit_transform(intermediate_data["z"])

        logging.info("Figure t-SNE scatter ....")
        f = plt.figure(figsize=(4, 3))
        plt.subplot(1, 1, 1)
        ss0 = np.argsort(intermediate_data["scores"])
        sizes = (intermediate_data["scores"]-intermediate_data["scores"].min()/16)**3 *400
        for i, c_ in enumerate(np.unique(intermediate_data["y"])):
            idx_ = intermediate_data["y"] == c_
            plt.scatter(z_tsne[idx_, 0], z_tsne[idx_, 1], s=sizes[idx_],
                        alpha=.25, edgecolors='none', color=colors[i], label=f"Cl. {c_}")
        plt.legend()
        # plt.title(f"t-SNE_scatter_{dataset.name}_{model.name}")
        plt.title(f"{dataset.name}")
        filename = os.path.join(results_folder, "tsne_" + fig_base_fname)
        plt.savefig(filename)
        logging.info(f"saving {filename}")
        plt.show()

    if args.path_lengths:
        import eif_old
        import igraph as ig

        def branch2num(branch, init_root=0):
            num = [init_root]
            for b in branch:
                if b == 'L':
                    num.append(num[-1] * 2 + 1)
                if b == 'R':
                    num.append(num[-1] * 2 + 2)
            return num

        def gen_graph(branches, g=None, init_root=0, pre=''):
            num_branches = [branch2num(i, init_root) for i in branches]
            all_nodes = [j for branch in num_branches for j in branch]
            all_nodes = np.unique(all_nodes)
            all_nodes = all_nodes.tolist()
            if g is None:
                g = ig.Graph()
            for k in all_nodes: g.add_vertex(pre + str(k))
            t = []
            for j in range(len(branches)):
                branch = branch2num(branches[j], init_root)
                for i in range(len(branch) - 1):
                    pair = [branch[i], branch[i + 1]]
                    if pair not in t:
                        t.append(pair)
                        g.add_edge(pre + str(branch[i]), pre + str(branch[i + 1]))
            return g, max(all_nodes)

        iForest_old = eif_old.iForest(intermediate_data["z_train"], ntrees=200, limit=15,
                                      sample_size=min([256, intermediate_data["z_train"].shape[0]]))
        scores = iForest_old.compute_paths(X_in=intermediate_data["z"])
        ss = np.argsort(scores)

        # retrieve tree
        idx = 0
        jt = idx
        T = iForest_old.Trees[jt]
        gg, nn = gen_graph(eif_old.all_branches(T.root, [], None, ), None, 0, '0_')

        # Tree
        vstyle = {}
        vstyle["vertex_size"] = [2.5] * gg.vcount()
        vstyle["vertex_color"] = ['black'] * gg.vcount()
        # vstyle["vertex_label"]=g.vs['name']
        vstyle["vertex_label_dist"] = 2
        vstyle["bbox"] = (400, 400)
        vstyle["edge_color"] = [(0, 0., 0.)] * gg.ecount()
        vstyle["edge_width"] = [0.4] * gg.ecount()
        vstyle["layout"] = gg.layout_reingold_tilford(root=[0])
        vstyle["edge_curved"] = 0.00
        vstyle["margin"] = 10
        # ig.plot(gg,**vstyle)

        # Path of a nominal graph
        nominal_idx = np.where(intermediate_data["y"]==dataset.classes[0])[0][idx]
        P = eif_old.PathFactor(intermediate_data["z"][nominal_idx], T)
        Gn = branch2num(P.path_list)
        lb = gg.get_shortest_paths('0_' + str(Gn[0]), '0_' + str(Gn[-1]))[0]
        le = gg.get_eids([(lb[i], lb[i + 1]) for i in range(len(lb) - 1)])
        for j in le:
            vstyle["edge_color"][j] = colors[0]
            vstyle["edge_width"][j] = 1.9
        for v in lb:
            vstyle["vertex_color"][v] = colors[0]

        # Path of an anomaly
        anomaly_idx = np.where(intermediate_data["y"]==dataset.classes[1])[0][idx]
        P = eif_old.PathFactor(intermediate_data["z"][anomaly_idx], T)
        Gn = branch2num(P.path_list)
        lb = gg.get_shortest_paths('0_' + str(Gn[0]), '0_' + str(Gn[-1]))[0]
        le = gg.get_eids([(lb[i], lb[i + 1]) for i in range(len(lb) - 1)])
        for j in le:
            vstyle["edge_color"][j] = colors[1]
            vstyle["edge_width"][j] = 1.9
        for v in lb:
            vstyle["vertex_color"][v] = colors[1]

        # Save figure
        filename = os.path.join(results_folder, "treepaths_" + fig_base_fname)
        ig.plot(gg, filename, **vstyle)
        logging.info(f"saving {filename}")
        ig.plot(gg,**vstyle)

        def getVals(forest, x, sorted=True):
            theta = np.linspace(0, 2 * np.pi, forest.ntrees)
            r = []
            for i in range(forest.ntrees):
                temp = forest.compute_paths_single_tree(np.array([x]), i)
                r.append(temp[0])
            if sorted:
                r = np.sort(np.array(r))
            return r, theta

        Sorted=True
        fig = plt.figure(figsize=(6, 3.5))
        ax1 = plt.subplot(111)

        label = {"label": "Nominal graph"}
        rn, thetan = getVals(iForest, intermediate_data["z"][nominal_idx], sorted=Sorted)
        for j in range(len(rn)):
            ax1.plot([thetan[j], thetan[j]], [0, rn[j]], color=colors[0], alpha=1, lw=1, **label)
            label = {}

        label = {"label": "Anomalous graph"}
        ra, thetaa = getVals(iForest, intermediate_data["z"][anomaly_idx], sorted=Sorted)
        for j in range(len(ra)):
            ax1.plot([thetaa[j], thetaa[j]], [0, ra[j]], color=colors[1], alpha=0.9, lw=1.3, **label)
            label = {}

        ax1.set_xticklabels([])
        ax1.set_xlabel("Trees")
        ax1.set_ylabel("Path Length")
        ax1.set_ylim(0, iForest.limit)
        plt.legend()

        filename = os.path.join(results_folder, "pathlen_hist" + fig_base_fname)
        plt.savefig(filename)
        logging.info(f"saving {filename}")
        plt.show()

parser = argparse.ArgumentParser()

parser.add_argument('--model', '-m', type=str, action='store', required=True,
                    help='model')
parser.add_argument('--dataset', '-d', type=str, action='store', required=True,
                    help='dataset')

parser.add_argument('--repetitions', '-r', type=int, action='store', default=1,
                    help='')
parser.add_argument('--out-channels', type=int, action='store', default=1,
                    help='')
parser.add_argument('--contamination', type=int, action='store', default=1,
                    help='')

parser.add_argument('--figures', '-f', action='store_true',
                    help='')
parser.add_argument('--histogram', action='store_true',
                    help='')
parser.add_argument('--scatterplot', action='store_true',
                    help='')
parser.add_argument('--path-lengths', action='store_true',
                    help='')

parser.add_argument('--seed', type=int, action='store', default=-1,
                    help='')
parser.add_argument('--base-folder', type=str, action='store', default="results",
                    help='')
parser.add_argument('--verbose', '-v', action='store_true',
                    help='')

if __name__ == "__main__":
    main()
    logging.info("The end!")
