import os.path
from datetime import datetime

import logging

import main

if __name__ == "__main__":
    model = "grnf"
    datasets = ["delaunay", "sbm"] #, "proteins", "imdb-binary", "mnist"]
    results_folder = os.path.abspath("./results/figures_wcci")

    for dataset in datasets:
        cmd = ["--dataset", dataset, "--model", model,
               "--base-folder", results_folder,
               "--repetitions", "1",
               "--contamination", "0",
               "--out-channels", "256",
               "--seed", "31012022",
               "--verbose",
               "--histogram", "--scatterplot"]
               # "--figures"]
               # "--path-lengths"]

        logging.info(f"### Running {model} on {dataset}")
        if not os.path.isdir(results_folder):  os.mkdir(results_folder)
        logging.info("Arguments: " + " ".join(cmd))
        main.main(cmd)

