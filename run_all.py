import os.path
from datetime import datetime

import logging
import logging.config

def config_logging(exp_suffix):
    t0 = datetime.now()
    time_str = t0.strftime("%y%m%d_%H%M")
    results_folder = os.path.abspath(f"results/logs_{time_str}_{exp_suffix}")
    if not os.path.isdir(results_folder):  os.mkdir(results_folder)
    logging.config.dictConfig(get_config_dict(results_folder, time_str))
    return results_folder

def get_config_dict(results_folder, time_str):
    log_config = {
        "version": 1,
        "root":{
            "handlers" : ["console", "file"],
            "level": "DEBUG"
        },
        "handlers":{
            "console":{
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "level": "DEBUG"
            },
            "file": {
                "filename": os.path.join(results_folder, f'{time_str}_red_button.log'),
                "formatter": "standard",
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "maxBytes": 10485760, # 10MB
                "backupCount": 20,
                "encoding": "utf8",
            }
        },
        "formatters":{
            "standard": {
                "format": '[%(asctime)s %(levelname)s %(name)s] %(message)s',
                "datefmt":'%Y%m%d-%H:%M:%S'
            }
        },
    }
    return log_config

import gif_experiment

def run_all(models, datasets, results_folder, **kwargs):

    logging.info("\n###################\n")
    logging.info("Running with")
    logging.info(f" - models: {models}")
    logging.info(f" - datasets: {datasets}")
    logging.info(f" - parameters: {kwargs}")

    rep = str(kwargs.pop("repetitions", 30))
    outch = str(kwargs.pop("out_channels", 30))
    sfx = str(kwargs.pop("exp_suffix", ""))
    cont = str(kwargs.pop("contamination", 5))
    seed = str(kwargs.pop("seed", -1))
    verbose = kwargs.pop("verbose", True)

    logging.info(f"Experiment suffix {sfx}")
    logging.info(f"Global seed: {seed}")

    if len(kwargs) > 0:
        logging.error(f"The following arguments were not understood: {kwargs}")
        raise ValueError(f"The following arguments were not understood: {kwargs}")

    for model in models:
        for dataset in datasets:
            print()
            logging.info("#######################################################")
            logging.info(f"### Running {model} on {dataset}")
            cmd = ["--dataset", dataset, "--model", model,
                   "--base-folder", results_folder,
                   "--repetitions", rep,
                   "--contamination", cont,
                   "--out-channels", outch,
                   "--seed", seed]
            if verbose:
                cmd.append("--verbose")

            logging.info("Arguments: " + " ".join(cmd))
            gif_experiment.main(cmd)


if __name__ == "__main__":

    params = dict(
        repetitions=30, verbose=True, seed=31012022,
        out_channels=512, contamination=5)

    # wcci 2022
    models = ["arma", "grnf", "netsimile-bare"]
    datasets = ["delaunay", "sbm", "proteins", "imdb-binary", "mnist"]

    # logging setup
    exp_suffix = "wcci22"
    if len(models) == 1:
        exp_suffix += f"_{models[0]}"
    results_folder = config_logging(exp_suffix)
    params["results_folder"] = results_folder

    run_all(models=models, datasets=datasets, **params)

    params["contamination"] = 0
    run_all(models=models, datasets=datasets, **params)

    models = ["netsimile-nf", "gcn"]
    params["contamination"] = 5
    run_all(models=models, datasets=datasets, **params)
