# Graph iForest

This repository is the official implementation of our paper:

Daniele Zambon, Lorenzo Livi, Cesare Alippi. [Graph iForest: Isolation of anomalous and outlier graphs.](#) IEEE WCCI IJCNN 2022.


## Requirements

The code is implemented in Python 3.8. To install requirements:

```setup
pip install -r requirements.txt
```

## Experiments

To generate Figures 1 and 2 of the paper:

```setup
python draw_figures.py
```

instead, Table II, III, IV and V can obtained by running script `gif_experiment.py` passing the appropriate arguments or,
to run all experiments at once sequentially:

```setup
python run_all.py
```

Results are stored in folder `results`.



