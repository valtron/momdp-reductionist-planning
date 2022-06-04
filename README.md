# A Principled Reductionist Approach to Planning in Multi-Objective Markov Decision Processes

This repository contains code for the paper
"A Principled Reductionist Approach to Planning in Multi-Objective Markov Decision Processes".


## Setup

Python 3.6 or newer is required, as well as the libraries in `requirements.txt`.
To install them, run:

```bash
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```


## Folder structure

- `algo`: algorithms for estimating the Pareto front (Benson, NLS, OLS, and Approx)
- `env`: environments (Deep Sea Treasure, Bonus World, LQR, and Hopper)
- `fig`: figures and graphs (output from scripts)
- `common`: things used throughout


## Generating the graphs

To generate the graphs in the paper, run:

```bash
python3 -m run --figdir fig --seeds 50
```

This takes about ... . Figures are saved in the subfolder `fig`.
