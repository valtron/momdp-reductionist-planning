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

- `algo`: algorithms for estimating the Pareto front (Benson, Chord, NLS, and OLS)
- `env`: environments (Deep Sea Tresure and Bonus World)


## Generating the graphs

To generate the graphs in the paper, run:

```bash
python3 graphs.py
```

This takes about three minutes. Figures are saved in the subfolder `figs`.
