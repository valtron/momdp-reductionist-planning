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

- `algo`: algorithms for estimating the Pareto front (Benson, Chord, NLS, and OLS) that use a LP-based MDP solver
- `env`: environments (Deep Sea Treasure, Bonus World, and Cartpole)
- `exp`: experiments (tabular and function approximation)
- `fig`: figures and graphs (output from scripts)
- `common`: things used throughout


## Generating the graphs

### Tabular experiments

To generate the graphs in the paper, run:

```bash
python3 -m exp.tabular
```

This takes about two minutes. Figures are saved in the subfolder `fig`.


### Function approximation experiments

The function approximation experiments take a long time. This command collects data and saves it in `.cache`:

```bash
python3 -m exp.fapprox collect <method> <runs> <splits> <index>
```

Arguments are:
- `method`: `nls`, `ncs`, or `ols`
- `runs`: number of runs
- `splits` and `index`: for splitting the work and only doing some parts of it.
	`splits` is how many pieces to split it into, and `0 <= index < splits` is which piece to work on.
	For no splitting, pass `1 0` (`splits = 1` and `index = 0`).

The experiments in the paper used 10 runs.

After the data has been collected, create the graphs with:

```bash
python3 -m exp.fapprox graphs <runs>
```

Figures are saved in `fig`.
