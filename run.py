from typing import Optional
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import funcli

from env import deep_sea_treasure, bonus_world, lqr, hopper
from common.hypervolume import hypervolume_convex
from common import dualdesc as dd, misc
from algo import benson, nls, ols, approx

ENVS = [deep_sea_treasure, bonus_world, lqr, hopper]
ALGOS = [nls, ols, benson, approx]

def main(seeds: int = 1, figdir: Optional[Path] = None):
	if figdir is not None:
		figdir.mkdir(exist_ok = True)
	graph_error_vs_queries(seeds, figdir)
	#graph_pareto_fronts(figdir)

def graph_error_vs_queries(seeds, outdir):
	# make graphs showing how approximation error decreases as a function
	# of # of queries for each algorithm and on both environments
	
	cache = _CACHE / 'hv'
	
	pbar1 = tqdm(ENVS)
	for env in pbar1:
		pbar1.set_postfix({ 'env': env.name })
		entry = cache / env.name
		if not entry.exists():
			pf_verts = env.pareto_front_vertices()
			true_hv = hypervolume_convex(pf_verts, env.min_return)
			entry.save(true_hv)
		true_hv = entry.load()
		B = np.linalg.norm(env.max_return - env.min_return, np.inf)
		
		data = {}
		pbar2 = tqdm(ALGOS)
		for algo_module in pbar2:
			pbar2.set_postfix({ 'algo': algo_module.name })
			data[algo_module] = sweep_queries(algo_module, env, seeds)
		
		# error vs. queries
		fig, ax = plt.subplots(figsize = (4, 3), constrained_layout = True)
		ax.set_title(env.name)
		ax.set_xlabel("# of queries")
		ax.set_ylabel("approximation error")
		ax.set_yscale('log')
		min_y = np.inf
		for algo_module in ALGOS:
			d = data[algo_module]
			plot_runs(ax, d[0,:,0], d[:,:,1], algo_module)
			min_y = min(min_y, np.min(np.mean(d[:,:,1], axis = 0)))
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.set_ylim([max(min_y * 0.9, B * 1e-5), B])
		ax.legend()
		if outdir:
			fig.savefig(outdir / 'error-{}.pdf'.format(env.name))
		else:
			plt.show()
		
		# hypervolume vs. queries
		fig, ax = plt.subplots(figsize = (4, 3), constrained_layout = True)
		ax.set_title("Hypervolume error vs. # queries")
		ax.set_xlabel("# queries")
		ax.set_ylabel("Hypervolume error")
		ax.set_yscale('log')
		min_y = np.inf
		for algo_module in ALGOS:
			if not algo_module.inner_approximation:
				continue
			d = data[algo_module]
			plot_runs(ax, d[0,:,0], true_hv - d[:,:,2], algo_module)
			min_y = min(min_y, np.min(np.mean(true_hv - d[:,:,2], axis = 0)))
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.set_ylim([max(min_y * 0.9, true_hv * 1e-5), true_hv])
		ax.legend()
		
		if outdir:
			fig.savefig(outdir / 'hv-{}.pdf'.format(env.name))
		else:
			plt.show()

def plot_runs(ax, x, y, algo_module):
	ly = np.log(np.maximum(y, np.max(y) * 1e-6))
	lmean = np.mean(ly, axis = 0)
	if len(y) > 1:
		lse = np.std(ly, axis = 0, ddof = 1) / np.sqrt(len(y))
		mn = np.exp(lmean - 3 * lse)
		mx = np.exp(lmean + 3 * lse)
		ax.fill_between(x, mn, mx, color = algo_module.color, alpha = 0.1)
	ax.plot(x, np.exp(lmean), color = algo_module.color, label = algo_module.name)

def sweep_queries(algo_module, env, seeds):
	if algo_module.deterministic:
		seeds = 1
	
	q_eval = { 1, 5, 10, 15, *range(20, 100 + 1, 10) }
	cache = _CACHE / 'eval/algo={},env={}'.format(algo_module.name, env.name)
	
	data = []
	for seed in tqdm(range(seeds), "seed", leave = False):
		rng = np.random.default_rng(seed)
		entry = cache / 'seed={}'.format(seed)
		if not entry.exists():
			data_seed = sweep_queries_run(algo_module, env, q_eval, rng)
			entry.save(data_seed)
		data_seed = entry.load()
		data.append(data_seed)
	
	return np.array(data)

def sweep_queries_run(algo_module, env, q_eval, rng):
	q_max = max(q_eval)
	
	data = []
	B = np.linalg.norm(env.max_return - env.min_return, np.inf)
	ws_eval = _get_eval_weights(env.k)
	pf_verts = env.pareto_front_vertices()
	hs_true = _eval_support(pf_verts, ws_eval)
	solver = env.solver
	algo = algo_module.Algo(env.min_return, env.max_return, rng = rng)
	
	with tqdm(total = q_max, desc = "q", leave = False) as pbar:
		q = 0
		done = False
		while not done:
			done = algo.update(solver)
			q += 1
			pbar.update(1)
			done = done or (q >= q_max)
			if done or (q in q_eval):
				estimated_pf = algo.approx_poly.get_generators()[0]
				hs_est = _eval_support(estimated_pf, ws_eval)
				error = np.linalg.norm(hs_true - hs_est, np.inf)
				if algo_module.inner_approximation:
					hv = hypervolume_convex(estimated_pf, env.min_return)
				else:
					hv = 0
				data.append([q, error, hv])
				done = done or (error <= 1e-4 * B)
	
	while len(data) < q_max:
		data.append(data[-1])
	
	return np.array(data)

def graph_pareto_fronts(outdir):
	# for a given epsilon, show the e-PF found by each algorithm on DST
	
	env = deep_sea_treasure
	epsilon = 1.8
	V = env.pareto_front_vertices()
	
	D_lower = dd.Polytope.FromGenerators(V - epsilon, -np.eye(env.k))
	D_lower.add_halfspace(-np.eye(env.k), -env.min_return)
	V_lower = D_lower.get_generators()[0]
	V_lower = _cheby_sort(V_lower[np.any(V_lower > env.min_return + 1e-4, axis = -1)])
	
	D_upper = dd.Polytope.FromGenerators(V + epsilon, -np.eye(env.k))
	D_upper.add_halfspace(-np.eye(env.k), -env.min_return)
	V_upper = D_upper.get_generators()[0]
	V_upper = _cheby_sort(V_upper[np.any(V_upper > env.min_return + 1e-4, axis = -1)])
	
	for algo_module in ALGOS:
		estimated_pf, num_queries = estimate_pf(env, algo_module, epsilon)
		estimated_pf.add_halfspace(-np.eye(env.k), -env.min_return)
		estimated_pf = estimated_pf.get_generators()[0]
		estimated_pf = _cheby_sort(estimated_pf[np.any(estimated_pf > env.min_return + 1e-4, axis = 1)])
		print(algo_module.name, len(estimated_pf) - 3, "Pareto-efficient facets")
		
		fig, ax = plt.subplots(figsize = (3, 2.3), constrained_layout = True)
		ax.fill(
			np.concatenate([V_upper[:,0], V_lower[::-1,0]]),
			np.concatenate([V_upper[:,1], V_lower[::-1,1]]),
			color = 'black', alpha = 0.1, linewidth = 0,
		)
		ax.plot(V[:,0], V[:,1], 'k-', linewidth = 1, label = "Pareto front")
		y_min = np.minimum(np.min(V, axis = 0), np.min(estimated_pf, axis = 0))
		ax.fill(
			np.concatenate([estimated_pf[:,0], [y_min[0]]]),
			np.concatenate([estimated_pf[:,1], [y_min[1]]]),
			color = algo_module.color, alpha = 0.1, linewidth = 0,
		)
		ax.plot(estimated_pf[:,0], estimated_pf[:,1], linewidth = 1, color = algo_module.color, label = algo_module.name)
		ax.set_xlabel("Objective 1")
		ax.set_xlim([np.min(V[:,0]), np.max(V[:,0]) + epsilon])
		ax.spines['right'].set_visible(False)
		ax.set_ylabel("Objective 2")
		ax.set_ylim([np.min(V[:,1]), np.max(V[:,1]) + epsilon])
		ax.spines['top'].set_visible(False)
		
		if outdir:
			fig.savefig(outdir / 'pf-{}.pdf'.format(algo_module.name))
		else:
			plt.show()

def estimate_pf(env, algo_module, epsilon):
	pf_verts = env.pareto_front_vertices()
	hs_true = _eval_support(pf_verts, ws_eval)
	solver = env.solver
	algo = algo_module.Algo(env.min_return, env.max_return, rng = np.random.default_rng(0))
	done = False
	q = 0
	while True:
		estimated_pf = algo.approx_poly.get_generators()[0]
		hs_est = _eval_support(estimated_pf, ws_eval)
		error = np.linalg.norm(hs_true - hs_est, np.inf)
		if done or error <= epsilon:
			break
		done = algo.update(solver)
		q += 1
	return algo.approx_poly, q

def _cheby_sort(points):
	return points[np.argsort(points @ [1, -1])]

def _get_eval_weights(k):
	assert k in { 2, 3 }, k
	if k == 2:
		n = 1000
	else:
		n = 50
	return misc.grid_simplex_covering(n, k)

def _eval_support(ys, ws):
	return np.max(np.sum(ys[:,:,None] * ws.T, axis = 1), axis = 0)

_CACHE = misc.Cache('.cache')

if __name__ == '__main__':
	funcli.main()
