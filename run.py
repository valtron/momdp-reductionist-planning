from typing import Optional
from pathlib import Path
import numpy as np
import scipy
from matplotlib import pyplot as plt, cm
from matplotlib.collections import LineCollection
from matplotlib.colors import LightSource
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import art3d
from tqdm.auto import tqdm
import funcli

import os
if os.name == 'nt':
	import ctypes
	from pathlib import Path
	# TODO: figure out why the default `winmode` flags don't work
	ctypes.CDLL(str(Path(np.__path__[0]).parent / 'pyparma/ppl.cp39-win_amd64.pyd'), winmode = 0)
import dualdesc as dd

from env import deep_sea_treasure, bonus_world, hopper, lqr
from common.hypervolume import hypervolume_convex
from common import misc, pareto
from algo import benson, nls, ols, sfa

ENVS = [deep_sea_treasure, bonus_world, hopper, lqr]
ALGOS = [nls, ols, benson, sfa]

def main(seeds: int = 1, figdir: Optional[Path] = None):
	if figdir is not None:
		figdir.mkdir(exist_ok = True)
	graph_error_vs_queries(seeds, figdir)
	graph_pareto_fronts(figdir)

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
		#ax.set_title(env.name)
		ax.set_xlabel("# of queries")
		ax.set_ylabel("Approximation error")
		ax.set_yscale('log')
		min_y = np.inf
		for algo_module in ALGOS:
			d = data[algo_module]
			plot_runs(ax, d[0,:,0], d[:,:,1], algo_module)
			min_y = min(min_y, np.min(np.mean(d[:,:,1], axis = 0)))
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.set_ylim([max(min_y * 0.9, B * 1e-4), B])
		ax.legend()
		if outdir:
			fig.savefig(outdir / 'error-{}.pdf'.format(env.name))
		else:
			plt.show()
		
		# hypervolume vs. queries
		fig, ax = plt.subplots(figsize = (4, 3), constrained_layout = True)
		#ax.set_title("Hypervolume error vs. # queries")
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
		ax.set_ylim([max(min_y * 0.9, true_hv * 3e-6), true_hv])
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
	# for each environment, plot the true PF
	color = [0.5, 0.5, 0.5, 1]
	
	for env in ENVS:
		verts = env.pareto_front_vertices()
		pl = dd.Polytope.FromGenerators(verts, -np.eye(env.k))
		pl.add_halfspace(-np.eye(env.k), -env.min_return)
		points, simplices, normals, offsets = _get_weakly_efficient_facets(pl)
		
		tmp = np.concatenate([normals, offsets[:,None]], axis = 1)
		tmp -= np.min(tmp, axis = 0)
		tmp *= 1e4 / np.max(tmp, axis = 0)
		tmp = tmp.astype(np.int32)
		tmp = set(map(tuple, tmp))
		
		max_return = np.max(verts, axis = 0)
		max_return += (max_return - env.min_return) * 0.05
		
		if env.k == 2:
			print(env.name, len(verts) + len(tmp))
			
			fig, ax = plt.subplots(constrained_layout = True)
			ax.set_xlim([env.min_return[0], max_return[0]])
			ax.set_ylim([env.min_return[1], max_return[1]])
			ax.set_xlabel(env.objective_names[0])
			ax.set_ylabel(env.objective_names[1])
			ax.grid()
			ax.add_collection(LineCollection(
				np.stack([points[simplices[:,0]], points[simplices[:,1]]], axis = 1),
				colors = [color[:-1]],
			))
		else:
			assert env.k == 3
			
			print(env.name, 2*(len(verts) + len(tmp)) - 1)
			
			fig, ax = make_3d_plot()
			ax.set_xlim([env.min_return[0], max_return[0]])
			ax.set_ylim([env.min_return[1], max_return[1]])
			ax.set_zlim([env.min_return[2], max_return[2]])
			ax.set_xlabel(env.objective_names[0])
			ax.set_ylabel(env.objective_names[1])
			ax.set_zlabel(env.objective_names[2])
			plot_polytope(ax, points, simplices, normals, offsets, color)
		
		if outdir:
			fig.savefig(outdir / 'pf-{}.png'.format(env.name))
		else:
			plt.show()

def make_3d_plot(**kwargs):
	ax_labels = kwargs.pop('ax_labels', None)
	fig = plt.figure(constrained_layout = True, **kwargs)
	ax = fig.add_subplot(projection = '3d')
	ax.view_init(elev = 40, azim = 45)
	if ax_labels is not None:
		ax.set_xlabel(ax_labels[0])
		ax.set_ylabel(ax_labels[1])
		ax.set_zlabel(ax_labels[2])
	return fig, ax

def plot_polytope(ax, points, simplices, normals, offsets, color):
	data = {}
	tmp = np.concatenate([normals, offsets[:,None]], axis = 1)
	tmp -= np.min(tmp, axis = 0)
	tmp *= 1e4 / np.max(tmp, axis = 0)
	for i, row in enumerate(tmp.astype(np.int32)):
		row = tuple(row)
		if row not in data:
			data[row] = { 'normal': normals[i], 'point_idxs': set() }
		data[row]['point_idxs'] |= set(simplices[i])
	
	polys = []
	poly_normals = []
	d = np.ones(3) / np.sqrt(3)
	for v in data.values():
		# sort around midpoint
		pts = points[sorted(v['point_idxs'])]
		m = np.mean(pts, axis = 0)
		pts_par = m + ((pts - m) @ d)[:,None] * d
		tmp = pts - pts_par
		b1 = tmp[0]
		b1 /= np.linalg.norm(b1)
		b2 = np.cross(d, b1)
		x1 = tmp @ b1
		x2 = tmp @ b2
		polys.append(pts[np.argsort(np.arctan2(x1, x2))])
		poly_normals.append(v['normal'])
	poly_normals = np.array(poly_normals)
	
	ls = LightSource(200, -30)
	colors = ls.shade_normals(poly_normals)
	colors = 0.9 * (colors[:,None] * color) + 0.1
	colors[:,-1] = color[-1]
	
	polyc = art3d.Poly3DCollection(polys)
	polyc.set_facecolors(colors)
	ax.add_collection(polyc)

def _get_weakly_efficient_facets(pl):
	V = pl.get_generators()[0]
	hull = scipy.spatial.ConvexHull(V)
	simplices = hull.simplices
	normals = hull.equations[:,:-1]
	tmp = np.linalg.norm(normals, axis = 1)
	normals = normals / tmp[:,None]
	offsets = hull.equations[:,-1] / tmp
	mask = np.all(normals >= -1e-4, axis = 1)
	simplices = simplices[mask]
	normals = normals[mask]
	offsets = offsets[mask]
	eqs = hull.equations[mask]
	return V, simplices, normals, offsets

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
