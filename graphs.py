import warnings
import numpy as np
import itertools
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path

from env import deep_sea_treasure, bonus_world
from common import LPMDPSolver, CMDPSolver, make_linear_comb, deduplicate_and_sort, mdp_to_matrices, FlowPolytope
from algo import benson, chord_2d, nls, ols

ENVS = [
	("DeepSeaTreasure", deep_sea_treasure, [nls, chord_2d, ols, benson]),
	("BonusWorld", bonus_world, [nls, ols, benson]),
]

ALGOS = {
	chord_2d: ("Chord" , 'green', '-' ),
	nls     : ("NLS"   , 'blue' , '-' ),
	ols     : ("OLS"   , 'red'  , '--'),
	benson  : ("Benson", 'cyan' , '--'),
}

SHOW_FIGS = False

def main():
	outdir = Path('figs')
	outdir.mkdir(exist_ok = True)
	graph_queries_vs_epsilon(outdir)
	graph_hypervolume_vs_queries(outdir)
	graph_pareto_fronts(outdir)

def graph_queries_vs_epsilon(outdir):
	# make graphs showing how # queries grows as a function of epsilon
	# for each algorithm and on both environments
	epsilons = np.geomspace(3, 0.2, 8)
	
	for env_name, env_module, algo_modules in ENVS:
		if not algo_modules:
			continue
		
		true_pf = env_module.true_pareto_front()
		ref = np.min(true_pf, axis = 0)
		true_hv = hypervolume(ref, true_pf)
		
		data = {
			algo_module: sweep_epsilons(algo_module, env_module, epsilons)
			for algo_module in algo_modules
		}
		
		# queries vs. epsilon
		fig, ax = plt.subplots(figsize = (4, 3), constrained_layout = True)
		ax.set_title("# queries vs. epsilon")
		ax.set_xlabel("1/epsilon")
		ax.set_ylabel("# queries")
		ax.set_yscale('log')
		for algo_module in algo_modules:
			algo_name, color, ls = ALGOS[algo_module]
			d = data[algo_module]
			ax.plot(1/d[:, 0], d[:, 1], linestyle = ls, color = color, label = algo_name)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.legend()
		if SHOW_FIGS:
			plt.show()
		else:
			plt.savefig(outdir / 'queries-{}.svg'.format(env_name))

def graph_hypervolume_vs_queries(outdir):
	# make graphs showing how hypervolume grows as a function of queries
	# for each algorithm and on both environments
	
	# pick epsilons for each env/algo so that the range of queries
	# is roughly the same for all
	epsilons = {
		(deep_sea_treasure, nls     ): np.geomspace( 12, 1.5 , 8),
		(deep_sea_treasure, chord_2d): np.geomspace(  1, 0.01, 8),
		(deep_sea_treasure, ols     ): np.geomspace(  1, 0.01, 8),
		(deep_sea_treasure, benson  ): np.geomspace(100, 0.2 , 8),
		(bonus_world      , nls     ): np.geomspace( 15, 1.5 , 8),
		(bonus_world      , ols     ): np.geomspace(  1, 0.1 , 8),
		(bonus_world      , benson  ): np.geomspace( 70, 5   , 8),
	}
	
	for env_name, env_module, algo_modules in ENVS:
		if not algo_modules:
			continue
		
		true_pf = env_module.true_pareto_front()
		ref = np.min(true_pf, axis = 0)
		true_hv = hypervolume(ref, true_pf)
		
		data = {
			algo_module: sweep_epsilons(algo_module, env_module, epsilons[env_module, algo_module])
			for algo_module in algo_modules
		}
		
		# hypervolume vs. queries
		max_num_queries = max(d[-1, 1] for _, d in data.items())
		fig, ax = plt.subplots(figsize = (4, 3), constrained_layout = True)
		ax.set_title("Hypervolume vs. # queries")
		ax.set_xlabel("# queries")
		ax.set_ylabel("Hypervolume")
		ax.axhline(true_hv, color = 'black', label = "True hypervolume")
		for algo_module in algo_modules:
			algo_name, color, ls = ALGOS[algo_module]
			d = data[algo_module]
			if d[-1, 1] < max_num_queries:
				d = np.concatenate([d, [[0, max_num_queries, d[-1, 2]]]], axis = 0)
			ax.plot(d[:, 1], d[:, 2], linestyle = ls, color = color, label = algo_name)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.legend()
		
		if env_module is deep_sea_treasure:
			ax.set_ylim([160, 172])
			ax.set_xlim([5, 20])
		else:
			ax.set_ylim([220, 257])
			ax.set_xlim([10, 45])
		
		if SHOW_FIGS:
			plt.show()
		else:
			plt.savefig(outdir / 'hv-{}.svg'.format(env_name))

def sweep_epsilons(algo_module, env_module, epsilons):
	true_pf = env_module.true_pareto_front()
	ref = np.min(true_pf, axis = 0)
	
	data = []
	for epsilon in tqdm(epsilons):
		estimated_pf, num_queries = estimate_pf(env_module, algo_module, epsilon)
		hv = hypervolume(ref, estimated_pf)
		data.append([epsilon, num_queries, hv])
	return np.array(data)

def graph_pareto_fronts(outdir):
	# for a given epsilon, show the e-PF found by each algorithm on DST
	
	epsilon = 2
	true_pf = deep_sea_treasure.true_pareto_front()
	y_min = np.min(true_pf[:,1])
	true_pf_lower = true_pf - epsilon
	
	for algo_module, (algo_name, color, linestyle) in ALGOS.items():
		estimated_pf, num_queries = estimate_pf(deep_sea_treasure, algo_module, epsilon)
		
		fig, ax = plt.subplots(figsize = (3, 2.3), constrained_layout = True)
		ax.fill(
			np.concatenate([true_pf[:,0], true_pf_lower[::-1,0]]),
			np.concatenate([true_pf[:,1], true_pf_lower[::-1,1]]),
			color = 'black', alpha = 0.1, linewidth = 0,
		)
		ax.plot(true_pf[:,0], true_pf[:,1], 'k-', linewidth = 1, label = "Pareto front")
		ax.fill_between(estimated_pf[:,0], y_min, estimated_pf[:,1], color = color, alpha = 0.1)
		ax.plot(estimated_pf[:,0], estimated_pf[:,1], linewidth = 1, linestyle = linestyle, color = color, label = algo_name)
		ax.set_xlabel("Objective 1")
		ax.set_xlim([np.min(true_pf[:,0]), np.max(true_pf[:,0])])
		ax.spines['right'].set_visible(False)
		ax.set_ylabel("Objective 2")
		ax.set_ylim([np.min(true_pf[:,1]), np.max(true_pf[:,1])])
		ax.spines['top'].set_visible(False)
		
		if SHOW_FIGS:
			plt.show()
		else:
			plt.savefig(outdir / 'pf-{}.svg'.format(algo_name))

def estimate_pf(env_module, algo_module, epsilon):
	transitions, rewards, start_distribution = env_module.get_mdp()
	true_pf = env_module.true_pareto_front()
	gamma = env_module.gamma
	k = rewards.shape[1]
	H = 1/(1 - gamma)
	
	if algo_module is benson:
		fp = FlowPolytope(transitions, start_distribution, gamma)
		estimated_pf, _, num_queries = algo_module.estimate_pareto_front(rewards.T, fp.A, fp.b, eps = epsilon/H)
		estimated_pf *= H
		estimated_pf = np.clip(estimated_pf, np.min(true_pf, axis = 0), np.max(true_pf, axis = 0))
		# `k` queries are to find the anti-ideal point, which we wouldn't
		# need if we assume [0, 1]-bounded rewards
		num_queries -= k + 1
	else:
		comb = make_linear_comb(LPMDPSolver(transitions, start_distribution, gamma), rewards, gamma)
		num_queries = 0
		def counting_comb(w):
			nonlocal num_queries
			num_queries += 1
			return comb(w)
		estimated_pf = algo_module.estimate_pareto_front(counting_comb, epsilon * np.ones(k))
		if algo_module is nls:
			# `2*k` queries are to figure out the range, but aren't necessary
			# (could also be done by using the reward vector directly).
			num_queries -= 2 * k
	
	return deduplicate_and_sort(estimated_pf), num_queries

def hypervolume(ref, points):
	hull = estimated_pf_hull(ref, points)
	return hull.volume

def estimated_pf_hull(ref, points):
	assert np.all(points >= (ref - 1e-5)), np.min(points - ref)
	points = np.maximum(points, ref)
	
	all_points = [[ref], points]
	
	# Project each point onto each (1 .. k-1)-dimensional boundary
	k = ref.shape[0]
	idxs = list(range(k))
	for i in range(1, k):
		for boundary in itertools.combinations(idxs, i):
			boundary = list(boundary)
			projected = points.copy()
			projected[:, boundary] = ref[boundary]
			all_points.append(projected)
	
	return ConvexHull(np.concatenate(all_points, axis = 0))

if __name__ == '__main__':
	main()
