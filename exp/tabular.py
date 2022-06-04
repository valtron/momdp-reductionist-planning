import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path

from env import deep_sea_treasure, bonus_world
from common.lp import TabularSolver
from common.hypervolume import hypervolume_convex
from common import pareto, misc
from algo import benson, chord, nls, ols

ENVS = [
	("DeepSeaTreasure", deep_sea_treasure, [nls, ols, benson, chord]),
	("BonusWorld", bonus_world, [nls, ols, benson]),
]

ALGOS = {
	chord : ("Chord" , 'green', '--'),
	nls   : ("NLS"   , 'blue' , '-' ),
	ols   : ("OLS"   , 'red'  , '-' ),
	benson: ("Benson", 'cyan' , '-' ),
}

SHOW_FIGS = False

def main():
	outdir = Path('fig')
	outdir.mkdir(exist_ok = True)
	graph_queries_vs_epsilon(outdir)
	graph_hypervolume_vs_queries(outdir)
	graph_pareto_fronts(outdir)

def graph_queries_vs_epsilon(outdir):
	# make graphs showing how # queries grows as a function of epsilon
	# for each algorithm and on both environments
	epsilons = np.geomspace(3, 0.2, 8)
	
	for env_name, env, algo_modules in ENVS:
		if not algo_modules:
			continue
		
		true_pf = env.true_pareto_front()
		
		data = {
			algo_module: sweep_epsilons(algo_module, env, epsilons)
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
		if env is deep_sea_treasure:
			ax.set_ylim([2, 600])
		else:
			ax.set_ylim([2, 8000])
		if SHOW_FIGS:
			plt.show()
		else:
			fig.savefig(outdir / 'queries-{}.pdf'.format(env_name))

def graph_hypervolume_vs_queries(outdir):
	# make graphs showing how hypervolume grows as a function of queries
	# for each algorithm and on both environments
	
	# pick epsilons for each env/algo so that the range of queries
	# is roughly the same for all
	epsilons = {
		(deep_sea_treasure, nls   ): np.geomspace(15, 1.5, 10),
		(deep_sea_treasure, chord ): np.geomspace( 5, 0.1, 10),
		(deep_sea_treasure, ols   ): np.geomspace( 5, 0.1, 10),
		(deep_sea_treasure, benson): np.geomspace( 8, 0.2, 10),
		(bonus_world      , nls   ): np.geomspace(11, 1.5, 10),
		(bonus_world      , ols   ): np.geomspace( 4, 0.1, 10),
		(bonus_world      , benson): np.geomspace( 4, 0.1, 10),
	}
	
	for env_name, env, algo_modules in ENVS:
		if not algo_modules:
			continue
		
		true_pf = env.true_pareto_front()
		ref = env.min_return
		true_hv = hypervolume_convex(true_pf, ref)
		
		data = {
			algo_module: sweep_epsilons(algo_module, env, epsilons[env, algo_module])
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
		
		if env is deep_sea_treasure:
			ax.set_ylim([720, 733])
			ax.set_xlim([0, 20])
			ax.set_xticks([0, 5, 10, 15, 20])
		else:
			ax.set_ylim([910, 940])
			ax.set_xlim([0, 100])
		
		if SHOW_FIGS:
			plt.show()
		else:
			fig.savefig(outdir / 'hv-{}.pdf'.format(env_name))

def sweep_epsilons(algo_module, env, epsilons):
	true_pf = env.true_pareto_front()
	ref = env.min_return
	
	data = []
	for epsilon in tqdm(epsilons):
		estimated_pf, num_queries = estimate_pf(env, algo_module, epsilon)
		hv = hypervolume_convex(estimated_pf, ref)
		data.append([epsilon, num_queries, hv])
	return np.array(data)

def graph_pareto_fronts(outdir):
	# for a given epsilon, show the e-PF found by each algorithm on DST
	
	epsilon = 1.8
	true_pf = deep_sea_treasure.true_pareto_front()
	y_min = np.min(true_pf[:,1])
	true_pf_lower = true_pf - epsilon
	
	for algo_module, (algo_name, color, linestyle) in ALGOS.items():
		estimated_pf, num_queries = estimate_pf(deep_sea_treasure, algo_module, epsilon)
		estimated_pf = estimated_pf[pareto.is_non_dominated(estimated_pf)]
		print(algo_name, len(estimated_pf) - 1)
		
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
			fig.savefig(outdir / 'pf-{}.pdf'.format(algo_name))

def estimate_pf(env, algo_module, epsilon):
	solver = CountSolverCalls(TabularSolver(env))
	estimated_pf = algo_module.estimate_pareto_front(solver, epsilon * np.ones(env.k))
	calls = solver.calls
	if algo_module is nls:
		# NLS queries twice at the basis vectors.
		# Those queries could be cached, but I'm too lazy, so I'll just not count them.
		calls -= env.k
	return misc.deduplicate_and_sort(estimated_pf), calls

class CountSolverCalls:
	def __init__(self, impl):
		self.impl = impl
		self.calls = 0
	
	def solve_linear(self, w):
		self.calls += 1
		return self.impl.solve_linear(w)
	
	def solve_chebyshev(self, r, w):
		self.calls += 1
		return self.impl.solve_chebyshev(r, w)

if __name__ == '__main__':
	main()
