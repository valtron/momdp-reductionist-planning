from pathlib import Path

import matplotlib.pyplot as plt
from joblib import delayed

from algo.hypervolume import *
from common import mdp_to_matrices, FlowPolytope, ProgressParallel
from env import deep_sea_treasure, bonus_world
from graphs import hypervolume

experiments = [
	{
		'env': deep_sea_treasure,
		'env_name': "DeepSeaTreasure",
		'runs': 12,
		'seed_init': 0,
		'schedule': np.linspace(5, 200, 10, dtype = np.int32),
	},
	{
		'env': bonus_world,
		'env_name': "BonusWorld",
		'runs': 12,
		'seed_init': 100,
		'schedule': np.linspace(5, 200, 10, dtype = np.int32),
	},
]

NUM_WORKERS = 4
SHOW_FIGS = False

def main():
	outdir = Path('fig')
	outdir.mkdir(exist_ok = True)
	
	for exp in experiments:
		true_pf = exp['env'].true_pareto_front()
		anti_utopia = np.min(true_pf, axis = 0) - 1e-3
		true_hv = hypervolume(anti_utopia, true_pf)
		
		gamma = exp['env'].Env.gamma
		H = 1 / (1 - gamma)
		transitions, rewards, start_distribution = mdp_to_matrices(exp['env'].Env)
		ref = np.min(rewards, axis = 0)
		fp = FlowPolytope(transitions, start_distribution, gamma)
		
		schedule = exp['schedule']
		
		def do_run(run):
			rng = np.random.default_rng(exp['seed_init'] + run)
			
			run_data = []
			for n_scalarizations in schedule:
				lambs = get_scalarizations(rng, n_scalarizations, rewards.shape[1])
				
				decision_points = [
					solve_scalarization(rewards.T, fp.A, fp.b, ref, lamb)
					for lamb in lambs
				]
				objective_points = np.array(decision_points) @ rewards * H
				
				run_data.append(hypervolume(anti_utopia, objective_points))
			
			return run_data
		
		data = ProgressParallel(n_jobs = NUM_WORKERS)(
			delayed(do_run)(i) for i in range(exp['runs'])
		)
		
		means = np.mean(data, axis = 0)
		stds = np.std(data, axis = 0)
		
		# hypervolume vs. scalarizations
		fig, ax = plt.subplots(figsize = (4, 3), constrained_layout = True)
		ax.set_title(exp['env_name'])
		ax.set_xlabel("# Scalarizations")
		ax.set_ylabel("Hypervolume")
		ax.axhline(true_hv, color = 'black')
		ax.errorbar(schedule, means, stds, linestyle = '-')
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		
		if SHOW_FIGS:
			plt.show()
		else:
			fig.savefig(outdir / 'hv-scalarizations-{}-{}runs.pdf'.format(
				exp['env_name'], exp['runs'],
			))

if __name__ == "__main__":
	main()
