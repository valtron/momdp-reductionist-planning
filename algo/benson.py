import numpy as np

from common.misc import dummy_progress, close_to_any
from common import dualdesc

def estimate_pareto_front(solver, epsilon, *, progress = dummy_progress):
	k = epsilon.shape[0]
	
	IA = []
	OA = []
	verts_checked = []
	
	for w in np.eye(k):
		y = solver.solve_linear(w)
		IA.append(y)
		OA.append((w, y))
	
	with progress() as pbar:
		while True:
			found = None
			
			A = np.array([w for w, _ in OA], dtype = np.float64)
			b = np.sum(A * np.array([y for _, y in OA], dtype = np.float64), axis = 1)
			OAh = dualdesc.HRepr(A, b)
			for v in OAh.to_v().Vc:
				if close_to_any(v, verts_checked, 1e-6):
					continue
				verts_checked.append(v)
				u, w = solver.solve_chebyshev(v, np.ones(k))
				IA.append(u)
				d = np.max(np.abs(u - v) / epsilon)
				if d > 1:
					found = w, u
					break
			
			if found is None:
				break
			
			pbar.set_postfix({ 'd': d })
			
			OA.append(found)
			pbar.update(1)
	
	return np.array(IA)

def main():
	from tqdm import tqdm
	from matplotlib import pyplot as plt, patches
	from env import deep_sea_treasure
	from common.lp import TabularSolver
	from common.misc import deduplicate_and_sort
	
	# Finds an epsilon-Pareto front using Benson's algorithm and plots it.
	env = deep_sea_treasure
	epsilon = 1 * np.array([1, 1], dtype = np.float64)
	
	solver = TabularSolver(env)
	true_pf = env.true_pareto_front()
	
	estimated_pf = estimate_pareto_front(solver, epsilon, progress = tqdm)
	estimated_pf = deduplicate_and_sort(estimated_pf)
	
	plt.ylabel("Discounted time penalty")
	plt.xlabel("Discounted treasure value")
	
	# The epsilon-boxes should touch the estimated PF
	ax = plt.gca()
	for point in true_pf:
		ax.add_patch(patches.Rectangle(point, -epsilon[0], -epsilon[1], facecolor = 'red', alpha = 0.1))
	
	plt.plot(estimated_pf[:,0], estimated_pf[:,1], 'c-')
	plt.plot(estimated_pf[:,0], estimated_pf[:,1], 'co', label = "Estimated PF (Benson)")
	plt.plot(true_pf[:,0], true_pf[:,1], 'r+', label = "True PF")
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
