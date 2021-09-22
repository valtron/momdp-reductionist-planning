import numpy as np
from scipy.optimize import linprog

def solve_scalarization(C, A, b, ref, lamb):
	"""
		Parameters
		--------------------
		C: The objective matrix. Each row is is one objective vector.
		A: The constraint matrix.
		b: The constraint vector.
	"""
	
	A_eq = np.column_stack((np.zeros(A.shape[0]), A))
	A_ub = np.column_stack((lamb, -C))
	c = np.zeros(1 + A.shape[1])
	c[0] = 1
	res = linprog(-c, A_eq = A_eq, A_ub = A_ub, b_eq = b, b_ub = -ref)
	assert res.success, res.message
	return res.x[1:]

def get_scalarizations(rng, n_points, dim):
	"""
		Get n_points sampled uniformly from the part of the (dim - 1)-sphere
		contained in the positive orthant.
	"""
	points = np.abs(rng.normal(size = (n_points, dim)))
	points = points / np.linalg.norm(points, axis = -1)[:, None]
	return points

def main():
	from matplotlib import pyplot as plt, patches
	from tqdm import tqdm
	from common import FlowPolytope, deduplicate_and_sort, mdp_to_matrices
	from env import deep_sea_treasure
	
	epsilon = 1 * np.array([1, 1], dtype = np.float64)
	
	gamma = deep_sea_treasure.Env.gamma
	H = 1 / (1 - gamma)
	transitions, rewards, start_distribution = mdp_to_matrices(deep_sea_treasure.Env)
	ref = np.min(rewards, axis = 0)
	fp = FlowPolytope(transitions, start_distribution, gamma)
	true_pf = deep_sea_treasure.true_pareto_front()
	
	rng = np.random.default_rng(0)
	lambs = get_scalarizations(rng, 50, rewards.shape[1])
	estimated_pf = []
	for lamb in tqdm(lambs):
		mu = solve_scalarization(rewards.T, fp.A, fp.b, ref, lamb)
		estimated_pf.append(mu @ rewards * H)
	estimated_pf = deduplicate_and_sort(estimated_pf)
	
	plt.ylabel("Discounted time penalty")
	plt.xlabel("Discounted treasure value")
	
	ax = plt.gca()
	for point in true_pf:
		ax.add_patch(patches.Rectangle(point, -epsilon[0], -epsilon[1], facecolor = 'red', alpha = 0.1))
	
	plt.plot(estimated_pf[:,0], estimated_pf[:,1], 'k-')
	plt.plot(estimated_pf[:,0], estimated_pf[:,1], 'ko', label = "Estimated PF (RHVS)")
	plt.plot(true_pf[:,0], true_pf[:,1], 'r+', label = "True PF")
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
