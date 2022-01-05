import numpy as np
import scipy
from tqdm import tqdm
from matplotlib import pyplot as plt, patches

from env import deep_sea_treasure
from common.lp import TabularSolver
from common.misc import deduplicate_and_sort, dummy_progress

# Chord algorithm
# "How good is the Chord algorithm?" https://arxiv.org/abs/1309.7084

def estimate_pareto_front(solver, epsilon, *, progress = dummy_progress):
	k = epsilon.shape[0]
	assert k == 2, "only works for 2D"
	with progress() as pbar:
		a = solver.solve_linear(np.array([1, 0], dtype = epsilon.dtype))
		pbar.update(1)
		b = solver.solve_linear(np.array([0, 1], dtype = epsilon.dtype))
		pbar.update(1)
		c = np.maximum(a, b)
		return np.array(chord(solver, a, b, c, epsilon, pbar))

def chord(solver, l, r, s, epsilon, pbar):
	normal = np.array([r[1] - l[1], l[0] - r[0]])
	p = (l + r) / 2
	
	ratio = ratio_distance(normal, p, s, epsilon, l, r)
	pbar.set_postfix({ 'ratio': ratio })
	if ratio <= 1:
		return [l, r]
	
	q = solver.solve_linear(normal)
	pbar.update(1)
	
	ratio = ratio_distance(normal, p, q, epsilon, l, r)
	pbar.set_postfix({ 'ratio': ratio })
	if ratio <= 1:
		return [l, r]
	
	sl = line_intersection(l, s - l, q, r - l)
	sr = line_intersection(r, s - r, q, r - l)
	ql = chord(solver, l, q, sl, epsilon, pbar)
	qr = chord(solver, q, r, sr, epsilon, pbar)
	
	return ql + qr

def ratio_distance(normal, p, q, epsilon, l, r):
	# Returns the largest `c` s.t. `q - c epsilon`
	# is on the line with normal `normal` that goes through `p`
	c = ((q - p) @ normal) / (epsilon @ normal)
	return c

def line_intersection(a, da, b, db):
	# Intersection of lines `A(t) = a + t da`, `B(t) = b + t db`
	M = np.stack([-da, db], axis = -1)
	t = np.linalg.solve(M, a - b)
	return a + t[0] * da

def main():
	env = deep_sea_treasure
	epsilon = 1 * np.array([1, 1], dtype = np.float64)
	
	solver = TabularSolver(env)
	true_pf = env.true_pareto_front()
	
	estimated_pf = estimate_pareto_front(solver, epsilon, progress = tqdm)
	estimated_pf = deduplicate_and_sort(estimated_pf)
	
	plt.ylabel("Discounted time penalty")
	plt.xlabel("Discounted treasure value")
	
	# The epsilon-boxes should touch the blue line
	ax = plt.gca()
	for point in true_pf:
		ax.add_patch(patches.Rectangle(point, -epsilon[0], -epsilon[1], facecolor = 'red', alpha = 0.1))
	
	plt.plot(estimated_pf[:,0], estimated_pf[:,1], 'g-')
	plt.plot(estimated_pf[:,0], estimated_pf[:,1], 'go', label = "Estimated PF (chord)")
	plt.plot(true_pf[:,0], true_pf[:,1], 'r+', label = "True PF")
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
