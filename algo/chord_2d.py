import heapq
import numpy as np
import scipy
from tqdm import tqdm
from matplotlib import pyplot as plt, patches

from env import deep_sea_treasure
from common import LPMDPSolver, deduplicate_and_sort, make_linear_comb, dummy_progress

# 2D Chord algorithm
# "How good is the Chord algorithm?" https://arxiv.org/abs/1309.7084

def estimate_pareto_front(comb, epsilon, *, progress = dummy_progress):
	k = epsilon.shape[0]
	assert k == 2, "only works for 2D"
	with progress() as pbar:
		a = comb([1, 0])
		pbar.update(1)
		b = comb([0, 1])
		pbar.update(1)
		c = np.maximum(a, b)
		return np.array(chord(comb, a, b, c, epsilon, pbar))

def chord(comb, l, r, s, epsilon, pbar):
	normal = np.array([r[1] - l[1], l[0] - r[0]])
	p = (l + r) / 2
	
	ratio = ratio_distance(normal, p, s, epsilon, l, r)
	pbar.set_postfix({ 'ratio': ratio })
	if ratio <= 1:
		return [l, r]
	
	q = comb(normal)
	pbar.update(1)
	
	ratio = ratio_distance(normal, p, q, epsilon, l, r)
	pbar.set_postfix({ 'ratio': ratio })
	if ratio <= 1:
		return [l, q, r]
	
	sl = line_intersection(l, s - l, q, r - l)
	sr = line_intersection(r, s - r, q, r - l)
	ql = chord(comb, l, q, sl, epsilon, pbar)
	qr = chord(comb, q, r, sr, epsilon, pbar)
	
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
	epsilon = 1 * np.array([1, 1], dtype = np.float64)
	
	gamma = 0.98
	transitions, rewards, start_distribution = deep_sea_treasure.get_mdp()
	mdp_solver = LPMDPSolver(transitions, start_distribution, gamma)
	comb = make_linear_comb(mdp_solver, rewards, gamma)
	true_pf = deep_sea_treasure.true_pareto_front()
	
	estimated_pf = estimate_pareto_front(comb, epsilon, progress = tqdm)
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
