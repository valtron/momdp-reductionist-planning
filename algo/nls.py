import itertools
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt, patches

from env import deep_sea_treasure
from common import (
	LPMDPSolver, deduplicate_and_sort, get_outcome_space_bbox,
	make_linear_comb, dummy_progress, mdp_to_matrices,
)

def estimate_pareto_front(comb, epsilon, *, progress = dummy_progress):
	# Estimates the Pareto front using linear scalarizations.
	k = epsilon.shape[0]
	
	# Get a bounding box on PF
	min_outcomes, max_outcomes = get_outcome_space_bbox(comb, k)
	ranges = max_outcomes - min_outcomes
	
	# `epsilon` is a vector, and can be non-uniform (different values for each dimension).
	# To handle that, first divide the outcomes by `epsilon` component-wise
	# (this is also done when calling `comb`, see lower down.)
	# so that, in the rescaled outcomes, the (new) epsilon is uniformly 1.
	# Then, uniformly scale the outcomes and new epsilon so that outcomes
	# fall in [0, 1]^k. This actually only needs to be done for the epsilon
	# because uniform scalings of the outcomes don't affect `comb`.
	# (As a sanity check, note that if `epsilon` is uniform, this simplifies
	# to `uniform_epsilon = epsilon / max(ranges)`.)
	uniform_epsilon = 1/np.max(ranges / epsilon)
	# Inflation factor from the proof
	# TODO: Could calculate a smaller value using `ranges`
	uniform_epsilon /= (k - 1)/np.sqrt(k)
	
	estimated_pf = []
	
	for w in progress(iter_weights(k, uniform_epsilon)):
		# Find optimal policy for linear scalarization `w`
		# (As mentioned previously, the `1/epsilon` is to handle non-uniform epsilon.)
		outcome = comb(w / epsilon)
		estimated_pf.append(outcome)
	
	estimated_pf = np.array(estimated_pf)
	return estimated_pf

class iter_weights:
	# Iterates over an epsilon-net of the positive part of a k-sphere
	
	def __init__(self, k, epsilon):
		self.k = k
		self.theta_max = np.arctan(np.sqrt(k - 1))
		self.m = int(np.ceil(self.theta_max / 2 * 1/epsilon))
	
	def __iter__(self):
		# Divide the hypercube into 2^k segments. Each segment is describe by whether
		# it falls on the surface of the hypercube along each dimension.
		# E.g. the segment `(0, 1)` is the top surface of a square; `(1, 1)` is the top-right corner.
		segments = itertools.product([0, 1], repeat = self.k)
		points = np.tan(self.theta_max * np.linspace(0, 1, self.m + 1))/np.tan(self.theta_max)
		
		ranges = [points[:-1], points[-1:]]
		# Skip the "interior" of the cube (segment represented by `{0}^k`)
		_ = next(segments)
		for segment in segments:
			yield from itertools.product(*(ranges[s] for s in segment))
	
	def __len__(self):
		return int(np.power(self.m + 1, self.k) - np.power(self.m, self.k))

def main():
	# Finds an epsilon-Pareto front using linear scalarizations and plots it.
	
	epsilon = 1 * np.array([1, 1], dtype = np.float64)
	
	gamma = deep_sea_treasure.Env.gamma
	transitions, rewards, start_distribution = mdp_to_matrices(deep_sea_treasure.Env)
	mdp_solver = LPMDPSolver(transitions, start_distribution, gamma)
	comb = make_linear_comb(mdp_solver, rewards, gamma)
	true_pf = deep_sea_treasure.true_pareto_front()
	
	estimated_pf = estimate_pareto_front(comb, epsilon, progress = tqdm)
	estimated_pf = deduplicate_and_sort(estimated_pf)
	
	plt.ylabel("Discounted time penalty")
	plt.xlabel("Discounted treasure value")
	
	# The epsilon-boxes should touch the estimated PF
	ax = plt.gca()
	for point in true_pf:
		ax.add_patch(patches.Rectangle(point, -epsilon[0], -epsilon[1], facecolor = 'red', alpha = 0.1))
	
	plt.plot(estimated_pf[:,0], estimated_pf[:,1], 'b-')
	plt.plot(estimated_pf[:,0], estimated_pf[:,1], 'bo', label = "Estimated PF (NLS)")
	plt.plot(true_pf[:,0], true_pf[:,1], 'r+', label = "True PF")
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
