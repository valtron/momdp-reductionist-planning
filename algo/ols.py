from typing import Tuple
import numpy as np

from common import dualdesc
from common.misc import dummy_progress, close_to_any

def estimate_pareto_front(solver, epsilon, *, progress = dummy_progress):
	# Estimates the Pareto front using Optimistic Linear Support
	# http://roijers.info/pub/thesis.pdf
	
	k = epsilon.shape[0]
	impl = OLS(None, epsilon)
	
	with progress() as pbar:
		while True:
			wt, dt = impl.get_next_weight()
			pbar.set_postfix({ 'd': dt })
			if dt <= 1:
				break
			if close_to_any(wt, impl.W, 1e-6):
				break
			yt = solver.solve_linear(wt)
			pbar.update(1)
			impl.W.append(wt)
			impl.Y.append(yt)
	
	return np.array(impl.Y, dtype = np.float64)

class OLS:
	def __init__(self, env, metric, *, approx = False):
		self.env = env
		self.k = metric.shape[0]
		self.metric = metric
		self.W = []
		self.Y = []
		self.approx = approx
	
	def get_next_weight(self):
		if len(self.W) < self.k:
			return np.eye(self.k)[len(self.W)], np.inf
		
		IAv, OAh = self._get_IA_OA()
		# H-repr of IA
		A, b = IAv.to_h().to_inequalities()
		# Vertices of OA
		V = OAh.to_v().Vc
		
		# Find Hausdorff distance to IA of each OA vertex
		with np.errstate(divide = 'ignore'):
			d = np.max((V @ A.T - b) / (A @ self.metric), axis = 1)
		
		# Get OA vertex furthest from IA
		i = np.argmax(d)
		# project vertex of OA on IA
		u = V[i] - d[i] * self.metric
		# find a supporting hyperplane of u
		w = np.clip(A[np.argmax(A @ u - b)], 0, np.inf)
		w = w / np.linalg.norm(w)
		
		return w, d[i]
	
	def _get_IA_OA(self) -> Tuple[dualdesc.VRepr, dualdesc.HRepr]:
		W = np.array(self.W, dtype = np.float64)
		Y = np.array(self.Y, dtype = np.float64)
		
		OAw = W
		if self.approx:
			# Calculate the smallest relative error `rel_err` s.t.
			# `yi.wi >= y.wi - rel_err * (ymax - ymin).wi` for some `y in Y`
			numer = np.sum((Y[:,:,None] - Y.T) * W.T, axis = 1)
			denom = W @ (self.env.max_return - self.env.min_return)
			rel_err = np.max(np.max(numer, axis = 0) / denom)
			abs_err = rel_err * denom
			
			# Replace outcomes that were suboptimal for their scalarizations
			Y = Y[np.argmax(Y @ W.T, axis = 0)]
			# Inflate OA vertices by the estimated error
			OAy = Y + abs_err[:,None] * W
		else:
			OAy = Y
		
		if self.approx:
			# Intersect outer approximation with upper bound of outcome space
			OAw = np.concatenate([OAw, np.eye(self.k)], axis = 0)
			OAy = np.concatenate([OAy, np.diag(self.env.max_return)], axis = 0)
		
		# Inner approximation
		IAv = dualdesc.VRepr(Y, -np.eye(self.k))
		# Outer approximation
		OAh = dualdesc.HRepr(OAw, np.sum(OAw * OAy, axis = 1))
		
		return IAv, OAh

def main():
	test_dst()
	test_sphere()

def test_sphere():
	from tqdm import tqdm
	from matplotlib import pyplot as plt
	
	class Solver:
		def solve_linear(self, w):
			w = np.maximum(w, 0)
			return w / np.linalg.norm(w)
	
	epsilon = 0.1 * np.array([1, 1, 1], dtype = np.float64)
	estimated_pf = estimate_pareto_front(Solver(), epsilon, progress = tqdm)
	
	fig = plt.figure()
	ax = fig.add_subplot(projection = '3d')
	ax.plot(estimated_pf[:,0], estimated_pf[:,1], estimated_pf[:,2], 'ro')
	plt.show()

def test_dst():
	from tqdm import tqdm
	from matplotlib import pyplot as plt, patches
	
	from env import deep_sea_treasure
	from common.lp import TabularSolver
	from common.misc import deduplicate_and_sort
	
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
	
	plt.plot(estimated_pf[:,0], estimated_pf[:,1], 'r--')
	plt.plot(estimated_pf[:,0], estimated_pf[:,1], 'ro', label = "Estimated PF (OLS)")
	plt.plot(true_pf[:,0], true_pf[:,1], 'r+', label = "True PF")
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
