import heapq
import itertools
import numpy as np
import scipy
from tqdm import tqdm
from matplotlib import pyplot as plt, patches

from env import deep_sea_treasure
from common import LPMDPSolver, deduplicate_and_sort, make_linear_comb, dummy_progress

def estimate_pareto_front(comb, epsilon, *, progress = dummy_progress):
	# Estimates the Pareto front using Optimistic Linear Support
	# http://roijers.info/pub/thesis.pdf
	
	k = epsilon.shape[0]
	epsilon = np.min(epsilon)
	
	S = []
	W = []
	
	with progress() as pbar:
		for w in np.eye(k):
			S.append(comb(w))
			pbar.update(1)
			W.append(w)
		
		w = normal(np.array(S))
		w /= np.sum(w)
		assert np.min(w) >= 0
		
		Q = [QueueItem(np.inf, w, maxValue(S, w))]
		
		while Q:
			Q = sorted(Q)
			item = Q.pop()
			pbar.set_postfix({ 'delta': item.priority })
			
			# (Near-)duplicate weights causes issues for `maxValueLP`
			if any(np.allclose(item.weight, w) for w in W):
				continue
			
			outcome = comb(item.weight)
			pbar.update(1)
			W.append(item.weight)
			
			if any(np.allclose(outcome, s) for s in S):
				continue
			
			# Remove obsolete weights from Q, put them in Wdel
			Wdel = [item.weight]
			Qkeep = []
			for other_item in Q:
				if other_item.weight @ outcome > other_item.min_value:
					Wdel.append(other_item.weight)
				else:
					Qkeep.append(other_item)
			Q = Qkeep
			
			ws = newCornerWeights(outcome, Wdel, S)
			
			S.append(outcome)
			
			deltas = []
			for w in ws:
				w_value = maxValue(S, w)
				delta = maxValueLP(w, S, W) - w_value
				deltas.append(delta)
				if delta > epsilon:
					Q.append(QueueItem(delta, w, w_value))
	
	return np.array(S)

def newCornerWeights(outcome, Wdel, S):
	Wnew = []
	
	k = len(outcome)
	fudge = 1e-8
	
	# In the paper, they iterate over all `k-1` combinations of the outcomes
	# of all `w in Wdel` (`inter_outcome_combos = True`). I can't figure out
	# if this is needed, but it seems to me that any new corner weight would
	# be formed from `k-1` outcomes of a deleted corner weight and `outcome`.
	inter_outcome_combos = False
	
	if inter_outcome_combos:
		def iter_combos():
			Vrel = list(itertools.chain(*(argmaxValues(S, w) for w in Wdel)))
			for i in range(0, k): # [0, .., k-1]
				yield from itertools.combinations(Vrel, i)
	else:
		def iter_combos():
			for w in Wdel:
				outcomes = argmaxValues(S, w)
				for i in range(0, k): # [0, .., k-1]
					yield from itertools.combinations(outcomes, i)
	
	for X in iter_combos():
		for wc in cornerWeightsForOutcomes(k, [outcome, *X]):
			if wc @ outcome >= maxValue(S, wc) - fudge:
				Wnew.append(wc)
	
	return Wnew

def cornerWeightsForOutcomes(k, outcomes):
	idxs = list(range(k))
	outcomes = np.array(outcomes)
	
	for boundary in itertools.combinations(idxs, len(outcomes)):
		# `boundary` represents a boundary of the weight simplex,
		# as a list of the indices of `w` that are not 0
		boundary = list(boundary)
		w = np.zeros(k)
		
		outcomes_boundary = outcomes[:, boundary]
		ns = scipy.linalg.null_space(outcomes_boundary - outcomes_boundary[0])
		assert ns.shape[1] != 0
		
		if ns.shape[1] > 1:
			continue
		
		w[boundary] = ns[:,0]
		if np.min(w) < 0:
			if np.max(w) > 0:
				# Intersection outside the weight simplex
				continue
			w *= -1
		
		w /= np.sum(w)
		yield w

def maxValueLP(w, S, W):
	# max_v w.v
	# s.t. W[i].v <= V_S*(W[i])
	values = np.array([maxValue(S, w) for w in W])
	res = scipy.optimize.linprog(-w, np.array(W), values, bounds = (None, None))
	assert res.success, res.message
	return w @ res.x

def argmaxValues(S, w):
	# V_S(w) (defn. 18, p. 38)
	fudge = 1e-8
	mx = maxValue(S, w)
	return [
		outcome for outcome in S
		if w @ outcome >= mx - fudge
	]

def maxValue(S, w):
	# V_S*(w) (defn. 17, p. 38)
	return max(w @ outcome for outcome in S)

def normal(points):
	ns = scipy.linalg.null_space(points - points[0])
	assert ns.shape[1] == 1
	return ns[:,0]

class QueueItem:
	def __init__(self, priority, weight, min_value):
		self.priority = priority
		self.weight = weight
		self.min_value = min_value
	
	def __lt__(self, other):
		return self.priority < other.priority

def main():
	test_dst()
	test_sphere()

def test_sphere():
	def comb(w):
		w = np.maximum(w, 0)
		return w / np.linalg.norm(w)
	
	epsilon = 0.1 * np.array([1, 1, 1], dtype = np.float64)
	estimated_pf = estimate_pareto_front(comb, epsilon, progress = tqdm)
	
	fig = plt.figure()
	ax = fig.add_subplot(projection = '3d')
	ax.plot(estimated_pf[:,0], estimated_pf[:,1], estimated_pf[:,2], 'ro')
	plt.show()

def test_dst():
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
