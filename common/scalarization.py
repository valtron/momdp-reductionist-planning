from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

def make_covering(n, m, space, rng):
	points = []
	while len(points) < n:
		samples = space.sample_vectors(rng, m)
		dists = cdist(samples, points or samples, metric = space.metric)
		i = np.argmax(np.min(dists, axis = 1))
		points.append(samples[i])
	return [space.make_scalarization(vec) for vec in points]

class SpherePlusSpace:
	def __init__(self, k):
		self.k = k
		self.metric = 'cosine'
	
	def sample_vectors(self, rng, n):
		vecs = rng.normal(size = (n, self.k))
		np.abs(vecs, out = vecs)
		vecs /= np.linalg.norm(vecs, axis = 1, keepdims = True)
		return vecs
	
	def make_scalarization(self, vec):
		return LinearScalarization(vec)

class OutcomeShadowSpace:
	def __init__(self, min_outcome, max_outcome):
		self.k = len(min_outcome)
		self.metric = 'chebyshev'
		self.min_outcome = min_outcome
		self.outcome_range = max_outcome - min_outcome
		
		tmp = np.product(self.outcome_range)
		tmp /= self.outcome_range
		tmp /= np.sum(tmp)
		self.probs = tmp
	
	def sample_vectors(self, rng, n):
		vecs = rng.uniform(size = (n, self.k)) * self.outcome_range
		sides = rng.choice(self.k, size = n, p = self.probs)
		for i in range(n):
			vecs[i, sides[i]] = 1
		vecs += self.min_outcome - np.mean(vecs, axis = 1, keepdims = True)
		return vecs
	
	def make_scalarization(self, vec):
		return ChebyshevScalarization(vec, None)

class LinearScalarization:
	def __init__(self, w: np.ndarray):
		self.w = w
	
	def __call__(self, outcome: np.ndarray):
		return outcome @ self.w
	
	def linearize_at(self, outcome: np.ndarray):
		return self.w
	
	def __repr__(self):
		return format(self, '')
	
	def __format__(self, spec):
		return 'lin([{}])'.format(','.join(
			format(x, spec) for x in self.w
		))

class ChebyshevScalarization:
	def __init__(self, r: Optional[np.ndarray], w: Optional[np.ndarray]):
		if w is not None:
			assert np.all(w >= 0)
			assert np.any(w > 0)
		
		self._r = r
		self._w = w
		
		if r is None:
			self.r = np.zeros_like(w)
		else:
			self.r = r
		
		if w is None:
			self.w = np.full_like(r, np.sqrt(len(r)))
		else:
			with np.errstate(divide = 'ignore'):
				self.w = 1 / w
	
	def __call__(self, outcome: np.ndarray):
		return np.min((outcome - self.r) * self.w)
	
	def linearize_at(self, outcome: np.ndarray):
		out = np.zeros_like(outcome)
		out[np.argmin((outcome - self.r) * self.w)] = 1
		return out
	
	def __repr__(self):
		return format(self, '')
	
	def __format__(self, spec):
		args = []
		if self._r is not None:
			args.append('r=[{}]'.format(','.join(
				format(x, spec) for x in self._r
			)))
		if self._w is not None:
			args.append('w=[{}]'.format(','.join(
				format(x, spec) for x in self._w
			)))
		return 'cheb({})'.format(','.join(args))
