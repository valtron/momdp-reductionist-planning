from typing import Tuple
import numpy as np
import dualdesc as dd
from common import scalarization as scal

name = "OLS"
color = 'r'
deterministic = True
inner_approximation = True

class Algo:
	def __init__(self, lo, hi, *, rng):
		self.impl = _OLS(lo, hi)
	
	@property
	def approx_poly(self):
		return self.impl.IA
	
	def update(self, solver):
		wt, dt, _ = self.impl.get_next_query()
		yt = solver.solve_linear(scal.Linear(wt))
		self.impl.update(wt, yt)
		return dt <= 1e-6

class _OLS:
	def __init__(self, lo, hi):
		self.lo = lo
		self.hi = hi
		k = len(lo)
		self.IA = dd.Polytope.FromGenerators(lo[None,:], -np.eye(k))
		self.OA = dd.Polytope.FromGenerators(hi[None,:], -np.eye(k))
		self.initial_done = False
		self.approx = False
	
	def get_next_query(self):
		# H-repr of IA
		A, b = self.IA.get_inequalities()
		# Vertices of OA
		V, _ = self.OA.get_generators()
		
		# Find Hausdorff distance to IA of each OA vertex
		with np.errstate(divide = 'ignore'):
			d = np.max((V @ A.T - b) / np.sum(A, axis = 1), axis = 1)
		
		# Get OA vertex furthest from IA
		i = np.argmax(d)
		
		if self.initial_done:
			# project vertex of OA on IA
			u = V[i] - d[i]
			# find a supporting hyperplane of u
			w = np.clip(A[np.argmax(A @ u - b)], 0, np.inf)
		else:
			w = np.ones_like(self.lo)
		w = w / np.sum(w)
		
		return w, d[i], V[i]
	
	def update(self, w, y):
		self.IA.add_point(y)
		self.OA.add_halfspace(w, w @ y)
		self.initial_done = True
	
	def _get_IA_OA(self) -> Tuple[dd.Polytope, dd.Polytope]:
		k = len(self.lo)
		
		if len(self.W) == 0:
			W = np.empty((0, k), dtype = np.float64)
			Y = np.empty((0, k), dtype = np.float64)
		else:
			W = np.array(self.W, dtype = np.float64)
			Y = np.array(self.Y, dtype = np.float64)
		
		OAw = W
		if self.approx:
			# Calculate the smallest relative error `rel_err` s.t.
			# `yi.wi >= y.wi - rel_err * (ymax - ymin).wi` for some `y in Y`
			numer = np.sum((Y[:,:,None] - Y.T) * W.T, axis = 1)
			denom = W @ (self.hi - self.lo)
			rel_err = np.max(np.max(numer, axis = 0) / denom)
			abs_err = rel_err * denom
			
			# Replace outcomes that were suboptimal for their scalarizations
			Y = Y[np.argmax(Y @ W.T, axis = 0)]
			# Inflate OA vertices by the estimated error
			OAy = Y + abs_err[:,None] * W
		else:
			OAy = Y
		
		# Intersect outer approximation with upper bound of outcome space
		OAw = np.concatenate([OAw, np.eye(k)], axis = 0)
		OAy = np.concatenate([OAy, np.diag(self.hi)], axis = 0)
		
		# Inner approximation
		if len(Y) == 0:
			IA = dd.Polytope.FromGenerators(self.lo[None,:], -np.eye(k))
		else:
			IA = dd.Polytope.FromGenerators(Y, -np.eye(k))
		# Outer approximation
		OA = dd.Polytope.FromHalfspaces(OAw, np.sum(OAw * OAy, axis = 1))
		
		return IA, OA
