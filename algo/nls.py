import numpy as np
import dualdesc as dd
from common import misc, scalarization as scal

name = "NLS"
color = 'b'
deterministic = False
inner_approximation = True

class Algo:
	def __init__(self, lo, hi, *, rng):
		k = len(lo)
		self.IA = dd.Polytope.FromGenerators(lo[None,:], -np.eye(k))
		self.C = misc.incremental_simplex_covering(k, rng)
	
	@property
	def approx_poly(self):
		return self.IA
	
	def update(self, solver):
		wt = next(self.C)
		yt = solver.solve_linear(scal.Linear(wt))
		self.IA.add_point(yt)
		return False
