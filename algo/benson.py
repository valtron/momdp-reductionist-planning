import numpy as np
import dualdesc as dd
from common import scalarization as scal
from common.misc import close_to_any

name = "Benson"
color = 'c'
deterministic = True
inner_approximation = True

class Algo:
	def __init__(self, lo, hi, *, rng):
		k = len(lo)
		self.IA = dd.Polytope.FromGenerators(lo[None,:], -np.eye(k))
		self.OA = dd.Polytope.FromGenerators(hi[None,:], -np.eye(k))
		self.verts_checked = []
	
	@property
	def approx_poly(self):
		return self.IA
	
	def update(self, solver):
		V = self.OA.get_generators()[0]
		for v in V:
			if close_to_any(v, self.verts_checked, 1e-5):
				continue
			self.verts_checked.append(v)
			u, w = solver.solve_chebyshev(scal.Chebyshev(v))
			self.IA.add_point(u)
			self.OA.add_halfspace(w, w @ u)
			return False
		return True
