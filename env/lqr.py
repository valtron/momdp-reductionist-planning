import numpy as np
import scipy
from tqdm.auto import tqdm

from common import misc, dualdesc as dd, scalarization as scal
from common.so_solver import PolytopeSolver, SOSolver

name = "LQR"
gamma = 1
horizon = np.inf
k = 3

def pareto_front_vertices():
	return _presolve_pf()[0]

def _presolve_pf():
	c = 0.1
	n = 50
	gamma = 0.9
	
	cache = misc.Cache('.cache') / 'lqr-c={:.1f},n={},gamma={:.1f}'.format(c, n, gamma)
	
	entry = cache / 'V'
	if not entry.exists():
		ws = misc.grid_simplex_covering(n, k)
		lqr = _make_lqr_prob()
		V = np.array([
			lqr.solve_linear(scal.Linear(w))
			for w in tqdm(ws, "LQR pre-solve")
		])
		entry.save(V)
	V = entry.load()
	
	entry = cache / 'A,b'
	if not entry.exists():
		pl = dd.Polytope.FromGenerators(V, -np.eye(k))
		Ab = pl.get_inequalities()
		entry.save(Ab)
	A, b = entry.load()
	
	return V, A, b

def _make_lqr_prob():
	c = 0.1
	gamma = 0.9
	
	s0 = np.array([10] * k)
	A = np.eye(k)
	B = np.eye(k)
	Q = np.array([
		np.diag([(1 - c if i == j else c) for j in range(k)])
		for i in range(k)
	])
	R = np.array([
		np.diag([(c if i == j else 1 - c) for j in range(k)])
		for i in range(k)
	])
	return _LQRSolver(A, B, Q, R, s0, gamma)

class _LQRSolver(SOSolver):
	def __init__(self, a, b, q, r, s0, gamma: float = 1):
		self.k = len(q)
		self.gamma = gamma
		self.q = q.transpose((1, 2, 0))
		self.r = r.transpose((1, 2, 0))
		self.a = np.sqrt(gamma) * a
		self.b = np.sqrt(gamma) * b
		if len(s0.shape) == 1:
			s0 = s0[None,:]
		self.s0 = s0
	
	def solve_linear(self, s):
		Q = self.q @ s.w
		R = self.r @ s.w
		P = scipy.linalg.solve_discrete_are(self.a, self.b, Q, R)
		gP = self.gamma * P
		F = np.linalg.solve(R + self.b.T @ gP @ self.b, self.b.T @ gP @ self.a)
		v = []
		for i in range(self.k):
			ss = self.q[...,i] + F.T @ self.r[...,i] @ F
			aa = self.a - self.b @ F
			mm = scipy.linalg.solve_discrete_lyapunov(aa.T, ss)
			v.append(np.mean(np.sum((self.s0 @ mm) * self.s0, axis = -1)))
		v = -np.array(v)
		return v
	
	def solve_chebyshev(self, s):
		w = np.ones(k) / k
		for t in range(100):
			v = self.solve_linear(scal.Linear(w))
			a = (v - s.r) * s.iw
			w[a >= np.min(a) + 1e-3] *= 1 - 1/(t+2)
			w /= np.sum(w)
		return v, w

V, A, b = _presolve_pf()
min_return = np.min(V, axis = 0)
max_return = np.max(V, axis = 0)
#solver = _make_lqr_prob()
solver = PolytopeSolver(V, A, b)
