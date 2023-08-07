import itertools
import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import cvxpy as cp
from sklearn import kernel_approximation

from common import dualdesc as dd, misc, scalarization as scal

name = "SFA"
color = 'g'
deterministic = False
inner_approximation = False

class Algo:
	def __init__(self, lo, hi, *, rng):
		k = len(lo)
		
		t = (120 if k == 2 else 15)
		gamma = 0.4
		self.lamb = 1e-6
		
		self.k = k
		self.lo = lo
		self.W = []
		self.Y = []
		self.Phi = []
		self.C = rng.permutation(misc.grid_simplex_covering(t, k))
		self.featurizer = _OrthoNystroem('rbf', gamma = gamma, n_components = len(self.C))
		self.featurizer.fit(self.C)
		self._approx_poly = None
	
	@property
	def approx_poly(self):
		if self._approx_poly is None:
			self._approx_poly = self._build_poly()
		return self._approx_poly
	
	def update(self, solver):
		self._approx_poly = None
		self.W.append(self.C[len(self.W)])
		self.Y.append(solver.solve_linear(scal.Linear(self.W[-1])))
		self.Phi.append(self.featurizer.transform(self.W[-1:])[0])
		return False
	
	def _build_poly(self):
		W = np.array(self.W)
		Y = np.array(self.Y)
		
		if len(W) < self.k:
			if len(W) == 0:
				Y = self.lo[None,:]
			return dd.Polytope.FromGenerators(Y, -np.eye(self.k))
		
		Phi = np.array(self.Phi)
		
		h_data = np.sum(W * Y, axis = -1)
		mu = np.mean(h_data)
		sd = np.std(h_data)
		weights = (self.featurizer.S[0] / self.featurizer.S)**(1/10)
		theta_est = sd * _basis_pursuit(Phi, (h_data-mu)/sd, self.lamb, weights)
		
		def h_est(ws):
			Phi = self.featurizer.transform(ws)
			return Phi @ theta_est + mu
		
		ws = misc.grid_simplex_covering(500 if self.k == 2 else 25, self.k)
		hs = h_est(ws)
		pf_approx = dd.Polytope.FromHalfspaces(ws, hs)
		pf_approx.add_halfspace(W, h_data)
		pf_approx.add_point(Y)
		pf_approx.add_point(self.lo)
		pf_approx.add_halfspace(-np.eye(self.k), -self.lo)
		return pf_approx

class _OrthoNystroem(kernel_approximation.Nystroem):
	def fit(self, X, y = None):
		X = self._validate_data(X, accept_sparse = 'csr')
		n_samples = X.shape[0]
		n_components = min(n_samples, self.n_components)
		basis_inds = np.arange(n_components)
		
		basis_kernel = kernel_approximation.pairwise_kernels(
			X,
			metric=self.kernel,
			filter_params=True,
			n_jobs=self.n_jobs,
			**self._get_kernel_params(),
		)
		
		U, S, Vh = kernel_approximation.svd(basis_kernel)
		S = np.maximum(S, 1e-8)
		self.S = S
		self.normalization_ = (1/S)[:,None] * Vh
		self.components_ = X
		self.component_indices_ = basis_inds
		self._n_features_out = n_components
		return self

def _basis_pursuit(A, C, lamb, weights = None):
	theta = cp.Variable(A.shape[1])
	theta.value = C @ A
	if weights is None:
		wtheta = theta
	else:
		wtheta = cp.multiply(weights, theta)
	# min_theta 1/2 |C - A theta|_2^2 + lamb |theta|_{weights,1}
	prob = cp.Problem(cp.Minimize(cp.norm(C - A @ theta, 2)**2 + (2 * lamb) * cp.norm(wtheta, 1)))
	prob.solve(warm_start = True)
	assert theta.value is not None, prob.status
	return theta.value
