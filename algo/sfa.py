import itertools
import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import cvxpy as cp
from sklearn import kernel_approximation
import dualdesc as dd
from common import misc, scalarization as scal

name = "SFA"
color = 'g'
deterministic = False
inner_approximation = False

class Algo:
	def __init__(self, lo, hi, *, rng):
		k = len(lo)
		
		t = (120 if k == 2 else 15)
		self.lamb = 1e-6
		
		self.k = k
		self.lo = lo
		self.W = []
		self.Y = []
		self.C = rng.permutation(misc.grid_simplex_covering(t, k))
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
		return False
	
	def _build_poly(self):
		W = np.array(self.W)
		Y = np.array(self.Y)
		
		if len(W) < self.k:
			if len(W) == 0:
				Y = self.lo[None,:]
			return dd.Polytope.FromGenerators(Y, -np.eye(self.k))
		
		h_data = np.sum(W * Y, axis = -1)
		mu = np.mean(h_data)
		sd = np.std(h_data)
		h_std = (h_data - mu)/sd
		
		l = _find_lengthscale(self.C[:len(W)], h_std, 0.1, 10, 20)
		featurizer = _OrthoNystroem('rbf', gamma = 0.5/l**2)
		featurizer.fit(self.C)
		
		weights = (featurizer.S[0] / featurizer.S)**(1/10)
		theta_est = sd * _basis_pursuit(featurizer.transform(W), h_std, self.lamb, weights)
		
		def h_est(ws):
			return featurizer.transform(ws) @ theta_est + mu
		
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
		K = kernel_approximation.pairwise_kernels(
			X,
			metric=self.kernel,
			filter_params=True,
			n_jobs=self.n_jobs,
			**self._get_kernel_params(),
		)
		S, U = np.linalg.eigh(K)
		idxs = (S >= 1e-13)
		S = S[idxs]
		U = U[:,idxs]
		n_components = len(S)
		self.S = S
		self.normalization_ = (U/S).T
		self.components_ = X
		self.component_indices_ = np.arange(n_components)
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

def _find_lengthscale(X, y, lmin, lmax, n):
	ls = np.geomspace(lmin, lmax, n)
	scores = []
	
	C = np.linalg.norm(X[:,:,None] - X.T, axis = 1)**2
	
	for l in ls:
		K = (-0.5/l**2) * C
		np.exp(K, out = K)
		S, U = np.linalg.eigh(K)
		idxs = (S >= 1e-13)
		S = S[idxs]
		U = U[:,idxs]
		lambdas_i = 1/S
		Ki = U @ np.diag(lambdas_i) @ U.T
		score = np.linalg.norm(Ki @ y)/np.mean(lambdas_i)
		scores.append(score)
	scores = np.array(scores)
	
	return ls[np.argmin(scores)]
