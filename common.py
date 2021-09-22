import numpy as np
from joblib import Parallel
from scipy.optimize import linprog
from tqdm.auto import tqdm

def get_outcome_space_bbox(comb, k):
	min_returns = np.array([comb(-e) for e in np.eye(k)])
	max_returns = np.array([comb(+e) for e in np.eye(k)])
	return np.min(min_returns, axis = 0), np.max(max_returns, axis = 0)

class LPMDPSolver:
	# Solves a discounted tabular MDP with linear programming

	def __init__(self, transitions, start_distribution, gamma):
		self.fp = FlowPolytope(transitions, start_distribution, gamma)

	def find_mu_star(self, rewards):
		# Finds the optimal state-action distribution (mu) for `rewards`
		res = linprog(-rewards, A_eq=self.fp.A, b_eq=self.fp.b)
		assert res.success, res.message
		return res.x

class CMDPSolver:
	def __init__(self, transitions, start_distribution, gamma):
		self.fp = FlowPolytope(transitions, start_distribution, gamma)
	
	def find_mu_star(self, rewards, constraints=None):
		# Finds the optimal state-action distribution (mu) for `rewards`
		if constraints:
			A_ub, b_ub = constraints
		else:
			A_ub, b_ub = None, None
		res = linprog(-rewards, A_eq=self.fp.A, b_eq=self.fp.b, A_ub=A_ub, b_ub=b_ub)
		if constraints is not None and res.status == 2:
			# Infeasible
			return None
		assert res.success, res.message
		return res.x
	
	def find_closest_mu(self, point, rewards):
		# Solves:
		# min_{y in R, mu in self.fp} y
		# s.t. |(1 - gamma) point - rewards @ mu| <= y
		
		n_sa, k = rewards.shape
		
		# Optimization variables are [y, mu]
		
		# Add `y` column to the flow polytope constraint matrix
		A_eq = np.concatenate([np.zeros((self.fp.A.shape[0], 1)), self.fp.A], axis=1)
		
		# Make constraint for `|(1 - gamma) point - rewards @ mu| <= y`
		A_ub = np.concatenate([
			np.concatenate([-np.ones((k, 1)), -rewards.T], axis=1),
			np.concatenate([-np.ones((k, 1)), rewards.T], axis=1),
		], axis=0)
		b_ub = np.concatenate([
			-point,
			point,
		], axis=0) * (1 - self.fp.gamma)
		
		# Objective is just `y`
		c = np.zeros((1 + n_sa,))
		c[0] = 1
		
		res = linprog(c, A_eq=A_eq, b_eq=self.fp.b, A_ub=A_ub, b_ub=b_ub)
		if res.status == 2:
			return None
		assert res.success, res.message
		return res.x[1:]
	
	def make_return_constraints(self, rewards, point):
		reward_contraint_matrix = -rewards.T
		constr_rhs = -point * (1 - self.fp.gamma)
		return reward_contraint_matrix, constr_rhs

class FlowPolytope:
	# Equality constraint on the state-action occupancy:
	# { mu | A mu = b }
	A: np.ndarray
	b: np.ndarray
	gamma: float
	
	def __init__(self, transitions, start_distribution, gamma):
		sa, s = transitions.shape
		a = sa // s
		occ_matrix = np.broadcast_to(np.eye(s)[:, None, :], (s, a, s)).reshape(transitions.shape) - gamma * transitions
		self.A = occ_matrix.T
		self.b = (1 - gamma) * start_distribution
		self.gamma = gamma

def deduplicate_and_sort(points):
	points = sorted(map(tuple, points))
	keep = [points[0]]
	for i in range(1, len(points)):
		if np.allclose(keep[-1], points[i]):
			continue
		keep.append(points[i])
	return np.array(keep)

def mdp_to_matrices(mdp):
	# Given a simulator of a finite, deterministic MDP,
	# construct the transition, reward, and state distribution matrices.
	
	assert mdp.deterministic
	
	next_states = []
	rewards = []
	
	state_map = {}
	n_actions = mdp.num_actions
	
	def handle_state(s):
		seen = (s in state_map)
		if not seen:
			s_i = len(state_map)
			state_map[s] = s_i
			for a in range(n_actions):
				r, sp = mdp.sample_transition(None, s, a)
				sp_i = handle_state(sp)
				sa_i = s_i * n_actions + a
				if len(next_states) <= sa_i:
					next_states.extend([None] * (sa_i + 1 - len(next_states)))
					rewards.extend([None] * (sa_i + 1 - len(rewards)))
				next_states[sa_i] = sp_i
				rewards[sa_i] = r
		return state_map[s]
	
	s0_i = handle_state(mdp.sample_start(None))
	start_distribution = np.zeros((len(state_map),))
	start_distribution[s0_i] = 1
	
	rewards = np.array(rewards)
	transitions = np.zeros((len(next_states), len(state_map)))
	for sa_i, sp_i in enumerate(next_states):
		transitions[sa_i, sp_i] = 1
	
	return transitions, rewards, start_distribution

def get_pareto_optimal(y):
	# return set of indices of pareto-optimal points
	# d[i,j] = (y[i] >= y[j] && y[i] != y[j])
	d = np.all(y[:, :, None] >= y.T[None, :, :], axis=1)
	d = d & ~np.all(np.isclose(y[:, :, None], y.T[None, :, :]), axis=1)
	return set(np.nonzero(~np.any(d, axis=0))[0])

def make_linear_comb(mdp_solver, rewards, gamma):
	# Given an MDP solver, returns "a subroutine `Comb` that [maximizes]
	# [...] linear combinations of the objectives" (https://arxiv.org/abs/1309.7084).
	# I.e., `comb(w)` returns an optimal outcome for the linear scalarization `w`.
	H = 1 / (1 - gamma)
	
	def comb(w):
		# Given linear scalarization `w`, return a maximizing policy
		mu = mdp_solver.find_mu_star(rewards @ w)
		# Calculate policy value (outcome)
		return (mu @ rewards) * H
	
	return comb

def dummy_progress(*args, **kwargs):
	kwargs['disable'] = True
	return tqdm(*args, **kwargs)

class CountCalls(object):
	"""
	Decorator that keeps track of the number of times a function is called.
	From the PythonDecoratorLibrary
	https://wiki.python.org/moin/PythonDecoratorLibrary#Counting_function_calls
	"""
	
	__instances = {}
	
	def __init__(self, f):
		self.__f = f
		self.__numcalls = 0
		CountCalls.__instances[f] = self
	
	def __call__(self, *args, **kwargs):
		self.__numcalls += 1
		return self.__f(*args, **kwargs)
	
	@staticmethod
	def count(f):
		"Return the number of times the function f was called."
		return CountCalls.__instances[f].__numcalls
	
	@staticmethod
	def counts():
		"Return a dict of {function: # of calls} for all registered functions."
		return dict([(f, CountCalls.count(f)) for f in CountCalls.__instances])

class ProgressParallel(Parallel):
	"""
		Progress bar for `joblib.Parallel`: https://stackoverflow.com/a/61900501/1329615
	"""
	
	def __init__(self, use_tqdm = True, total = None, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._use_tqdm = use_tqdm
		self._total = total
	
	def __call__(self, *args, **kwargs):
		with tqdm(disable = not self._use_tqdm, total = self._total) as self._pbar:
			return super().__call__(*args, **kwargs)
	
	def print_progress(self):
		if self._total is None:
			self._pbar.total = self.n_dispatched_tasks
		self._pbar.n = self.n_completed_tasks
		self._pbar.refresh()
