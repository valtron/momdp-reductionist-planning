import abc
import numpy as np
from scipy.optimize import linprog
import dualdesc as dd
from common import scalarization as scal

Outcome = np.ndarray

class SOSolver(abc.ABC):
	@abc.abstractmethod
	def solve_linear(self, s: scal.Linear) -> Outcome: pass
	
	@abc.abstractmethod
	def solve_chebyshev(self, s: scal.Chebyshev) -> Outcome: pass

class PolytopeSolver(SOSolver):
	def __init__(self, V: np.ndarray, A: np.ndarray, b: np.ndarray):
		self.V = V
		self.A = A
		self.b = b
	
	def solve_linear(self, s):
		i = np.argmax(s(self.V))
		return self.V[i]
	
	def solve_chebyshev(self, s):
		tmp = (self.b - self.A @ s.r) / (self.A @ s.iw)
		i = np.argmin(tmp)
		return s.r + tmp[i] * s.iw, self.A[i]

class TabularSolver(SOSolver):
	def __init__(self, env):
		assert env.horizon == np.inf
		gamma = env.gamma
		assert gamma < 1
		
		transitions, rewards, start_distribution = _mdp_to_matrices(env)
		self.rewards = rewards
		self.A, self.b = _flow_polytope(transitions, start_distribution, gamma)
		self.H = 1/(1 - gamma)
	
	def solve_linear(self, s):
		c = -s(self.rewards)
		A_eq = self.A
		b_eq = self.b
		res = linprog(c, A_eq = A_eq, b_eq = b_eq, bounds = (0, None), method = 'highs')
		assert res.success, res.message
		mu = res.x
		value = (mu @ self.rewards) * self.H
		return value
	
	def solve_chebyshev(self, s):
		n_sa, _ = self.rewards.shape
		nz = (s.iw < np.inf)
		k_nz = np.sum(nz)
		
		c = np.zeros(n_sa + 1)
		c[-1] = 1
		
		A_eq = np.concatenate([self.A, np.zeros((self.A.shape[0], 1))], axis = 1)
		b_eq = self.b
		
		A_ub = np.zeros((k_nz, n_sa + 1))
		A_ub[:,:-1] = -((self.rewards[:,nz] - s.r[nz]/self.H) * s.iw[nz]).T
		A_ub[:, -1] = -1
		b_ub = np.zeros(k_nz)
		
		bounds = [(0, None)] * n_sa + [(None, None)]
		res = linprog(
			c, A_eq = A_eq, b_eq = b_eq, A_ub = A_ub, b_ub = b_ub, bounds = bounds, method = 'highs',
			options = { 'dual_feasibility_tolerance': 1e-9, 'primal_feasibility_tolerance': 1e-9 },
		)
		assert res.success, res.message
		mu = res.x[:-1]
		value = (mu @ self.rewards) * self.H
		
		return value, -res.ineqlin.marginals

def _flow_polytope(transitions, start_distribution, gamma):
	# Equality constraint on the state-action occupancy:
	# { mu | A mu = b }
	sa, s = transitions.shape
	a = sa // s
	occ_matrix = np.broadcast_to(np.eye(s)[:, None, :], (s, a, s)).reshape(transitions.shape) - gamma * transitions
	A = occ_matrix.T
	b = (1 - gamma) * start_distribution
	return A, b

def _mdp_to_matrices(mdp):
	# Given a simulator of a finite, deterministic MDP,
	# construct the transition, reward, and state distribution matrices.
	
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
				r, sp = mdp.transition(s, a)
				sp_i = handle_state(sp)
				sa_i = s_i * n_actions + a
				if len(next_states) <= sa_i:
					next_states.extend([None] * (sa_i + 1 - len(next_states)))
					rewards.extend([None] * (sa_i + 1 - len(rewards)))
				next_states[sa_i] = sp_i
				rewards[sa_i] = r
		return state_map[s]
	
	s0_i = handle_state(mdp.start)
	start_distribution = np.zeros((len(state_map),))
	start_distribution[s0_i] = 1
	
	rewards = np.array(rewards)
	transitions = np.zeros((len(next_states), len(state_map)))
	for sa_i, sp_i in enumerate(next_states):
		transitions[sa_i, sp_i] = 1
	
	return transitions, rewards, start_distribution
