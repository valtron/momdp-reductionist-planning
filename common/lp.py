import numpy as np
from scipy.optimize import linprog

class TabularSolver:
	def __init__(self, env):
		gamma = env.gamma
		assert gamma < 1
		
		transitions, rewards, start_distribution = mdp_to_matrices(env)
		self.rewards = rewards
		self.fp = FlowPolytope(transitions, start_distribution, gamma)
		self.H = 1/(1 - gamma)
	
	def solve_linear(self, w: np.ndarray):
		c = self.rewards @ (-w)
		A_eq = self.fp.A
		b_eq = self.fp.b
		res = linprog(c, A_eq = A_eq, b_eq = b_eq, bounds = (0, None), method = 'highs')
		assert res.success, res.message
		mu = res.x
		value = (mu @ self.rewards) * self.H
		return value
	
	def solve_chebyshev(self, r: np.ndarray, w: np.ndarray):
		n_sa, _ = self.rewards.shape
		nz = (w > 0)
		k_nz = np.sum(nz)
		
		c = np.zeros(n_sa + 1)
		c[-1] = 1
		
		A_eq = np.concatenate([self.fp.A, np.zeros((self.fp.A.shape[0], 1))], axis = 1)
		b_eq = self.fp.b
		
		A_ub = np.zeros((k_nz, n_sa + 1))
		A_ub[:,:-1] = -((self.rewards[:,nz] - r[nz]/self.H) / w[nz]).T
		A_ub[:, -1] = -1
		b_ub = np.zeros(k_nz)
		
		bounds = [(0, None)] * n_sa + [(None, None)]
		res = linprog(c, A_eq = A_eq, b_eq = b_eq, A_ub = A_ub, b_ub = b_ub, bounds = bounds, method = 'highs')
		assert res.success, res.message
		mu = res.x[:-1]
		value = (mu @ self.rewards) * self.H
		return value
	
	def solve_hyperplane(self, y: np.ndarray):
		A, b = self.fp.A, self.fp.b
		
		n, k = self.rewards.shape
		m = A.shape[0]
		
		c = np.hstack((-y / self.H, b))
		
		A_eq = np.vstack((np.hstack((np.ones(k), np.zeros(m))),))
		b_eq = np.ones(1)
		
		A_ub = np.vstack((np.hstack((self.rewards, -A.T)),))
		b_ub = np.zeros(n)
		
		bounds = [(0, None)] * k + [(None, None)] * m
		res = linprog(c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq, b_eq = b_eq, bounds = bounds, method = 'highs')
		assert res.success, res.message
		
		w, u = res.x[:k], res.x[k:]
		const = (b @ u) * self.H
		value = y + const - y @ w
		
		return w, value

class FlowPolytope:
	# Equality constraint on the state-action occupancy:
	# { mu | A mu = b }
	A: np.ndarray
	b: np.ndarray
	
	def __init__(self, transitions, start_distribution, gamma):
		sa, s = transitions.shape
		a = sa // s
		occ_matrix = np.broadcast_to(np.eye(s)[:, None, :], (s, a, s)).reshape(transitions.shape) - gamma * transitions
		self.A = occ_matrix.T
		self.b = (1 - gamma) * start_distribution

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
