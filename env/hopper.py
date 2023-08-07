import numpy as np
from tqdm.auto import tqdm
from mo_gymnasium.envs.mujoco import hopper as _impl

from common import misc, dualdesc as dd
from common.so_solver import PolytopeSolver

name = "Hopper"
gamma = 1
horizon = 100
objective_names = ["Forward", "Jump", "Control Cost"]
k = 3

def pareto_front_vertices():
	return _presolve_pf()[0]

def _presolve_pf():
	seed = 0
	nu = 0.05
	kq = 1000
	
	cache = misc.Cache('.cache') / 'hopper-seed={},nu={:.2f},kq={}'.format(
		seed, nu, kq,
	)
	
	entry = cache / 'outcomes,policies'
	if not entry.exists():
		env = _GymEnv(seed)
		ops = _random_search(env, seed, nu, kq * 1000)
		entry.save(ops)
	outcomes, _ = entry.load()
	
	entry = cache / 'A,b'
	if not entry.exists():
		pl = dd.Polytope.FromGenerators(outcomes, -np.eye(k))
		V, _ = pl.get_generators()
		A, b = pl.get_inequalities()
		entry.save((V, A, b))
	V, A, b = entry.load()
	
	return V, A, b

class _GymEnv(_impl.MOHopperEnv):
	dim_state = 11
	dim_action = 3
	
	def __init__(self, seed):
		super().__init__(terminate_when_unhealthy = False, healthy_reward = 0)
		self._t = None
		self.reset(seed)
	
	def reset(self, seed = None):
		self._t = 0
		return super().reset(seed = seed)
	
	def step(self, action):
		self._t += 1
		return super().step(action)
	
	@property
	def terminated(self):
		return self._t >= horizon

def _random_search(env, seed, nu, max_queries):
	da = env.dim_action
	ds = env.dim_state
	rng = np.random.default_rng(seed)
	
	pm = _ParetoMap(k)
	
	total = 0
	
	def update(policy):
		nonlocal total
		outcome = _eval_linear(env, policy)
		pm.add(outcome, policy)
		pbar.update(1)
		total += 1
		if total % 1000 == 0:
			tqdm.write("{} {}".format(total, len(pm)))
	
	with tqdm(total = max_queries, disable = False) as pbar:
		for _ in range(10_000):
			policy = rng.uniform(-0.5, 0.5, size = (da, ds))
			update(policy)
		
		while total < max_queries:
			policy = rng.choice(pm.values()) + nu * rng.normal(size = (da, ds))
			update(policy)
	
	return pm.outcomes(), pm.policies()

def _eval_linear(env, policy):
	v = np.zeros(k)
	s, _ = env.reset()
	while True:
		a = np.clip(policy @ s, -1, 1)
		sp, r, done, _, _ = env.step(a)
		v += r
		s = sp
		if done:
			break
	return v

class _ParetoMap:
	# Collection of Pareto-optimal (outcome, policy) pairs
	
	def __init__(self, k):
		self.k = k
		self.buf_outcome = np.empty((10, k))
		self.buf_policy = np.empty(10, dtype = object)
		self.len = 0
	
	def add(self, outcome, policy):
		cmp = (outcome >= self.outcomes())
		if not np.all(np.any(cmp, axis = 1)):
			return False
		m = np.all(cmp, axis = 1)
		if np.any(m):
			idxs = np.nonzero(~m)[0]
			self.buf_outcome[:len(idxs)] = self.buf_outcome[idxs]
			self.buf_policy[:len(idxs)] = self.buf_policy[idxs]
			self.len = len(idxs)
		if self.len >= len(self.buf_outcome):
			self.buf_outcome = np.concatenate([self.buf_outcome, self.buf_outcome], axis = 0)
			self.buf_policy = np.concatenate([self.buf_policy, self.buf_policy], axis = 0)
		self.buf_outcome[self.len] = outcome
		self.buf_policy[self.len] = policy
		self.len += 1
		return True
	
	def __getitem__(self, idx):
		if isinstance(idx, int):
			assert 0 <= idx < self.len, (idx, self.len)
			return (self.buf_outcome[idx], self.buf_policy[idx])
		ps = type(self)(self.k)
		ps.buf_outcome = self.keys()[idx]
		ps.buf_policy = self.values()[idx]
		ps.len = len(ps.buf_outcome)
		return ps
	
	def __len__(self):
		return self.len
	
	def items(self):
		for i in range(self.len):
			yield self.buf_outcome[i], self.buf_policy[i]
	
	def outcomes(self):
		return self.buf_outcome[:self.len]
	
	def policies(self):
		return self.buf_policy[:self.len]

V, A, b = _presolve_pf()
min_return = np.min(V, axis = 0)
max_return = np.max(V, axis = 0)
solver = PolytopeSolver(V, A, b)
