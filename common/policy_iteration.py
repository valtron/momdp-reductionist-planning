import abc

import numpy as np
from numba import jit

from .misc import SeedTree, dummy_progress

class PIImpl:
	@abc.abstractmethod
	def evaluate(self, policy, st: SeedTree): pass
	
	@abc.abstractmethod
	def improve(self, policy, theta): pass

def policy_iteration(pi0, t0: int, tmax: int, impl: PIImpl, st: SeedTree, *, progress = dummy_progress):
	policies = []
	values = []
	policy = pi0
	pbar = progress(range(t0, tmax))
	for t in pbar:
		theta, value = impl.evaluate(policy, st / t)
		policies.append(policy)
		values.append(value)
		pbar.set_postfix_str(str(value))
		policy = impl.improve(policy, theta)
	return policies, np.array(values), policy

class PolitexPolicy:
	def __init__(self, featurizer, theta = None):
		if theta is None:
			self.theta = np.zeros((featurizer.env.num_actions, featurizer.d))
		else:
			self.theta = theta
		self.featurizer = featurizer
		self.deterministic = False
	
	def __call__(self, rng, s):
		phi = self.featurizer.featurize_state(s)
		a = politex_sample_action(self.theta @ phi, rng.uniform())
		return a
	
	def probas(self, s):
		phi = self.featurizer.featurize_state(s)
		tmp = self.theta @ phi
		tmp -= np.max(tmp)
		np.exp(tmp, out = tmp)
		return tmp / np.sum(tmp)
	
	def __add__(self, theta):
		policy = PolitexPolicy(self.featurizer)
		policy.theta = self.theta + theta
		return policy

@jit(nopython = True)
def politex_sample_action(logits, u):
	logits -= np.max(logits)
	c = np.cumsum(np.exp(logits))
	return np.argmax(c[-1] * u <= c)
