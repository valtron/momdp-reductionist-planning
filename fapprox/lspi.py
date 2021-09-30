import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import tilecoding
from common import ScalarizedEnv, evaluate
from env import deep_sea_treasure

def main():
	env = deep_sea_treasure.Env
	#featurizer = DirectFeaturizer(env)
	featurizer = TileCodedFeaturizer(env, 4, 4)
	
	rng = np.random.default_rng(0)
	l = list(deep_sea_treasure.OCCUPIABLE)
	#rng.shuffle(l)
	#l = l[:len(l)//2]
	l = [(y, x) for (y, x) in l if (y + x) % 2 == 1]
	C = [
		(s, a)
		for s in l
		for a in range(env.num_actions)
	]
	
	estimated_pf = []
	for th in np.linspace(0, np.pi / 2, 3):
		w = np.array([np.cos(th), np.sin(th)])
		senv = ScalarizedEnv(env, w)
		policy = lspi(senv, featurizer, C, 200, 100, 1, None)
		outcome = evaluate(policy, env)
		print(w, outcome)
		estimated_pf.append(outcome)
	estimated_pf = np.array(estimated_pf)
	
	true_pf = deep_sea_treasure.true_pareto_front()
	plt.ylabel("Discounted time penalty")
	plt.xlabel("Discounted treasure value")
	plt.plot(estimated_pf[:,0], estimated_pf[:,1], 'bo', label = "Found points")
	plt.plot(true_pf[:,0], true_pf[:,1], 'r+', label = "True PF")
	plt.legend()
	plt.show()

def lspi(env, featurizer, C, K, H, m, rng):
	theta = np.zeros(featurizer.d)
	m = 100
	
	C_phi = featurizer(C)
	
	for k in range(K):
		R_hat = np.array([
			rollout(c, env, featurizer, theta, H, m, rng)
			for c in C
		])
		theta_new = np.linalg.lstsq(C_phi, R_hat, rcond = None)[0]
		delta = np.max(np.abs(theta_new - theta))
		print(k, delta)
		if delta < 1e-4:
			break
		theta = theta_new
	
	return LSPIPolicy(theta, env, featurizer)

class LSPIPolicy:
	def __init__(self, theta, env, featurizer):
		self.theta = theta
		self.featurizer = featurizer
		self.num_actions = env.num_actions
		self.deterministic = True
	
	def __call__(self, rng, s):
		qs = self.featurizer([(s, a) for a in range(self.num_actions)]) @ self.theta
		return np.argmax(qs)

def rollout(c, env, featurizer, theta, H, m, rng):
	ret = 0
	gamma = env.gamma
	if env.deterministic:
		s, a = c
		for t in range(H):
			tv = env.terminal_value(s)
			if tv is not None:
				ret += gamma**t * tv
				break
			if t > 0:
				qs = featurizer([(s, a) for a in range(env.num_actions)]) @ theta
				a = np.argmax(qs)
			r, sp = env.sample_transition(rng, s, a)
			ret += gamma**t * r
			s = sp
	else:
		for j in range(m):
			Hj = rng.geometric(1 - gamma)
			s, a = c
			for t in range(Hj):
				tv = env.terminal_value(s)
				if tv is not None:
					ret += (1 - gamma**(Hj-t)) * tv
					break
				if t > 0:
					qs = featurizer([(s, a) for a in range(env.num_actions)]) @ theta
					a = np.argmax(qs)
				r, sp = env.sample_transition(rng, s, a)
				ret += r
				s = sp
	return ret / m

class TileCodedFeaturizer:
	def __init__(self, env, n_tilings, n_tiles):
		self.tc = tilecoding.TileCoder(env.feature_ranges, n_tilings, n_tiles)
		self.env = env
		self.d = env.num_actions * self.tc.output_size
	
	def __call__(self, state_actions):
		env = self.env
		features = []
		for s, a in state_actions:
			feature = np.zeros((env.num_actions, self.tc.output_size))
			if env.terminal_value(s) is None:
				tiles = self.tc.tiles(np.array(list(s)))
				feature[a, tiles] = 1
			features.append(feature.flatten())
		return np.array(features, dtype = np.float32)

class DirectFeaturizer:
	"""
		- one-hot-encodes the action
		- uses the state vector directly
		- indicator feature for terminal state
	"""
	
	def __init__(self, env):
		self.env = env
		self.d = len(env.feature_ranges) + env.num_actions + 1
	
	def __call__(self, state_actions):
		env = self.env
		d = len(env.feature_ranges)
		num_actions = env.num_actions
		features = []
		for s, a in state_actions:
			feature = np.zeros(d + num_actions + 1)
			if env.terminal_value(s) is None:
				feature[:d] = s
			else:
				feature[-1] = 1
			feature[d + 1 + a] = 1
			features.append(feature)
		return np.array(features, dtype = np.float32)

if __name__ == '__main__':
	main()
