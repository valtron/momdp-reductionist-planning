import numpy as np
from numba import jit

from common import tilecoding

def cache(impl):
	return _CachedFeaturizer(impl.env, impl)

class _CachedFeaturizer:
	def __init__(self, env, impl):
		assert env.states is not None
		
		self.env = env
		self.impl = impl
		self._features_s = {}
		
		states = [
			s for s in env.states
			if env.terminal_value(s) is None
		]
		features_s = impl.featurize_states(states)
		stds = np.std(features_s, axis = 0)
		max_std = np.max(stds)
		if max_std < 1e-6:
			features_s = np.ones((features_s.shape[0], 1), dtype = features_s.dtype)
		else:
			features_s = features_s[:, stds >= 1e-4 * max_std]
		self.d = features_s.shape[1]
		self._features_s = {
			s: feature
			for s, feature in zip(states, features_s)
		}
	
	def featurize_state(self, state):
		return self._features_s[state]
	
	def featurize_states(self, states):
		return np.array([
			self._features_s[s] for s in states
		], dtype = np.float32)
	
	def __repr__(self):
		return repr(self.impl)
	
	def __reduce__(self):
		return (TabularFeaturizer, (self.env, self.impl))

class OneHotFeaturizer:
	def __init__(self, env):
		assert env.states is not None
		
		self.env = env
		self.num_actions = env.num_actions
		self.num_states = len(env.states)
		self.d = self.num_states
		self._state_index = {
			s: i for i, s in enumerate(env.states)
		}
	
	def featurize_state(self, state):
		features = np.zeros(self.d, dtype = np.float32)
		features[self._state_index[state]] = 1
		return features
	
	def featurize_states(self, states):
		features = np.zeros((len(states), self.d), dtype = np.float32)
		for i, s in enumerate(states):
			features[i, self._state_index[s]] = 1
		return features
	
	def __repr__(self):
		return 'onehot'
	
	def __reduce__(self):
		return (OneHotFeaturizer, (self.env,))

class TileCodingFeaturizer:
	def __init__(self, env, n_tilings, n_tiles):
		self.env = env
		self.tc = tilecoding.TileCoder(env.feature_ranges, n_tilings, n_tiles)
		self.d = self.tc.output_size
	
	def featurize_state(self, state):
		features = np.zeros(self.d, dtype = np.float32)
		if state is not None:
			features[self.tc.tiles(state)] = 1
		return features
	
	def featurize_states(self, states):
		features = np.zeros((len(states), self.d), dtype = np.float32)
		for i, s in enumerate(states):
			if s is None:
				continue
			features[i, self.tc.tiles(s)] = 1
		return features
	
	def __repr__(self):
		return 'tc({}, {})'.format(self.tc.n_tilings, self.tc.tiling_dims)
	
	def __reduce__(self):
		return (TileCodingFeaturizer, (self.impl,))

class FourierFeaturizer:
	def __init__(self, env, order):
		self.env = env
		self.order = order
		scale = np.pi/(env.feature_ranges[:,1] - env.feature_ranges[:,0])
		
		tmp = np.stack(
			np.meshgrid(*(
				np.arange(order + 1)
				for _ in range(len(env.feature_ranges))
			), indexing = 'ij'),
			axis = -1,
		).reshape((-1, len(env.feature_ranges)))
		tmp = tmp[np.sum(tmp, axis = 1) <= order]
		
		self.c = scale[:,None] * tmp.astype(np.float32).T
		self.d = self.c.shape[1]
		self.offset = env.feature_ranges[:,0]
	
	def featurize_state(self, state):
		return fourier_featurize_impl(state, self.offset, self.c)
	
	def featurize_states(self, states):
		states = np.asarray(states)
		return fourier_featurize_impl(states, self.offset, self.c)
	
	def __repr__(self) -> str:
		return 'fourier({})'.format(self.order)
	
	def __reduce__(self):
		return (FourierFeaturizer, (self.env, self.order))

@jit(nopython = True)
def fourier_featurize_impl(s, offset, c):
	return np.cos((s - offset) @ c)
