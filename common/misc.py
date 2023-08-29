from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm

def dummy_progress(*args, **kwargs):
	kwargs['disable'] = True
	return tqdm(*args, **kwargs)

def close_to_any(x, points, dist):
	return any(np.linalg.norm(x - p, np.inf) < dist for p in points)

def incremental_simplex_covering(k, rng):
	alpha = np.ones(k)
	C = np.zeros((k + 1, k))
	C[0] = np.ones(k) / k
	i = 0
	while True:
		yield C[i]
		i += 1
		if i >= len(C):
			C = np.concatenate([C, C], axis = 0)
		samples = rng.dirichlet(alpha, size = i)
		dists = np.linalg.norm(C[:i,:,None] - samples.T[None,:,:], np.inf, axis = 1)
		C[i] = samples[np.argmax(np.min(dists, axis = 0))]

def grid_simplex_covering(n, k, *, mode = 'contain'):
	if mode not in { 'contain', 'cover' }:
		raise ValueError("invalid mode", mode)
	if mode == 'contain':
		if n < 2:
			raise ValueError("n must be >= 2 for mode = 'contain'")
	def _aux(n, k):
		if k == 1:
			return [(n - 1,)]
		l = []
		for i in range(n):
			for a in _aux(n - i, k - 1):
				l.append((i, *a))
		return l
	a = np.array(_aux(n, k), dtype = np.float64)
	if mode == 'contain':
		a /= n - 1
	else:
		a += 1/k
		a /= n
	return a

class Cache:
	def __init__(self, path):
		self.dir_path = Path(path).absolute()
		self.file_path = self.dir_path.with_suffix('.pickle')
	
	def __truediv__(self, key):
		return Cache(self.dir_path / key)
	
	def exists(self):
		return self.file_path.exists()
	
	def save(self, data):
		self.file_path.parent.mkdir(parents = True, exist_ok = True)
		pickle_save(self.file_path, data)
	
	def load(self):
		return pickle_load(self.file_path)

def pickle_load(path):
	with path.open('rb') as fh:
		return pickle.load(fh)

def pickle_save(path, data):
	with path.open('wb') as fh:
		pickle.dump(data, fh)
