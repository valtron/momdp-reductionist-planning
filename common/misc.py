from joblib import Parallel

import numpy as np
from tqdm import tqdm

def dummy_progress(*args, **kwargs):
	kwargs['disable'] = True
	return tqdm(*args, **kwargs)

class SeedTree:
	def __init__(self, *key: int):
		assert len(key) > 0
		self.key = key
	
	def rng(self) -> np.random.Generator:
		ss = np.random.SeedSequence(self.key[0], spawn_key = self.key[1:])
		return np.random.default_rng(ss)
	
	def __truediv__(self, key: int):
		return SeedTree(self.key + (key,))

def deduplicate_and_sort(points):
	points = sorted(map(tuple, points))
	keep = [points[0]]
	for i in range(1, len(points)):
		if np.allclose(keep[-1], points[i]):
			continue
		keep.append(points[i])
	return np.array(keep)

def close_to_any(x, points, dist):
	return any(np.linalg.norm(x - p, np.inf) < dist for p in points)

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
