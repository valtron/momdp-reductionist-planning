from typing import Optional
import numpy as np

class Linear:
	def __init__(self, w: np.ndarray):
		assert np.all(w >= 0), w
		assert np.any(w > 0), w
		self.w = w
	
	def __call__(self, outcome: np.ndarray):
		return outcome @ self.w

class Chebyshev:
	def __init__(self, r: Optional[np.ndarray] = None, w: Optional[np.ndarray] = None):
		assert (r is not None) or (w is not None)
		
		if w is not None:
			assert np.all(w >= 0), w
			assert np.any(w > 0), w
		
		if r is None:
			self.r = np.zeros_like(w)
		else:
			self.r = r
		
		if w is None:
			self.w = np.ones(len(r))
		else:
			self.w = w
		
		with np.errstate(divide = 'ignore'):
			self.iw = 1 / self.w
	
	def __call__(self, outcome: np.ndarray):
		return np.min((outcome - self.r) * self.iw, axis = -1)
