# Transplanted from BoTorch:
# https://github.com/pytorch/botorch/blob/main/botorch/utils/multi_objective/pareto.py

import numpy as np

# maximum array size for simple pareto computation
MAX_BYTES = 5e6

def is_non_dominated(Y: np.ndarray) -> np.ndarray:
	assert Y.ndim == 2
	n, _ = Y.shape
	if n == 0:
		return np.zeros((n,), dtype = np.bool_)
	if n > 1000 or n**2 * Y.dtype.itemsize > MAX_BYTES:
		return _is_non_dominated_loop(Y)
	Y1 = np.expand_dims(Y, -3)
	Y2 = np.expand_dims(Y, -2)
	dominates = ((Y1 >= Y2).all(axis = -1) & (Y1 > Y2).any(axis = -1))
	nd_mask = ~dominates.any(axis = -1)
	return nd_mask

def _is_non_dominated_loop(Y: np.ndarray) -> np.ndarray:
	assert Y.ndim == 2
	n, _ = Y.shape
	is_efficient = np.ones((n,), dtype = np.bool_)
	for i in range(n):
		if not is_efficient[i]:
			continue
		update = (Y > Y[i:i+1,:]).any(axis = -1)
		# If an element in Y[i, :] is efficient, mark it as efficient
		update[i] = True
		# Only include batches where Y[i, :] is efficient
		is_efficient2 = is_efficient.copy()
		# Only include elements from in_efficient from the batches
		# where Y[i, :] is efficient
		is_efficient[is_efficient2] = update[is_efficient2]
	return is_efficient
