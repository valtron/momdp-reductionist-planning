from typing import Union

import numpy as np
from scipy.stats import qmc

class TileCoder:
	def __init__(self, ranges: np.ndarray, n_tilings: int, n_tiles: Union[int, np.ndarray]):
		"""
			ranges: float[d, 2]; `ranges[i]` is the (min, max) of the `i`th dimension
			n_tilings: number of tilings
			n_tiles: number of tiles; same for all dimensions
		"""
		
		d = len(ranges)
		
		if isinstance(n_tiles, int):
			tiling_dims = np.ones((d,), dtype = np.int32) * n_tiles
		else:
			tiling_dims = n_tiles
		
		mins = ranges[:,0]
		maxs = ranges[:,1]
		
		tile_widths = (1 - 1 / ((tiling_dims - 1) * n_tilings + 1)) * (maxs - mins) / (tiling_dims - 1)
		max_offsets = tile_widths * (1 - 1/n_tilings)
		
		self.d = d
		self.n_tilings = n_tilings
		self.offsets = (mins - qmc.LatinHypercube(d, seed = 0).random(n_tilings) * max_offsets).T
		
		self.scales = (1 / tile_widths)[:,None]
		self.tiling_dims = tiling_dims
		self.tile_offsets = np.arange(n_tilings, dtype = np.int32) * np.prod(self.tiling_dims)
		self.output_size = np.prod(self.tiling_dims) * n_tilings
	
	def tiles(self, x: np.ndarray):
		"""
			x: float[*S, d]; S is an arbitary shape
			return: int[*S, n_tilings]; indices of the active tiles
		"""
		S = x.shape[:-1]
		tmp = x[..., None] - self.offsets
		tmp *= self.scales
		tmp = tmp.astype(np.int32)
		tmp = np.swapaxes(tmp.reshape((-1, self.d, self.n_tilings)), 0, 1)
		tiles = np.ravel_multi_index(tmp, self.tiling_dims, mode = 'clip')
		tiles += self.tile_offsets
		tiles = tiles.reshape(S + (self.n_tilings,))
		return tiles
	
	def features(self, tiles: np.ndarray, *, dtype: np.dtype = np.float32):
		"""
			tiles: int[*S, n_tilings]; indices of the active tiles
			dtype: np.dtype to use for output
			return: dtype[*S, output_size]; 0-1 features of the active tiles
		"""
		S = tiles.shape[:-1]
		tmp = tiles.reshape((-1, self.n_tilings))
		features = np.zeros((tmp.shape[0], self.output_size), dtype = dtype)
		features[np.arange(len(tmp)), tmp.T] = 1
		features = features.reshape(S + (self.output_size,))
		return features

def _test():
	from matplotlib import pyplot as plt
	
	ranges = np.array([[-1, 1], [0, 2]], dtype = np.float32)
	n_tilings = 3
	tc = TileCoder(ranges, n_tilings, np.array([4, 3], dtype = np.int32))
	
	pts = np.stack(
		np.meshgrid(*(np.linspace(*r, 80) for r in ranges), indexing = 'ij'), axis = -1,
	)
	
	tiles = tc.tiles(pts)
	assert np.all(tiles[0,0] == tc.tiles(pts[0,0]))
	assert np.all(tiles[0] == tc.tiles(pts[0]))
	
	features = tc.features(tiles, dtype = np.bool_)
	assert np.all(features[0,0] == tc.features(tiles[0,0], dtype = np.bool_))
	assert np.all(features[0] == tc.features(tiles[0], dtype = np.bool_))
	assert np.all(np.sum(features, axis = -1) == n_tilings)
	
	m = np.prod(tc.tiling_dims)
	
	fig, axs = plt.subplots(1, n_tilings, figsize = (2 * n_tilings, 2), constrained_layout = True)
	for i in range(n_tilings):
		for j in range(m):
			idxs = (tiles[..., i] == m*i + j)
			axs[i].plot(pts[idxs,0], pts[idxs,1], '.', markersize = 2)
	plt.show()

if __name__ == '__main__':
	_test()
