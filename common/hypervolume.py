# Transplanted from BoTorch:
# https://github.com/pytorch/botorch/blob/main/botorch/utils/multi_objective/hypervolume.py

from typing import List, Optional
import numpy as np
import scipy.spatial

from common import dualdesc as dd

def hypervolume(Y: np.ndarray, ref: np.ndarray) -> float:
	return Hypervolume(ref)(Y)

def hypervolume_convex_cdd(Y: np.ndarray, ref: np.ndarray) -> float:
	k = Y.shape[1]
	dt = Y.dtype
	Rplus = np.eye(k, dtype = dt)
	h1 = dd.VRepr(Y, -Rplus).to_h()
	h2 = dd.VRepr(ref[None,:], Rplus).to_h()
	v = dd.HRepr.intersect(h1, h2).to_v()
	assert np.allclose(v.Vn, 0)
	assert np.allclose(v.Vl, 0)
	if len(v.Vc) <= k:
		return 0
	return scipy.spatial.ConvexHull(v.Vc).volume

def hypervolume_convex_qhull(Y: np.ndarray, ref: np.ndarray) -> float:
	import itertools
	from scipy.spatial import ConvexHull
	
	assert np.all(Y >= (ref - 1e-5)), np.min(Y - ref)
	Y = np.maximum(Y, ref)
	
	all_points = [[ref], Y]
	
	# Project each point onto each (1 .. k-1)-dimensional boundary
	k = ref.shape[0]
	idxs = list(range(k))
	for i in range(1, k):
		for boundary in itertools.combinations(idxs, i):
			boundary = list(boundary)
			projected = Y.copy()
			projected[:, boundary] = ref[boundary]
			all_points.append(projected)
	
	hull = ConvexHull(np.concatenate(all_points, axis = 0))
	return hull.volume

hypervolume_convex = hypervolume_convex_qhull

class Hypervolume:
	def __init__(self, ref: np.ndarray) -> None:
		assert ref.ndim == 1
		self.ref = ref
		self.list = None
	
	def __call__(self, Y: np.ndarray) -> float:
		assert Y.shape[1:] == self.ref.shape
		Y = Y[(Y >= self.ref).all(axis = 1)]
		# Shift reference point to zero, and flip to minimization
		self._initialize_multilist(self.ref - Y)
		bounds = np.full_like(self.ref, -np.inf)
		return self._hv_recursive(len(self.ref) - 1, len(Y), bounds)
	
	def _hv_recursive(self, i: int, n_pareto: int, bounds: np.ndarray) -> float:
		if n_pareto == 0:
			# base case: no points
			return 0
		
		sentinel = self.list.sentinel
		if i == 0:
			# base case: one dimension
			return -sentinel.next[0].data[0].item()
		
		hvol = 0
		if i == 1:
			# two dimensions, end recursion
			q = sentinel.next[1]
			h = q.data[0]
			p = q.next[1]
			while p is not sentinel:
				hvol += h * (q.data[1] - p.data[1])
				if p.data[0] < h:
					h = p.data[0]
				q = p
				p = q.next[1]
			hvol += h * q.data[1]
			return hvol
		
		p = sentinel
		q = p.prev[i]
		while q.data is not None:
			if q.ignore < i:
				q.ignore = 0
			q = q.prev[i]
		q = p.prev[i]
		while n_pareto > 1 and (q.data[i] > bounds[i] or q.prev[i].data[i] >= bounds[i]):
			p = q
			_, bounds = self.list.remove(p, i, bounds)
			q = p.prev[i]
			n_pareto -= 1
		q_prev = q.prev[i]
		if n_pareto > 1:
			hvol = q_prev.volume[i] + q_prev.area[i] * (q.data[i] - q_prev.data[i])
		else:
			q.area[0] = 1
			q.area[1 : i + 1] = q.area[:i] * -(q.data[:i])
		q.volume[i] = hvol
		if q.ignore >= i:
			q.area[i] = q_prev.area[i]
		else:
			q.area[i] = self._hv_recursive(i - 1, n_pareto, bounds)
			if q.area[i] <= q_prev.area[i]:
				q.ignore = i
		while p is not sentinel:
			p_data = p.data[i]
			hvol += q.area[i] * (p_data - q.data[i])
			bounds[i] = p_data
			bounds = self.list.reinsert(p, i, bounds)
			n_pareto += 1
			q = p
			p = p.next[i]
			q.volume[i] = hvol
			if q.ignore >= i:
				q.area[i] = q.prev[i].area[i]
			else:
				q.area[i] = self._hv_recursive(i - 1, n_pareto, bounds)
				if q.area[i] <= q.prev[i].area[i]:
					q.ignore = i
		hvol -= q.area[i] * q.data[i]
		return hvol

	def _initialize_multilist(self, pareto_Y: np.ndarray) -> None:
		m = pareto_Y.shape[-1]
		nodes = [
			Node(m, pareto_Y.dtype, point)
			for point in pareto_Y
		]
		self.list = MultiList(m, pareto_Y.dtype)
		for i in range(m):
			_sort_by_dimension(nodes, i)
			self.list.extend(nodes, i)

class Node:
	def __init__(self, m: int, dtype: np.dtype, data: Optional[np.ndarray] = None) -> None:
		self.data = data
		self.next = [None] * m
		self.prev = [None] * m
		self.ignore = 0
		self.area = np.zeros(m, dtype=dtype)
		self.volume = np.zeros_like(self.area)

def _sort_by_dimension(nodes: List[Node], i: int) -> None:
	# build a list of tuples of (point[i], node)
	decorated = [(node.data[i], index, node) for index, node in enumerate(nodes)]
	# sort by this value
	decorated.sort()
	# write back to original list
	nodes[:] = [node for (_, _, node) in decorated]

class MultiList:
	def __init__(self, m: int, dtype: np.dtype) -> None:
		self.m = m
		self.sentinel = Node(m, dtype)
		self.sentinel.next = [self.sentinel] * m
		self.sentinel.prev = [self.sentinel] * m
	
	def append(self, node: Node, index: int) -> None:
		last = self.sentinel.prev[index]
		node.next[index] = self.sentinel
		node.prev[index] = last
		# set the last element as the new one
		self.sentinel.prev[index] = node
		last.next[index] = node
	
	def extend(self, nodes: List[Node], index: int) -> None:
		for node in nodes:
			self.append(node, index)
	
	def remove(self, node: Node, index: int, bounds: np.ndarray) -> Node:
		for i in range(index):
			predecessor = node.prev[i]
			successor = node.next[i]
			predecessor.next[i] = successor
			successor.prev[i] = predecessor
		return node, np.minimum(bounds, node.data)
	
	def reinsert(self, node: Node, index: int, bounds: np.ndarray) -> None:
		for i in range(index):
			node.prev[i].next[i] = node
			node.next[i].prev[i] = node
		return np.minimum(bounds, node.data)
