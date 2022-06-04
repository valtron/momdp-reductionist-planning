from __future__ import annotations
from typing import Optional

import numpy as np
import cdd

class HRepr:
	def __init__(
		self,
		Au: Optional[np.ndarray] = None,
		bu: Optional[np.ndarray] = None,
		Ae: Optional[np.ndarray] = None,
		be: Optional[np.ndarray] = None,
	) -> None:
		assert (Au is None) == (bu is None)
		assert (Ae is None) == (be is None)
		A = _or(Au, Ae)
		assert A is not None
		
		self.dim = A.shape[1]
		self.dtype = A.dtype
		
		if Au is None:
			Au = np.empty((0, self.dim), dtype = A.dtype)
		if bu is None:
			bu = np.empty(0, dtype = A.dtype)
		if Ae is None:
			Ae = np.empty((0, self.dim), dtype = A.dtype)
		if be is None:
			be = np.empty(0, dtype = A.dtype)
		
		self.Au = Au
		self.bu = bu
		self.Ae = Ae
		self.be = be
	
	@classmethod
	def intersect(cls, *hs: HRepr) -> HRepr:
		Au = []
		bu = []
		Ae = []
		be = []
		for h in hs:
			Au.append(h.Au)
			bu.append(h.bu)
			Ae.append(h.Ae)
			be.append(h.be)
		return cls(
			np.concatenate(Au, axis = 0),
			np.concatenate(bu, axis = 0),
			np.concatenate(Ae, axis = 0),
			np.concatenate(be, axis = 0),
		)
	
	def to_inequalities(self) -> Tuple[np.ndarray, np.ndarray]:
		A = self.Au
		b = self.bu
		if len(self.Ae) > 0:
			A = np.concatenate([A, self.Ae, -self.Ae], axis = 0)
			b = np.concatenate([b, self.be, -self.be], axis = 0)
		return A, b
	
	def to_cdd_polyhedron(self, *, number_type = None) -> cdd.Polyhedron:
		M = np.concatenate([
			np.concatenate([self.be[:,None], -self.Ae], axis = 1),
			np.concatenate([self.bu[:,None], -self.Au], axis = 1),
		], axis = 0)
		mat = cdd.Matrix(M, number_type = number_type or 'float')
		mat.lin_set = frozenset(range(self.Ae.shape[0]))
		mat.rep_type = cdd.RepType.INEQUALITY
		return cdd.Polyhedron(mat)
	
	def to_v(self, *, number_type = None) -> VRepr:
		mat = self.to_cdd_polyhedron(number_type = number_type).get_generators()
		return VRepr.from_cdd_matrix(mat, dtype = self.dtype)
	
	@classmethod
	def from_cdd_matrix(cls, mat: cdd.Matrix, *, dtype: Optional[np.dtype] = None) -> HRepr:
		if dtype is None:
			dtype = np.float64
		
		A = []
		b = []
		
		for i in range(mat.row_size):
			bi, *nAi = mat[i]
			A.append(nAi)
			b.append(bi)
		A = -np.array(A, dtype = dtype)
		b = np.array(b, dtype = dtype)
		
		e = np.zeros(mat.row_size, dtype = np.bool_)
		e[list(mat.lin_set)] = True
		u = ~e
		
		return cls(A[u], b[u], A[e], b[e])

class VRepr:
	def __init__(
		self,
		Vc: Optional[np.ndarray] = None,
		Vn: Optional[np.ndarray] = None,
		Vl: Optional[np.ndarray] = None,
	) -> None:
		V = _or(Vc, Vn, Vl)
		assert V is not None
		self.dim = V.shape[1]
		self.dtype = V.dtype
		
		if Vc is None:
			Vc = np.empty((0, self.dim), dtype = V.dtype)
		if Vn is None:
			Vn = np.empty((0, self.dim), dtype = V.dtype)
		if Vl is None:
			Vl = np.empty((0, self.dim), dtype = V.dtype)
		
		self.Vc = Vc
		self.Vn = Vn
		self.Vl = Vl
	
	def to_finite_basis(self) -> Tuple[np.ndarray, np.ndarray]:
		Vn = self.Vn
		if len(self.Vl) > 0:
			Vn = np.concatenate([Vn, self.Vl, -self.Vl])
		return self.Vc, Vn
	
	def to_cdd_polyhedron(self, *, number_type = None) -> cdd.Polyhedron:
		M = np.concatenate([self.Vl, self.Vc, self.Vn], axis = 0)
		M = np.concatenate([np.zeros_like(M[:,:1]), M], axis = 1)
		M[len(self.Vl):len(self.Vl) + len(self.Vc), 0] = 1
		mat = cdd.Matrix(M, number_type = number_type or 'float')
		mat.lin_set = frozenset(range(len(self.Vl)))
		mat.rep_type = cdd.RepType.GENERATOR
		return cdd.Polyhedron(mat)
	
	def to_h(self, *, number_type = None) -> HRepr:
		mat = self.to_cdd_polyhedron(number_type = number_type).get_inequalities()
		return HRepr.from_cdd_matrix(mat, dtype = self.dtype)
	
	@classmethod
	def from_cdd_matrix(cls, mat: cdd.Matrix, *, dtype: Optional[np.dtype] = None) -> VRepr:
		if dtype is None:
			dtype = np.float64
		
		Vc = []
		Vn = []
		Vl = []
		
		for i in range(mat.row_size):
			t, *row = mat[i]
			V = (Vc if t > 0.5 else (Vl if i in mat.lin_set else Vn))
			V.append(row)
		
		dim = mat.col_size - 1
		if Vc:
			Vc = np.array(Vc, dtype = dtype)
		else:
			Vc = np.empty((0, dim), dtype = dtype)
		if Vn:
			Vn = np.array(Vn, dtype = dtype)
		else:
			Vn = np.empty((0, dim), dtype = dtype)
		if Vl:
			Vl = np.array(Vl, dtype = dtype)
		else:
			Vl = np.empty((0, dim), dtype = dtype)
		
		return cls(Vc, Vn, Vl)

def _or(*mats: np.ndarray) -> Optional[np.ndarray]:
	for m in mats:
		if m is not None:
			return m
	return None
