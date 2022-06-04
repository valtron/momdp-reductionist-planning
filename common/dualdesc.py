from typing import Optional, Any, List

import numpy as np

import os
if os.name == 'nt':
	import ctypes
	from pathlib import Path
	# TODO: figure out why the default `winmode` flags don't work
	ctypes.CDLL(str(Path(np.__path__[0]).parent / 'pyparma/ppl.cp39-win_amd64.pyd'), winmode = 0)

import pyparma

NDArray = Any
DType = Any
DefaultScale = 10000

class Polytope:
	@classmethod
	def Empty(cls, dim: int, *, scale: float = None, dtype: Optional[DType] = None):
		return cls.FromGenerators(np.empty((0, dim), dtype = dtype), scale = scale)
	
	@classmethod
	def Universe(cls, dim: int, *, scale: float = None, dtype: Optional[DType] = None):
		return cls.FromHalfspaces(np.empty((0, dim), dtype = dtype), np.empty((0,), dtype = dtype), scale = scale)
	
	@classmethod
	def FromHalfspaces(cls, A: NDArray, b: NDArray, *, scale: float = None):
		if scale is None:
			scale = DefaultScale
		dtype = A.dtype
		
		# (Ax <= b) => (0 <= b + (-A)x)
		impl = pyparma.Constraint_System()
		A = _quantize(A, scale)
		b = _quantize(b, scale**2)
		for Ai, bi in zip(A, b):
			ex = pyparma.Linear_Expression(Ai, 0)
			impl.insert(ex <= bi)
		
		return cls(pyparma.C_Polyhedron(impl), scale = scale, dtype = dtype)
	
	@classmethod
	def FromGenerators(cls, V: Optional[NDArray] = None, R: Optional[NDArray] = None, *, scale: float = None):
		if scale is None:
			scale = DefaultScale
		
		M = (R if V is None else V)
		if M is None:
			raise ValueError("vertices and rays cannot both be omitted")
		dtype = M.dtype
		dim = M.shape[-1]
		
		if V is None:
			V = np.empty((0, dim), dtype = dtype)
		if R is None:
			R = np.empty((0, dim), dtype = dtype)
		
		impl = pyparma.Generator_System()
		V = _quantize(V, scale)
		R = _quantize(R, scale)
		for Vi in V:
			impl.insert(pyparma.point(pyparma.Linear_Expression(Vi, 0), 1))
		for Ri in R:
			impl.insert(pyparma.ray(pyparma.Linear_Expression(Ri, 0)))
		
		return cls(pyparma.C_Polyhedron(impl), scale = scale, dtype = dtype)
	
	def __init__(self, impl: pyparma.C_Polyhedron, *, scale: float = None, dtype: Optional[DType] = None):
		if scale is None:
			scale = DefaultScale
		if scale <= 0:
			raise ValueError("scale must be positive", scale)
		if dtype is None:
			dtype = np.float64
		self.impl = impl
		self.dim = impl.space_dimension()
		self.dtype = dtype
		self.scale = scale
	
	def add_halfspace(self, A, b):
		# P
		# = { x | Ax <= b }
		# = conv(V) + nonneg(R)
		# scale P
		# = { scale x | Ax <= b }
		# = { x | A <= scale b }
		# = conv(scale V) + nonneg(R)
		
		if len(A.shape) == 1:
			A = A[None,:]
			b = np.broadcast_to(b, (1,))
		A = _quantize(A, self.scale)
		b = _quantize(b, self.scale**2)
		for Ai, bi in zip(A, b):
			self.impl.add_constraint(pyparma.Linear_Expression(Ai, 0) <= bi)
	
	def add_point(self, V):
		if len(V.shape) == 1:
			V = V[None,:]
		V = _quantize(V, self.scale)
		for Vi in V:
			self.impl.add_generator(pyparma.point(pyparma.Linear_Expression(Vi, 0), 1))
	
	def add_ray(self, R):
		if len(R.shape) == 1:
			R = R[None,:]
		R = _quantize(R, self.scale)
		for Ri in R:
			self.impl.add_generator(pyparma.ray(pyparma.Linear_Expression(Ri, 0)))
	
	def get_inequalities(self):
		A = []
		b = []
		for c in self.impl.minimized_constraints():
			A.append(c.coefficients())
			b.append(c.inhomogeneous_term() / self.scale)
		
		if A:
			A = np.array(A, dtype = self.dtype)
			# PPL constraints are `coeffs @ x + b >= 0` so `A = -coeffs`
			np.negative(A, out = A)
			norms = np.linalg.norm(A, axis = 1)
			A /= norms[:,None]
			b = np.array(b, dtype = self.dtype)
			b /= norms
		else:
			A = np.empty((0, self.dim), dtype = self.dtype)
			b = np.empty((0,), dtype = self.dtype)
		
		return A, b
	
	def get_generators(self):
		V = []
		R = []
		for g in self.impl.minimized_generators():
			coef = np.array(g.coefficients(), dtype = self.dtype)
			if g.is_point():
				V.append(coef / g.divisor())
			else:
				R.append(coef)
		if V:
			V = np.array(V, dtype = self.dtype)
			V /= self.scale
		else:
			V = np.empty((0, self.dim), dtype = self.dtype)
		if R:
			R = np.array(R, dtype = self.dtype)
			R /= np.linalg.norm(R, axis = 1, keepdims = True)
		else:
			R = np.empty((0, self.dim), dtype = self.dtype)
		return V, R
	
	def copy(self):
		return Polytope(pyparma.C_Polyhedron(self.impl), scale = self.scale, dtype = self.dtype)

def _quantize(m, scale):
	m = m * scale
	np.around(m, out = m)
	return m.astype(np.int64)
