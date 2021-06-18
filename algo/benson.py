import numpy as np
from numpy.linalg import norm
from scipy.optimize import linprog
from pypoman.duality import compute_polytope_vertices

from common import CountCalls


def estimate_pareto_front(C, A, b, eps=0):
	"""
	Implements an exact and approximate version of Benson's outer approximation algorithm
	   (exact, Benson 1998) https://link.springer.com/article/10.1023/A:1008215702611
	   (approx, Shao 2008) https://link.springer.com/article/10.1007/s00186-008-0220-2

	A multiobjective linear program (MOLP):
	   max  Cx
	   s.t. Ax=b, x>=0

	If eps is provided, we compute (eps) inner and outer approximations of the Pareto front of the
	outcome polytope, as in (Shao 2008).
	Otherwise, we compute the vertices of the Pareto front.

	This implementation closely follows the description in
	https://link.springer.com/article/10.1023/A:1008215702611

	Parameters
	--------------------
	C: The objective matrix. Each row is is one objective vector.
	A: The constraint matrix.
	b: The constrain vector.
	eps: When provided, we compute (eps) inner and outer approximations of
	     the Pareto front (set of nondominated outcome points).
	     Otherwise we compute the nondominated vertices of it.

	Also returns the total # of calls to the LP solver used.
	"""

	# to count the number of times linprog is called
	c_linprog = CountCalls(linprog)

	k, n = C.shape
	e = np.ones(k)
	y_AI = get_anti_ideal_point(C, A, b, c_linprog)
	y_hat = y_AI - 1

	res = Qw(y_AI, C, A, b, c_linprog)
	assert res.success, res.message
	z_star = res.x[:-1]
	y_interior = 0.5 * y_AI + 0.5 * (C @ z_star)

	# find supporting hyperplane normal to e
	# TODO is this the same as max <e,y> for y in target approx?? (benson pg 7)
	res = c_linprog(-e @ C, A_eq=A, b_eq=b)
	assert res.success, res.message
	beta = -res.fun
	
	# halfspace constraints for the first outer approximation
	
	# the initial supporting hyperplane normal to e
	supporting_hp_normal = np.ones(k)
	supporting_hp_const = np.array([beta])
	
	# the axis aligned hyperplanes (each defined by point y_hat and a normal vector perpendicular to an axis,
	# pointing away from it)
	constraint_matrix = -np.eye(k)
	constraint_vector = constraint_matrix @ y_hat
	
	constraint_matrix = np.vstack((supporting_hp_normal, constraint_matrix))
	constraint_vector = np.hstack((supporting_hp_const, constraint_vector))
	
	I = []
	O = []
	
	def close_to_any(x, points, dist):
		return any(norm(x - p) < dist for p in points)
	
	while True:
		vertices = compute_polytope_vertices(constraint_matrix, constraint_vector)
		
		# Step 1
		vertex_found = False
		for v in vertices:
			if close_to_any(v, O, 1e-6):
				continue
			if close_to_any(v, I, 1e-6):
				continue
			if is_in_target(v, C, A, b, c_linprog):
				I.append(v)
				continue
			vertex_found = True
			break
		
		if not vertex_found:
			break
		
		# Step 2
		boundary_point = get_point_on_target_boundary(v, y_interior, C, A, b, c_linprog)
		
		# Step 3
		if eps > 0 and norm(boundary_point - v) < eps:
			O.append(v)
			I.append(boundary_point)
			# Back to Step 1
			continue
		
		hp_normal, hp_const = get_supporting_hyperplane(boundary_point, C, A, b, c_linprog)
		constraint_matrix = np.vstack((constraint_matrix, hp_normal))
		constraint_vector = np.hstack((constraint_vector, hp_const))
	
	# Step 5
	if eps > 0:
		vertices_minus_O = [v for v in vertices if not close_to_any(v, O, 1e-6)]
		inner_approx = vertices_minus_O + I
		inner_approx = np.array(inner_approx)
		outer_approx = np.array(vertices)
		return inner_approx, outer_approx, c_linprog.count(linprog)
	else:
		# separate nondomindated vertices from others
		nondominated_vertices = np.array([v for v in vertices if np.all(v > y_hat)])
		other_vertices = np.array([v for v in vertices if not close_to_any(v, nondominated_vertices, 1e-6)])
		return nondominated_vertices, other_vertices, c_linprog.count(linprog)


def get_anti_ideal_point(C, A, b, lp_solver):
	"""
	Computes y_AI, where (y_AI)_i = min{y_i | (y_1,..,y_k)^T in Y}) where
	Y = {y in R^k | y=Cx for some x in X} and  X = {x in R^n | Ax=b, x>=0},
	by solving the LPs:

	min  <e_i, Cx>
	s.t. Ax=b, x>=0
	for i=1,..k
	"""
	# TODO is (y_AI)_i to just the min of row i of C
	k, n = C.shape
	y_AI = np.zeros(k)
	for i in range(k):
		res = lp_solver(C[i], A_eq=A, b_eq=b, method='highs-ds')
		assert res.success, res.message
		y_AI[i] = res.fun
	return y_AI


def Qw(w, C, A, b, lp_solver):
	"""
	max  t
	s.t. Cx - et >= w
		 Ax = b
		x >= 0, t >= 0
	where e is a vector of ones.
	"""
	# variables x' = (x, t)^T
	k, n = C.shape
	c = np.zeros(n + 1)
	c[n] = 1.0

	A_eq = np.column_stack((A, np.zeros(A.shape[0])))
	A_ub = np.column_stack((C, -np.ones(k)))

	res = lp_solver(-c, A_eq=A_eq, b_eq=b, A_ub=-A_ub, b_ub=-w, method='highs-ds')
	return res


def QDw(w, C, A, b, lp_solver):
	"""
	min  -<w, u> + <b, v>
	s.t. -u^T C + v^T A >= 0,
		 <e, lambda> >=1, u >= 0
	"""
	# variables (u, v^+ v^-)^T
	k, n = C.shape
	upper = np.hstack((-C.T, A.T, -A.T))
	lower = np.hstack((np.ones(k), np.zeros(A.shape[0]), np.zeros(A.shape[0])))
	A_ub = np.vstack((upper, lower))
	b_ub = np.hstack((np.zeros(n), np.ones(1)))

	c = np.hstack((-w, b, -b))
	res = lp_solver(c, A_ub=-A_ub, b_ub=-b_ub, method='highs-ds')
	assert res.success, res.message
	return res


def is_in_target(y, C, A, b, lp_solver):
	"""
	Test whether y is in Y' (target outer approximation)
	"""
	res = Qw(y, C, A, b, lp_solver)
	status = res.status
	if status == 0 or status == 3:
		# solution found or looks unbounded
		return True
	elif status == 2:
		# infeasible
		return False
	else:
		raise RuntimeError(f'Failed to determine feasibility: {res.message}')


def get_point_on_target_boundary(y, y_interior, C, A, b, lp_solver):
	"""
	Find the unique point on both the boundary of target outer approximation Y' and
	the segment [y_interior, y], by solving the LP:
	max  t
	s.t. y_interior + t(y - y_interior) <= Cx
		 Ax = b
		x >= 0, t >= 0
	where e is a vector of ones.
	"""
	# variables x' = (t, x)^T
	k, n = C.shape
	c = np.zeros(n + 1)
	c[0] = 1.0

	# Ax=b -> [0 A]x' = b
	A_eq = np.column_stack((np.zeros(A.shape[0]), A))

	# q + t(y-y_int) <= Cx -> [y_int-y C]x' >= y_int
	A_ub = np.column_stack((y_interior - y, C))
	b_ub = y_interior
	# b_ub = q

	res = lp_solver(-c, A_eq=A_eq, b_eq=b, A_ub=-A_ub, b_ub=-b_ub, method='highs-ds')
	assert res.success, res.message
	assert -res.fun <= 1
	w = y_interior + -res.fun * (y - y_interior)
	return w


def get_supporting_hyperplane(w, C, A, b, lp_solver):
	"""
	Return the halfspace constraint of the supporting hyperplane contianing w (a point on the boundary of Y')
	"""
	k, n = C.shape
	m = A.shape[0]
	res = QDw(w, C, A, b, lp_solver)
	assert res.success, res.message
	# u is a k vector, v+ and v- are m vectors
	u, v_plus, v_minus = res.x[0:k], res.x[k:k + m], res.x[k + m:]
	v = v_plus - v_minus
	hp_normal = u
	hp_constant = b @ v
	return hp_normal, hp_constant


def get_intersection_line_and_hyperplane(p1, p2, hp_normal, hp_point, eps=1e-6):
	"""
	Find intersection point between segment [p1, p2] and a plane.
	"""
	line_direction = p2 - p1
	dp = line_direction @ hp_normal

	if abs(dp) < eps:
		if (hp_point - p1) @ hp_normal == 0:
			raise RuntimeError('Line contained in hyperplane.')
		else:
			raise RuntimeError('Line and hyperplane are parallel.')

	d = ((hp_point - p1) @ hp_normal) / dp
	intersection_point = p1 + d * line_direction

	if norm(intersection_point - p1) > norm(p2 - p1):
		raise RuntimeError('Intersection point outside of segment.')

	return intersection_point


def main():
	# Example 9.4.1 from 'Multiobjective Linear Programming' Dinh The Luc
	C = np.array([[1, 3, 2], [1, 1, 2]])
	A = np.array([[1, 1, 1]])
	b = np.array([1])

	nd_vertices, other_vertices, n_linprog_calls = estimate_pareto_front(C, A, b)
	print(f'Exact vertices:{nd_vertices}\nlinprog calls:{n_linprog_calls}')

	eps = 1.0
	inner_approx, outer_approx, n_linprog_calls = estimate_pareto_front(C, A, b, eps=eps)
	print(f'Inner {eps}-approx:{inner_approx}\nOuter {eps}-approx:{outer_approx}\nlinprog calls:{n_linprog_calls}')


if __name__ == '__main__':
	main()
