import sys
import numpy as np
import dualdesc as dd
from common.so_solver import TabularSolver

# "Bonus World" environment from
# "Softmax exploration strategies for multiobjective reinforcement learning"
# https://www.sciencedirect.com/science/article/abs/pii/S0925231217310974

name = "BonusWorld"
gamma = 0.95
horizon = np.inf
objective_names = ["Objective 1", "Objective 2", "Time Penalty"]
k = 3

def pareto_front_vertices():
	# Undiscounted returns
	pf = np.array([
		# The commented out ones are deterministic PO outcomes,
		# but dominated by a stochastic policy (gamma = 0.95)
		[ 1,  9,  -8],
		#[ 3,  9, -10],
		#[ 5,  9, -12],
		#[ 7,  9, -14],
		[10, 18, -14],
		[14, 18, -16],
		[18, 18, -18],
		[18, 14, -16],
		[18, 10, -14],
		#[ 9,  9, -16],
		#[ 9,  7, -14],
		#[ 9,  5, -12],
		#[ 9,  3, -10],
		[ 9,  1,  -8],
	], dtype = np.float64)
	
	# Apply discounting
	for i in range(len(pf)):
		n_steps = -pf[i,2]
		pf[i,:2] *= gamma**(n_steps - 1)
		pf[i,2] = -(1 - gamma**n_steps)/(1 - gamma)
	
	return pf

num_actions = 4
min_return = np.array([0, 0, -1/(1 - gamma)], dtype = np.float32)
max_return = np.array([18, 18, 0], dtype = np.float32)

def transition(s, a):
	y, x, bonus_multiplier = s
	
	if (y, x) in _OBJECTIVES:
		# Terminal state
		return np.array([0, 0, 0], dtype = np.float32), s
	
	yp, xp = _grid_move(y, x, a)
	
	# Walls
	if (yp, xp) in _WALLS:
		yp, xp = (y, x)
	
	# Pits
	if (yp, xp) in _PITS:
		yp, xp = (1, 1)
	
	# Bonus
	if (yp, xp) == (4, 4):
		bonus_multiplier = 2
	
	if (yp, xp) in _OBJECTIVES:
		r = [yp * bonus_multiplier, xp * bonus_multiplier, -1]
	else:
		r = [0, 0, -1]
	
	return np.array(r, dtype = np.float32), (yp, xp, bonus_multiplier)

start = (1, 1, 1)

def _grid_move(y, x, a):
	if a == _Action.Up:
		if y > 1: y -= 1
	elif a == _Action.Down:
		if y < 9: y += 1
	elif a == _Action.Left:
		if x > 1: x -= 1
	else:
		if x < 9: x += 1
	return y, x

class _Action:
	Up = 0
	Down = 1
	Left = 2
	Right = 3

_WALLS = {
	(3, 3), (3, 4), (4, 3),
}
_PITS = {
	(8, 2), (8, 4), (8, 6),
	(2, 8), (4, 8), (6, 8),
}
_OBJECTIVES = {
	(9, 1), (9, 3), (9, 5), (9, 7), (9, 9),
	(1, 9), (3, 9), (5, 9), (7, 9),
}
_OCCUPIABLE = {
	(y, x, b)
	for y in range(1, 9 + 1)
	for x in range(1, 9 + 1)
	for b in [1, 2]
	if not ((y, x) in _WALLS or (y, x) in _PITS)
}

solver = TabularSolver(sys.modules[__name__])
