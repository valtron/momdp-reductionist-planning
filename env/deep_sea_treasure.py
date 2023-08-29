import sys
import numpy as np
import dualdesc as dd
from common.so_solver import TabularSolver

# "Deep Sea Treasure" environment from
# "A distributional view on multi-objective policy optimization"
# https://arxiv.org/pdf/2005.07513.pdf

name = "DeepSeaTreasure"
gamma = 0.98
horizon = np.inf
objective_names = ["Time Penalty", "Treasure Value"]
k = 2

def pareto_front_vertices():
	pf = []
	# one vertex corresponding to each treasure location
	for (row, col), value in _TREASURES.items():
		steps = row + col
		# calculating second objective
		obj2 = 0
		discount = 1
		for i in range(steps):
			obj2 -= discount
			discount *= gamma
		obj1 = value * discount
		pf.append([obj1, obj2])
	return np.array(pf)

_TREASURES = {
	( 1, 0):  0.7,
	( 2, 1):  8.2,
	( 3, 2): 11.5,
	( 4, 3): 14.0,
	( 4, 4): 15.1,
	( 4, 5): 16.1,
	( 7, 6): 19.6,
	( 7, 7): 20.3,
	( 9, 8): 22.4,
	(10, 9): 23.7,
}

_OCCUPIABLE = sorted(set.union(*(
	{ (y, x) for y in range(y_max + 1) }
	for y_max, x in _TREASURES.keys()
)))

num_actions = 4
min_return = np.array([0, -1/(1-gamma)], dtype = np.float32)
max_return = np.array([max(_TREASURES.values()), 0], dtype = np.float32)

def transition(s, a):
	if s is None:
		return np.array([0, 0], dtype = np.float32), s
	y, x = s
	value = _TREASURES.get((y, x))
	if value is not None:
		return np.array([value, 0]), None
	sp = _grid_move(y, x, a)
	if sp not in _OCCUPIABLE:
		sp = s
	return np.array([0, -1]), sp

start = (0, 0)

def _grid_move(y, x, a):
	if a == _Action.Up:
		if y >  0: y -= 1
	elif a == _Action.Down:
		if y < 10: y += 1
	elif a == _Action.Left:
		if x >  0: x -= 1
	else:
		if x <  9: x += 1
	return y, x

class _Action:
	Up = 0
	Down = 1
	Left = 2
	Right = 3

solver = TabularSolver(sys.modules[__name__])
