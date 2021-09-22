import enum
import numpy as np

import common

# "Deep Sea Treasure" environment from
# "A distributional view on multi-objective policy optimization"
# https://arxiv.org/pdf/2005.07513.pdf

def true_pareto_front():
	"""
		Output: list of vectors that are on the Pareto front of the deep sea
			treasure instance
	"""
	
	pf_vertices = []
	# one vertex corresponding to each treasure location
	for (row, col), value in TREASURES.items():
		steps = row + col
		# calculating second objective
		obj2 = 0
		discount = 1
		for i in range(steps):
			obj2 -= discount
			discount *= Env.gamma
		obj1 = value * discount
		pf_vertices.append([obj1, obj2])
	return np.array(pf_vertices)

class Env:
	gamma = 0.98
	feature_ranges = np.array([
		[0, 11],
		[0, 10],
	], dtype = np.float32)
	k = 2
	num_actions = 4
	deterministic = True
	
	@classmethod
	def sample_transition(cls, rng, s, a):
		if s is None:
			return np.array([0, 0], dtype = np.float32), s
		y, x = s
		value = TREASURES.get((y, x))
		if value is not None:
			return np.array([value, 0]), None
		return np.array([0, -1]), grid_move(y, x, a)
	
	@classmethod
	def sample_start(cls, rng):
		return (0, 0)
	
	@classmethod
	def sample_state(cls, rng):
		return rng.choice(OCCUPIABLE)
	
	@classmethod
	def terminal_value(cls, s):
		if s is None:
			return np.array([0, 0], dtype = np.float32)
		return None

class Action(enum.IntEnum):
	Up = 0
	Down = 1
	Left = 2
	Right = 3

def grid_move(y, x, a):
	if a == Action.Up:
		if y >  0: y -= 1
	elif a == Action.Down:
		if y < 10: y += 1
	elif a == Action.Left:
		if x >  0: x -= 1
	else:
		if x <  9: x += 1
	return y, x

TREASURES = {
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

OCCUPIABLE = sorted(set.union(*(
	{ (y, x) for y in range(y_max + 1) }
	for y_max, x in TREASURES.keys()
)))
