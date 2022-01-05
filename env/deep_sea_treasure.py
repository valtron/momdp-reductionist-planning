import enum
import numpy as np

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
	for (row, col), value in _TREASURES.items():
		steps = row + col
		# calculating second objective
		obj2 = 0
		discount = 1
		for i in range(steps):
			obj2 -= discount
			discount *= gamma
		obj1 = value * discount
		pf_vertices.append([obj1, obj2])
	return np.array(pf_vertices)

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

name = __name__
gamma = 0.98
feature_ranges = np.array([
	[0, 11],
	[0, 10],
], dtype = np.float32)
k = 2
num_actions = 4
deterministic_start = True
deterministic_transitions = True
deterministic = deterministic_transitions and deterministic_start
states = [None] + _OCCUPIABLE
min_return = np.array([0, -1/(1-gamma)], dtype = np.float32)
max_return = np.max(true_pareto_front(), axis = 0).astype(np.float32)

def sample_transition(rng, s, a):
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

def sample_start(rng):
	return (0, 0)

def sample_state(rng):
	return rng.choice(_OCCUPIABLE)

def terminal_value(s):
	if s is None:
		return np.array([0, 0], dtype = np.float32)
	return None

class _Action(enum.IntEnum):
	Up = 0
	Down = 1
	Left = 2
	Right = 3

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
