import enum
import numpy as np

import common

# "Bonus World" environment from
# "Softmax exploration strategies for multiobjective reinforcement learning"
# https://www.sciencedirect.com/science/article/abs/pii/S0925231217310974

def true_pareto_front():
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
		pf[i,:2] *= Env.gamma**(n_steps - 1)
		pf[i,2] = -(1 - Env.gamma**n_steps)/(1 - Env.gamma)
	
	pf = pf[sorted(common.get_pareto_optimal(pf))]
	return pf

class Env:
	gamma = 0.95
	feature_ranges = np.array([
		[1, 10],
		[1, 10],
		[1, 3],
	], dtype = np.float32)
	k = 3
	num_actions = 4
	deterministic = True
	
	@classmethod
	def sample_transition(cls, rng, s, a):
		y, x, bonus_multiplier = s
		
		if (y, x) in OBJECTIVES:
			# Terminal state
			return (0, 0, 0), s
		
		yp, xp = grid_move(y, x, a)
		
		# Walls
		if (yp, xp) in WALLS:
			yp, xp = (y, x)
		
		# Pits
		if (yp, xp) in PITS:
			yp, xp = (1, 1)
		
		# Bonus
		if (yp, xp) == (4, 4):
			bonus_multiplier = 2
		
		if (yp, xp) in OBJECTIVES:
			r = (yp * bonus_multiplier, xp * bonus_multiplier, -1)
		else:
			r = (0, 0, -1)
		
		return r, (yp, xp, bonus_multiplier)
	
	@classmethod
	def sample_start(cls, rng):
		return (1, 1, 1)
	
	@classmethod
	def sample_state(cls, rng):
		return rng.choice(OCCUPIABLE)
	
	@classmethod
	def terminal_value(cls, s):
		if (s[0], s[1]) in OBJECTIVES:
			return np.array([0, 0, 0], dtype = np.float32)
		return None

class Action(enum.IntEnum):
	Up = 0
	Down = 1
	Left = 2
	Right = 3

def grid_move(y, x, a):
	if a == Action.Up:
		if y > 1: y -= 1
	elif a == Action.Down:
		if y < 9: y += 1
	elif a == Action.Left:
		if x > 1: x -= 1
	else:
		if x < 9: x += 1
	return y, x

WALLS = {
	(3, 3), (3, 4), (4, 3),
}
PITS = {
	(8, 2), (8, 4), (8, 6),
	(2, 8), (4, 8), (6, 8),
}
OBJECTIVES = {
	(9, 1), (9, 3), (9, 5), (9, 7), (9, 9),
	(1, 9), (3, 9), (5, 9), (7, 9),
}
OCCUPIABLE = {
	(y, x, b)
	for y in range(1, 9 + 1)
	for x in range(1, 9 + 1)
	for b in [1, 2]
	if not ((y, x) in WALLS or (y, x) in PITS)
}
