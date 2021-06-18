import enum
import numpy as np

from common import get_pareto_optimal, mdp_to_matrices

# "Bonus World" environment from
# "Softmax exploration strategies for multiobjective reinforcement learning"
# https://www.sciencedirect.com/science/article/abs/pii/S0925231217310974

gamma = 0.95

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
		pf[i,:2] *= gamma**(n_steps - 1)
		pf[i,2] = -(1 - gamma**n_steps)/(1 - gamma)
	
	pf = pf[sorted(get_pareto_optimal(pf))]
	return pf

def get_mdp():
	return mdp_to_matrices(BonusWorld())

class BonusWorld:
	def __init__(self):
		self.start_state = (1, 1, 1)
		self.actions = [Action.Up, Action.Down, Action.Left, Action.Right]
	
	def transition(self, s, a):
		y, x, bonus_multiplier = s
		
		if (y, x) in OBJECTIVES:
			# Terminal state
			return s, (0, 0, 0)
		
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
		
		return (yp, xp, bonus_multiplier), r

class Action(enum.IntEnum):
	Up = 0
	Down = 1
	Left = 2
	Right = 3

def grid_move(y, x, a):
	if a is Action.Up:
		if y > 1: y -= 1
	elif a is Action.Down:
		if y < 9: y += 1
	elif a is Action.Left:
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