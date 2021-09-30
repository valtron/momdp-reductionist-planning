import enum
import numpy as np

class Env:
	min_position = -1.2
	max_position = 0.6
	max_speed = 0.07
	goal_position = 0.5
	num_actions = 3
	force = 0.001
	gravity = 0.0025
	gamma = 0.999
	k = 3
	
	pos_range = (min_position, max_position)
	vel_range = (-max_speed, max_speed)
	feature_ranges = np.array([pos_range, vel_range], dtype = np.float32)
	deterministic = False
	
	@classmethod
	def sample_transition(cls, rng, s, a):
		p, v = s
		
		if p >= cls.goal_position:
			sp = s
			r = np.array([0, 0, 0], dtype = np.float32)
		else:
			v += (a - 1) * cls.force - cls.gravity * np.cos(3 * p)
			v = np.clip(v, -cls.max_speed, cls.max_speed)
			p = np.clip(p + v, cls.min_position, cls.max_position)
			if p <= cls.min_position:
				v = max(v, 0)
			sp = (p, v)
			r = np.array([
				-1, (-1 if a == Action.Backward else 0), (-1 if a == Action.Forward else 0)
			], dtype = np.float32)
		
		return r, sp
	
	@classmethod
	def sample_start(cls, rng):
		return (rng.uniform(-0.6, -0.4), 0)
	
	@classmethod
	def sample_state(cls, rng):
		mins = cls.feature_ranges[:, 0]
		maxs = cls.feature_ranges[:, 1]
		return rng.uniform(size = 2) * (maxs - mins) + mins
	
	@classmethod
	def terminal_value(cls, s):
		p, v = s
		if p >= cls.goal_position and v >= 0:
			return np.array([0, 0, 0], dtype = np.float32)
		return None

class Action(enum.IntEnum):
	Forward = 2
	Coast = 1
	Backward = 0
