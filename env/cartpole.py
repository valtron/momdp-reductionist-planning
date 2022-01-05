# Cartpole environment
# From https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
# MOMDP based on CMDP from https://arxiv.org/abs/2011.05869

import numpy as np
from numba import jit

_MASS_CART = 1
_MASS_POLE = 0.1
_MASS_TOTAL = _MASS_CART + _MASS_POLE
_HALF_LENGTH = 0.5
_POLEMASS_LENGTH = _MASS_POLE * _HALF_LENGTH
_FORCE_MAG = 10
_DT = 0.02
_FAIL_POS = 2.4
_FAIL_ANGLE = np.deg2rad(12)
_G = 9.8
_PENALTY_AREAS = np.array([
	[-2.4, -2.2],
	[-1.3, -1.1],
	[-0.1,  0.1],
	[ 1.1,  1.3],
	[ 2.2,  2.4],
], dtype = np.float32)
_PENALTY_ANGLE = np.deg2rad(6)

name = __name__
gamma = 1
max_steps = 200
feature_ranges = np.array([
	[-_FAIL_POS, _FAIL_POS],
	[-3, 3],
	[-_FAIL_ANGLE, _FAIL_ANGLE],
	[-3, 3],
], dtype = np.float32)
k = 3
num_actions = 2
deterministic_start = False
deterministic_transitions = True
deterministic = deterministic_transitions and deterministic_start
min_return = np.array([0, -1, -1], dtype = np.float32) * max_steps
max_return = np.array([1,  0,  0], dtype = np.float32) * max_steps

def sample_start(rng):
	s = rng.uniform(-0.05, 0.05, size = len(feature_ranges)).astype(np.float32)
	return s

def sample_state(rng):
	s = rng.uniform(size = len(feature_ranges)).astype(np.float32)
	s *= feature_ranges[:,1] - feature_ranges[:,0]
	s += feature_ranges[:,0]
	return s

def terminal_value(s):
	if _is_terminal(s):
		return np.array([0, 0, 0], dtype = np.float32)
	return None

def sample_transition(rng, s, a):
	return _sample_transition_impl(s, a)

def _is_terminal(s):
	theta = s[2]
	return not (-_FAIL_ANGLE <= theta <= _FAIL_ANGLE)

@jit(nopython = True)
def _sample_transition_impl(s, a):
	x, x_dot, theta, theta_dot = s
	
	if not (-_FAIL_ANGLE <= theta <= _FAIL_ANGLE):
		return np.array([0, 0, 0], dtype = np.float32), s
	
	force = _FORCE_MAG * (1 if a == 1 else -1)
	costheta = np.cos(theta)
	sintheta = np.sin(theta)
	
	temp = (
		force + _POLEMASS_LENGTH * theta_dot**2 * sintheta
	) / _MASS_TOTAL
	thetaacc = (_G * sintheta - costheta * temp) / (
		_HALF_LENGTH * (4 / 3 - _MASS_POLE * costheta**2 / _MASS_TOTAL)
	)
	xacc = temp - _POLEMASS_LENGTH * thetaacc * costheta / _MASS_TOTAL
	
	x += _DT * x_dot
	x_dot += _DT * xacc
	theta += _DT * theta_dot
	theta_dot += _DT * thetaacc
	
	if x < -_FAIL_POS:
		x = -_FAIL_POS
		x_dot = 0
	elif x > _FAIL_POS:
		x = _FAIL_POS
		x_dot = 0
	
	sp = np.array([x, x_dot, theta, theta_dot], dtype = np.float32)
	
	pos_penalty = 0
	if np.any((x >= _PENALTY_AREAS[:,0]) & (x <= _PENALTY_AREAS[:,1])):
		pos_penalty = -1
	
	angle_penalty = 0
	if abs(theta) > _PENALTY_ANGLE:
		angle_penalty = -1
	
	return np.array([1, pos_penalty, angle_penalty], dtype = np.float32), sp
