from collections import namedtuple
import numpy as np

# "Deep Sea Treasure" environment from
# "A distributional view on multi-objective policy optimization"
# https://arxiv.org/pdf/2005.07513.pdf

gamma = 0.98

def get_mdp():
	# Parameters defining the deep sea treasure environment
	num_rows = 11
	num_cols = 10
	num_states = num_rows * num_cols
	num_actions = 4
	
	# transitions[sa][s'] = Pr(s' | s,a)
	transitions = np.zeros((num_states * num_actions, num_states))
	
	# rewards[sa, i] = ith r(s,a)
	rewards = np.zeros((num_states * num_actions, 2))
	
	# start[i]: probability that start state is i
	start = np.zeros(num_states)
	start[0] = 1
	
	# blocked[r][c] = true if (r,c) is an inaccessible cell
	blocked = np.zeros((num_rows, num_cols), dtype = np.bool_)
	for t in treasures:
		blocked[t.pos_row+1:,t.pos_col] = True
	
	for sa in range(num_states * num_actions):
		s, a = divmod(sa, num_actions)
		r, c = state_to_cell(s, num_rows, num_cols)
		assert s == cell_to_state(r, c, num_rows, num_cols)
		if blocked[r][c]:
			transitions[sa][s] = 1
			continue
		rewards[sa][0] = treasure_value_at_location(r,c,treasures)
		# if s is treasure, move to blocked position to indicate termination
		rp = r
		cp = c
		if rewards[sa][0] > 0:
			rewards[sa][1] = 0
			if r < num_rows - 1:
				rp = r + 1
			else:
				assert c == num_cols-1, "Something is wrong, this should be corner cell"
				cp = c - 1
			assert blocked[rp, cp], "Something is wrong, this should be blocked"
		else:
			rewards[sa][1] = -1
			if (a == 0 and r > 0): #up
				rp = r - 1
			if (a == 1 and c < (num_cols-1)): #right
				cp = c + 1
			if (a == 2 and r < (num_rows-1)): #down
				rp = r + 1
			if (a == 3 and c > 0): #left
				cp = c - 1
		sp = cell_to_state(rp, cp, num_rows, num_cols)
		transitions[sa][sp] = 1
	
	return transitions, rewards, start

def true_pareto_front():
	"""
		Output: list of vectors that are on the Pareto front of the deep sea
			treasure instance
	"""
	
	pf_vertices = []
	#one vertex corresponding to each treasure location
	for t in treasures:
		steps = t.pos_row + t.pos_col
		#calculating second objective
		obj2 = 0
		discount = 1
		for i in range(steps):
			obj2 -= discount
			discount *= gamma
		obj1 = t.value * discount
		pf_vertices.append([obj1, obj2])
	return np.array(pf_vertices)

Treasure = namedtuple('Treasure', ['pos_row', 'pos_col', 'value'])
treasures = [
	Treasure(1,0,0.7), Treasure(2,1,8.2), Treasure(3,2,11.5),
	Treasure(4,3,14.0), Treasure(4,4,15.1), Treasure(4,5,16.1),
	Treasure(7,6,19.6), Treasure(7,7,20.3), Treasure(9,8,22.4),
	Treasure(10,9,23.7),
]

def treasure_value_at_location(r, c, ts):
	"""
		Input: r: row index, c: column index, ts: list of treasures
		Output: value of treasure at cell (r,c). If no treasure, 0 is returned
	"""
	for t in ts:
		if (t.pos_row == r and t.pos_col == c):
			return t.value
	return 0

def cell_to_state(r, c, num_rows, num_cols):
	"""
		Input: r: row index, c: column index, rows: number of rows in grid,
			cols: number of columns in grid
		Output: The index of the state corresponding to the cell at row r and col c
	"""
	return r*num_cols + c

def state_to_cell(s, num_rows, num_cols):
	"""
		Input: s: index of state, rows: number of rows in grid,
			cols: number of columns in grid
		Output: (r,c) where r is the row of the cell corresponding to state s and c is
			the column
	"""
	return divmod(s, num_cols)
