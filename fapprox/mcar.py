from pathlib import Path

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import tilecoding
from env import mountain_car
from algo import nls
from common import deduplicate_and_sort, ScalarizedEnv, evaluate, estimated_pf_hull

import cacher

cached = cacher.Cacher(Path('.cache'))

def main():
	estimated_pf = find_mcar_pf()
	estimated_pf = deduplicate_and_sort(estimated_pf)
	ref = -np.ones(3) / (1 - mountain_car.Env.gamma)
	hull = estimated_pf_hull(ref, estimated_pf)
	
	normals = hull.equations[:,:3]
	idxs = np.max(normals, axis = 1) > 0
	normals = normals[idxs]
	normals = normals / np.max(normals, axis = 1)[:,None]
	simplices = hull.simplices[idxs]
	fcols = np.concatenate([normals, np.ones((len(normals), 1)) * 0.7], axis = 1)
	
	fig = plt.figure(constrained_layout = True)
	
	ax = fig.add_subplot(projection = '3d')
	ax.set_xlabel("Time penalty")
	ax.set_ylabel("Backward penalty")
	ax.set_zlabel("Forward penalty")
	ax.plot(estimated_pf[:, 0], estimated_pf[:, 1], estimated_pf[:, 2], 'ko')
	ax.view_init(30, 45)
	
	obj = ax.plot_trisurf(
		hull.points[:,0], hull.points[:,1], hull.points[:,2], triangles = simplices,
		linewidth = 0, antialiased = False,
	)
	obj.set_fc(fcols)
	plt.show()

@cached('mcar_pf.pickle')
def find_mcar_pf():
	env = mountain_car.Env
	epsilon = 1000 * np.ones(env.k)
	def comb(w):
		return solve_scalarized(env, w)
	return nls.estimate_pareto_front(comb, epsilon, progress = tqdm)

def solve_scalarized(env_mo, w: np.ndarray):
	w = w / np.sum(w)
	
	env = ScalarizedEnv(env_mo, w)
	rng = np.random.default_rng(0)
	
	agent = QLearningAgent(env)
	for ep in tqdm(range(300)):
		s = env.sample_state(rng)
		t = 0
		while env.terminal_value(s) is None:
			if t >= 1000:
				break
			a = agent.act(rng, s)
			r, sp = env.sample_transition(rng, s, a)
			agent.update(s, a, r, sp)
			s = sp
			t += 1
	
	return evaluate(agent.get_policy(), env_mo)

class QLearningAgent:
	def __init__(self, env):
		self.env = env
		self.tc = tilecoding.TileCoder(env.feature_ranges, 8, 4)
		self.weights_q = np.zeros((self.tc.output_size, env.num_actions))
		self.weights_n = np.zeros((self.tc.output_size, env.num_actions))
	
	def get_policy(self):
		weights_q = self.weights_q.copy()
		tc = self.tc
		def policy(rng, s):
			s = np.array(s)
			tiles = tc.tiles(s)
			qs = np.mean(weights_q[tiles], axis = 0)
			return np.argmax(qs)
		policy.deterministic = True
		return policy
	
	def act(self, rng, s):
		s = np.array(s)
		qs = np.mean(self.weights_q[self.tc.tiles(s)], axis = 0)
		return np.argmax(qs)
	
	def update(self, s, a, r, sp):
		s = np.array(s)
		tiles = self.tc.tiles(s)
		self.weights_n[tiles, a] += 1
		n = np.mean(self.weights_n[tiles, a])
		
		bootstrap = self.env.terminal_value(sp)
		if bootstrap is None:
			sp = np.array(sp)
			qsp = np.mean(self.weights_q[self.tc.tiles(sp)], axis = 0)
			bootstrap = np.max(qsp)
		
		H = 1/(1 - self.env.gamma)
		alpha = (H + 1) / (H + n)
		target = r + self.env.gamma * bootstrap
		err = target - np.mean(self.weights_q[tiles, a])
		self.weights_q[tiles, a] += alpha * err

if __name__ == '__main__':
	main()
