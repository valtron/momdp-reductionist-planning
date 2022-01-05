from pathlib import Path
import itertools
from typing import Tuple

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import funcli

from algo import ols
from common import scalarization, cacher, featurizer, policy_iteration as pi, misc, hypervolume, pareto, dualdesc
from env import cartpole
import settings

cache = cacher.Cacher(settings.CACHE / 'fapprox')

def collect(method: str, runs: int, splits: int, index: int) -> None:
	assert method in { 'nls', 'ncs', 'ols' }
	load_into_cache(splits, index, runs)
	for run in range(runs):
		load_ols(run)

def graphs(runs: int) -> None:
	colors = {
		'nls': 'blue',
		'ncs': 'orange',
		'ols': 'red',
	}
	
	# NCS vs. NLS; order in [5, 7, 9]; #scal -> hypervolume
	for order in [5, 7, 9]:
		fig, ax = plt.subplots(constrained_layout = True, figsize = (3, 2))
		ax.set_xlabel("# scalarizations")
		ax.set_ylabel("Hypervolume")
		ax.set_ylim([6e6, 7.4e6])
		
		for method in ['nls', 'ncs']:
			color = colors[method]
			_, hvs = get_data(order, method, runs)
			hv_mean, hv_serr = aggregate_runs(hvs)
			xs = np.arange(1, len(hv_mean) + 1)
			ax.fill_between(xs, hv_mean - 2*hv_serr, hv_mean + 2*hv_serr, color = color, alpha = 0.2)
			ax.plot(xs, hv_mean, color = color, alpha = 0.5, label = method.upper())
		
		ax.legend(loc = 'lower right')
		ax.grid()
		fig.savefig('fig/fa-hv-ns-{}.pdf'.format(order))
	
	# NLS vs. OLS; order = 9; #scal -> hypervolume
	fig, ax = plt.subplots(constrained_layout = True, figsize = (4, 3))
	ax.set_xlabel("# scalarizations")
	ax.set_ylabel("Hypervolume")
	ax.set_ylim([6e6, 7.4e6])
	for method in ['nls', 'ols']:
		color = colors[method]
		_, hvs = get_data(9, method, runs)
		hv_mean, hv_serr = aggregate_runs(hvs)
		xs = np.arange(1, len(hv_mean) + 1)
		ax.fill_between(xs, hv_mean - 2*hv_serr, hv_mean + 2*hv_serr, color = color, alpha = 0.2)
		ax.plot(xs, hv_mean, color = color, alpha = 0.5, label = method.upper())
	ax.legend(loc = 'lower right')
	ax.grid()
	fig.savefig('fig/fa-hv-ols.pdf')
	
	# PF of run with median hypervolume
	
	def calc_Z(X, Y, values):
		Z = np.zeros_like(X)
		for i in range(X.shape[0]):
			for j in range(X.shape[1]):
				vs = values[(values[:,0] >= X[i,j]) & (values[:,1] >= Y[i,j]),2]
				Z[i,j] = (np.max(vs) if len(vs) > 0 else -200)
		return Z
	
	for method in ['nls', 'ncs', 'ols']:
		color = colors[method]
		if color == 'blue':
			color = 'tab:blue'
		fig = plt.figure(constrained_layout = True, figsize = (4, 3))
		ax = plt.axes(projection = '3d')
		ax.set_xlim([0, 200])
		ax.set_ylim([-200, 0])
		ax.set_zlim([-200, 0])
		
		values, hvs = get_data(9, method, runs)
		median_run = np.argsort([hvs_run[-1] for hvs_run in hvs])[runs // 2]
		values = values[median_run]
		values = values[pareto.is_non_dominated(values)]
		
		X = np.linspace(0, 200.1, 200)
		Y = np.linspace(-200, 0.1, 200)
		X, Y = np.meshgrid(X, Y)
		Z = calc_Z(X, Y, values)
		ax.plot_surface(X, Y, Z, linewidth = 0, shade = True, color = color, edgecolor = None, antialiased = False)
		ax.scatter(values[:,0], values[:,1], values[:,2], color = 'black', alpha = 1, zorder = 1, marker = '.', s = 10)
		
		ax.view_init(30, 45)
		fig.savefig('fig/fa-pf-{}.png'.format(method))

def get_data(order, method, runs):
	values = []
	hvs = []
	for seed in range(runs):
		values_run, hvs_run = get_data_run(order, method, seed)
		values.append(values_run)
		hvs.append(hvs_run)
	return values, hvs

def aggregate_runs(hvs):
	hvs_padded = []
	max_len = max(len(hvs_run) for hvs_run in hvs)
	for hvs_run in hvs:
		if len(hvs_run) < max_len:
			padding = np.tile(hvs_run[-1:], max_len - len(hvs_run))
			hvs_run = np.concatenate([hvs_run, padding], axis = 0)
		hvs_padded.append(hvs_run)
	hv_mean = np.mean(hvs_padded, axis = 0)
	hv_serr = np.std(hvs_padded, axis = 0, ddof = 1) / np.sqrt(len(hvs_padded))
	return hv_mean, hv_serr

@cache('{seed:02}/{method}-{order}.pickle')
def get_data_run(order, method, seed):
	env = cartpole
	eta_scale = 1
	m = 500
	segment_size = 100
	T = 300
	num_scalarizations = 50
	
	hvcalc = hypervolume.Hypervolume(env.min_return)
	
	if method == 'ols':
		values = solve_ols(env, eta_scale, m, order, T, segment_size, seed)
	else:
		if method == 'nls':
			space = scalarization.SpherePlusSpace(env.k)
			mode = 'linear'
		else:
			space = scalarization.OutcomeShadowSpace(env.min_return, env.max_return)
			mode = 'chebyshev'
		
		rng = np.random.default_rng(seed)
		scalarizations = cache.call('{:02}/{}-{}-{}.pickle'.format(
			seed, mode, 1000, 50,
		), scalarization.make_covering, 1000, 50, space, rng)[:50]
		
		values = np.stack([
			solve_scalarization(env, eta_scale, m, order, scalarization, T, segment_size, seed)[1][-1]
			for scalarization in scalarizations
		], axis = 0)
	
	hypervolumes = np.stack([
		hvcalc(values[:i])
		for i in range(len(values))
	], axis = 0)
	
	return values, hypervolumes

def solve_ols(env, eta_scale, m, order, T, segment_size, seed):
	impl = ols.OLS(env, np.ones(env.k), approx = True)
	
	def solve(w):
		scal = scalarization.LinearScalarization(w)
		_, values = solve_scalarization(env, eta_scale, m, order, scal, T, segment_size, seed)
		return values[-1]
	
	while True:
		wt, _ = ols.get_next_weight()
		if np.any(np.all(np.abs(wt - impl.W) < 0.001, axis = 1)):
			break
		yt = solve(wt)
		impl.W.append(wt)
		impl.Y.append(yt)
	
	return np.array(impl.Y, dtype = np.float64)

def plot_polyhedron(ax, c, o, col) -> None:
	from scipy import spatial
	
	k = c.shape[1]
	
	A, b = dualdesc.VRepr(c, -np.eye(k)).to_h().to_inequalities()
	A = np.concatenate([A, -np.eye(k)], axis = 0)
	b = np.concatenate([b, -o], axis = 0)
	c = dualdesc.HRepr(A, b).to_v().Vc
	
	if len(c) >= k:
		if len(c) <= k:
			simplices = np.arange(k)[None,:]
		else:
			hull = spatial.ConvexHull(c)# + np.random.default_rng(0).uniform(size = c.shape) * 0.01)
			normals = hull.equations[:,:3]
			simplices = hull.simplices
			idxs = np.min(normals, axis = 1) >= -1e-2
			simplices = simplices[idxs]
		if len(simplices) > 0:
			ax.plot_trisurf(
				c[:,0], c[:,1], c[:,2],
				triangles = simplices,
				color = col, shade = False, alpha = 0.5,
			)
	if True:
		ax.plot(c[:,0], c[:,1], c[:,2], 'o', color = col)
		ax.view_init(30, 45)

def load_into_cache(splits, index, runs):
	env = cartpole
	eta_scale = 1
	m = 500
	modes = ['linear', 'chebyshev']
	segment_size = 100
	T = 300
	num_scalarizations = 50
	scalarizations = list(range(num_scalarizations))
	seeds = list(range(runs))
	orders = [5, 7, 9]
	
	spaces = {
		'linear': scalarization.SpherePlusSpace(env.k),
		'chebyshev': scalarization.OutcomeShadowSpace(env.min_return, env.max_return),
	}
	
	todo = list(itertools.product(modes, seeds, scalarizations, orders))
	
	for i, (mode, seed, scal_index, order) in enumerate(tqdm(todo, "work")):
		if i % splits != index:
			continue
		
		ss = spaces[mode]
		rng = np.random.default_rng(seed)
		
		scals = cache.call('{:02}/{}-{}-{}.pickle'.format(
			seed, mode, 1000, 50,
		), scalarization.make_covering, 1000, 50, ss, rng)
		scal = scals[scal_index]
		
		do_scal(env, eta_scale, m, order, scal, T, segment_size, seed)

def solve_scalarization(env, eta_scale, m, order, scal, T, segment_size, seed):
	policies = {}
	mean_returns_all = []
	
	Tprev = segment_size * ((T - 1) // segment_size)
	for Ts in list(range(segment_size, Tprev + 1, segment_size)) + [T]:
		policy, mean_returns = policy_iteration_segment(env, eta_scale, m, order, scal, Ts, segment_size, seed)
		policies[Ts] = policy
		mean_returns_all.append(mean_returns)
	
	mean_returns_all = np.concatenate(mean_returns_all, axis = 0)
	return policies, mean_returns_all

@cache('{seed:02}/{env.name}-{segment_size}-{eta_scale}-{m}-{order}/{scal:.2f}/T={T}.pickle')
def policy_iteration_segment(env, eta_scale, m, order, scal, T, segment_size, seed):
	Tprev = segment_size * ((T - 1) // segment_size)
	phi = featurizer.FourierFeaturizer(env, order)
	impl = MyImpl(env, phi, eta_scale, m, scal)
	if Tprev == 0:
		policy = impl.initial_policy()
	else:
		policy, _ = policy_iteration_segment(env, eta_scale, m, order, scal, Tprev, segment_size, seed)
	_, values, policy = pi.policy_iteration(policy, Tprev, T, impl, misc.SeedTree(seed), progress = tqdm)
	return policy, values

class MyImpl:
	def __init__(self, env, featurizer, eta_scale, m, scal):
		self.env = env
		self.featurizer = featurizer
		self.m = m
		return_range = scal(env.max_return) - scal(env.min_return)
		self.eta = eta_scale * np.sqrt(8 * np.log(env.num_actions)) / return_range
		self.scal = scal
	
	def initial_policy(self):
		return pi.PolitexPolicy(self.featurizer)
	
	def evaluate(self, policy, seed_tree):
		if self.env.gamma < 1:
			H = int(1/(1-self.env.gamma) * np.log(1/0.01))
		else:
			H = self.env.max_steps
		data = collect_data(self.env, self.m, H, policy, seed_tree.rng())
		thetas, value = fit_parameters_mc(data, self.featurizer, self.env.gamma)
		theta = thetas @ self.scal.linearize_at(value)
		return theta, value
	
	def improve(self, policy, theta):
		return policy + self.eta * theta

def collect_data(env, m, H, policy, rng):
	trajectories = []
	for episode in range(m):
		traj = []
		s = env.sample_start(rng)
		for _ in range(H):
			tv = env.terminal_value(s)
			if tv is not None:
				break
			a = policy(rng, s)
			r, sp = env.sample_transition(rng, s, a)
			traj.append((s, a, r, sp))
			s = sp
		trajectories.append(traj)
	return trajectories

def fit_parameters_mc(data, featurizer, gamma):
	R = []
	returns = []
	states = []
	actions = []
	action_indices = [
		[] for _ in range(featurizer.env.num_actions)
	]
	j = 0
	for traj in data:
		states  += [s for s, _, _, _ in traj]
		actions += [a for _, a, _, _ in traj]
		for _, a, _, _ in traj:
			action_indices[a].append(j)
			j += 1
		l = len(traj)
		r = np.zeros((l, traj[0][2].shape[0]), dtype = np.float64)
		r[l - 1] = traj[l - 1][2]
		for i in range(l - 1, 0, -1):
			r[i - 1] = r[i] * gamma + traj[i - 1][2]
		returns.append(r[0])
		R.append(r)
	
	thetas = []
	Phi = featurizer.featurize_states(states).astype(np.float64)
	R = np.concatenate(R, axis = 0)
	reg = np.eye(featurizer.d) * 1e-4
	for a, indices in enumerate(action_indices):
		Phi_a = Phi[indices]
		R_a = R[indices]
		tmp1 = Phi_a.T @ Phi_a + reg
		tmp2 = Phi_a.T @ R_a
		thetas.append(np.linalg.solve(tmp1, tmp2))
	thetas = np.stack(thetas, axis = 0)
	
	return thetas, np.mean(returns, axis = 0)

if __name__ == '__main__':
	funcli.main({ collect, graphs })
