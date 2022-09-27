import gym
import d4rl
import d3rlpy
from d3rlpy.metrics.scorer import evaluate_on_environment
import numpy as np
from reporter import get_reporter
import pickle
from d3rlpy.dataset import MDPDataset


env_name = 'hopper'
dataset = 'medium-replay'
dir_path = '.'
dataset_path = f'{dir_path}/data/{env_name}-{dataset}-v2.pkl'
with open(dataset_path, 'rb') as f:
		trajectories = pickle.load(f)

states, traj_lens, returns = [], [], []
for path in trajectories:
		states.append(path['observations'])
		traj_lens.append(len(path['observations']))
		returns.append(path['rewards'].sum())
traj_lens, returns = np.array(traj_lens), np.array(returns)

# used for input normalization
states = np.concatenate(states, axis=0)
state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

# dataset, _  = d3rlpy.datasets.get_dataset("hopper-medium-replay-v2")
d4rlenv = gym.make("hopper-medium-replay-v2")
d4rldataset = d4rlenv.get_dataset()
env = gym.make("Hopper-v3")

env.seed(0)
d3rlpy.seed(0)

cql = d3rlpy.algos.CQL(use_gpu=True)

# cql.build_with_dataset(dataset)

reporter = get_reporter("original_cql")

terminals = d4rldataset["terminals"]
timeouts = d4rldataset["timeouts"]
episode_terminals = np.logical_or(terminals, timeouts)
mdp_dataset = MDPDataset(
		# observations=np.array(observations, dtype=np.float32),
		# actions=np.array(actions, dtype=np.float32),
		observations=(d4rldataset['observations'] - state_mean)/state_std,
		actions=np.array(d4rldataset['actions'], dtype=np.float32),
		rewards=np.array(d4rldataset['rewards'], dtype=np.float32),
		terminals=np.array(terminals, dtype=np.float32),
		episode_terminals=np.array(episode_terminals, dtype=np.float32),
)

info = cql.fit(mdp_dataset, n_steps=int(1e3 / 2), n_steps_per_epoch=10, save_metrics=False,verbose=False,show_progress=False)
for i in range(100):
	total_rwds = []
	for _ in range(10):
		obs = env.reset()
		obs = (obs-state_mean) / state_std
		stop = False
		total_rwd = 0
		while not stop:
			# obs = env.reset()
			act = cql.predict([obs])[0]
			obs, rwd, stop, _ = env.step(act)
			obs = (obs-state_mean) / state_std
			total_rwd += rwd
		# print(f"total_rwd is: {total_rwd}")
		total_rwds.append(total_rwd)

	print(f"finish evaluation {i}: total_rwd: {np.mean(total_rwds)}, {np.std(total_rwds)}")

	info = cql.fit(mdp_dataset, n_steps=int(1e3 / 2), n_steps_per_epoch=10, save_metrics=False,verbose=False,show_progress=False)
	logs = dict()
	logs['training/alpha'] = np.mean([i[1]['alpha'] for i in info])
	logs['training/alpha_loss_mean'] = np.mean([i[1]['alpha_loss'] for i in info])
	logs['training/actor_loss_mean'] = np.mean([i[1]['actor_loss'] for i in info])
	logs['training/critic_loss_mean'] = np.mean([i[1]['critic_loss'] for i in info])
	reporter({'return_mean': np.mean(total_rwds), 'return_std': np.std(total_rwds), **logs})

