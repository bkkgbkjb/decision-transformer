import gym
import d4rl
import d3rlpy
from d3rlpy.metrics.scorer import evaluate_on_environment
import numpy as np
from reporter import get_reporter


dataset, _  = d3rlpy.datasets.get_dataset("hopper-medium-replay-v2")
env = gym.make("Hopper-v3")

env.seed(0)
d3rlpy.seed(0)

cql = d3rlpy.algos.CQL(use_gpu=True)

cql.build_with_dataset(dataset)

reporter = get_reporter("original_cql")


for i in range(100):
	total_rwds = []
	for _ in range(10):
		obs = env.reset()
		stop = False
		total_rwd = 0
		while not stop:
			# obs = env.reset()
			act = cql.predict([obs])[0]
			obs, rwd, stop, _ = env.step(act)
			total_rwd += rwd
		# print(f"total_rwd is: {total_rwd}")
		total_rwds.append(total_rwd)

	print(f"finish evaluation {i}: total_rwd: {np.mean(total_rwds)}, {np.std(total_rwds)}")

	info = cql.fit(dataset, n_steps=int(1e3 / 2), n_steps_per_epoch=10, save_metrics=False,verbose=False,show_progress=False)
	logs = dict()
	logs['training/alpha'] = np.mean([i[1]['alpha'] for i in info])
	logs['training/alpha_loss_mean'] = np.mean([i[1]['alpha_loss'] for i in info])
	logs['training/actor_loss_mean'] = np.mean([i[1]['actor_loss'] for i in info])
	logs['training/critic_loss_mean'] = np.mean([i[1]['critic_loss'] for i in info])
	reporter({'return_mean': np.mean(total_rwds), 'return_std': np.std(total_rwds), **logs})

