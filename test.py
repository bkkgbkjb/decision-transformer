import gym
import d4rl
import d3rlpy
from d3rlpy.metrics.scorer import evaluate_on_environment
import numpy as np



dataset, _  = d3rlpy.datasets.get_dataset("hopper-medium-v2")
env = gym.make("hopper-medium-v2")

env.seed(0)
d3rlpy.seed(0)

cql = d3rlpy.algos.CQL(use_gpu=True)

cql.build_with_dataset(dataset)


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

	print(f"finish evaluation {i}: total_rwd: {np.mean([total_rwds])}, {np.std(total_rwds)}")

	ret = cql.fit(dataset, n_steps=int(1e2), n_steps_per_epoch=1, save_metrics=False,verbose=False,show_progress=False)

	print('sd')

