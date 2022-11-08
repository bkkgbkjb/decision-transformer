GoalPoints = {
    "antmaze-medium-diverse-v2": (20., 20.0),
    "antmaze-medium-play-v2": (20., 20.),
    "antmaze-large-diverse-v2": (32., 24.),
    "antmaze-large-play-v2": (32., 24.),
}

if __name__ == "__main__":
  import gym
  import d4rl
  env = gym.make("antmaze-large-diverse-v2")
  ob = env.reset()
  for _ in range(5):
    rlt = env.step(env.action_space.sample())
    print('')