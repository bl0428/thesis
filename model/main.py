from fencing_env import FencingEnv
import numpy as np

env = FencingEnv(render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)
    env.render()
    if done:
        obs, info = env.reset()

env.close()