import os
import gymnasium as gym
from stable_baselines3 import PPO

# Register the custom env
gym.register(
    id="FencingBiped-v0",
    entry_point="fencing_env.envs.fencing_env:FencingEnv",
    max_episode_steps=1000,
)

# Absolute model path without double .zip
MODEL_PATH = "/Users/brandon/Documents/thesis/model/logs/ppo_fencing/fencer_final"

eval_env = gym.make("Humanoid-v5", render_mode="human")
obs, info = eval_env.reset()

model = PPO.load(MODEL_PATH)

for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    eval_env.render()
    
    if terminated or truncated:
        obs, info = eval_env.reset()

eval_env.close()