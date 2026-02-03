import gymnasium as gym
from stable_baselines3 import PPO

env_id = "InvertedPendulum-v5"
eval_env = gym.make(env_id, render_mode="human")
obs, info = eval_env.reset()

model = PPO.load("logs/ppo_pendulum/pendulum_final")

for _ in range(1000):
    action, states = model.predict(obs)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    eval_env.render()
    
    if terminated or truncated:
        obs, info = eval_env.reset()

eval_env.close()