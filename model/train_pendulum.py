import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

num_cpu = 4
env_id = "InvertedPendulum-v5"

vec_env = make_vec_env(env_id, n_envs=num_cpu)
eval_env = make_vec_env(env_id, n_envs=1)

log_dir = "/Users/brandon/Documents/thesis/model/logs/ppo_pendulum"
os.makedirs(log_dir, exist_ok=True)

total_timesteps = 100000

model = PPO("MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=log_dir + "/tb/")

eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir + "/ppo_pendulum/",
                             log_path=log_dir + "/ppo_pendulum/", eval_freq=1000,
                             deterministic=True, render=False)

model.learn(total_timesteps=total_timesteps, callback=eval_callback)
model.save(os.path.join(log_dir, "pendulum_final"))

vec_env.close()

eval_env = gym.make(env_id, render_mode="human")
obs, info = eval_env.reset()

for _ in range(1000):
    action, states = model.predict(obs)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    eval_env.render()
    
    if terminated or truncated:
        obs, info = eval_env.reset()

eval_env.close()