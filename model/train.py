import os
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

# ---- Register the environment ----
# Make sure the entry_point points to your env class: "custom_env.biped_env:BipedEnv"
register(
    id="FencingBiped-v0",
    entry_point="fencing_env.fencing_env:FencingEnv",
    max_episode_steps=1000,
)

# ---- Helper to create a monitored single env (for both training and eval) ----
def make_env(render_mode=None):
    def _init():
        env = gym.make("FencingBiped-v0", render_mode=render_mode)
        # Wrap with Monitor to record episode reward/length (SB3 expects this)
        env = Monitor(env)
        return env
    return _init

# ---- Vectorized training environment ----
num_cpu = 4  # tune to number of cores; 1-8 typical
vec_env = DummyVecEnv([make_env(render_mode=None) for _ in range(num_cpu)])
# Optionally normalize observations and rewards (often helpful)
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
vec_env = VecMonitor(vec_env)  # logging

# ---- Create evaluation environment (non-vectorized) ----
# ---- Create evaluation environment ----
eval_env = DummyVecEnv([make_env(render_mode=None)])  # vectorized, same as training
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
eval_env = VecMonitor(eval_env)

# ---- Callbacks ----
log_dir = "logs/ppo_biped"
os.makedirs(log_dir, exist_ok=True)

# Save a model checkpoint every 50k steps
checkpoint_callback = CheckpointCallback(save_freq=50_000 // num_cpu, save_path=log_dir,
                                         name_prefix="ppo_biped")

# Early stop if we reach high reward (optional)
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=500.0, verbose=1)
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=stop_callback,
    best_model_save_path=log_dir + "/best_model",
    log_path=log_dir + "/eval_logs",
    eval_freq=25_000 // num_cpu,
    n_eval_episodes=5,
    deterministic=True,
)

# ---- Create the model ----
model = PPO(
    policy="MlpPolicy",
    env=vec_env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,      # number of steps to run for each environment per update (try 1024-8192)
    batch_size=64,
    n_epochs=10,       # number of epochs to update the policy
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log=log_dir + "/tb/",
)

# ---- Train ----
total_timesteps = 1_000_000  # change as needed
model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])

# Save final model and normalizer
model.save(os.path.join(log_dir, "ppo_biped_final"))
vec_env.save(os.path.join(log_dir, "vecnormalize.pkl"))

# ---- Evaluate final policy (non-vectorized, with rendering) ----
eval_env = gym.make("FencingBiped-v0", render_mode="human")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"Eval mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
eval_env.close()

