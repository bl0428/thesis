import gymnasium as gym
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from gymnasium.envs.registration import register
from time import sleep

# ---- Register the environment ----
register(
    id="FencingBiped-v0",
    entry_point="fencing_env.envs.fencing_env:FencingEnv",
    max_episode_steps=1000,
)

num_cpu = 4
env_id = "FencingBiped-v0"

vec_env = make_vec_env(env_id, n_envs=num_cpu, env_kwargs={"render_mode": None})
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# Evaluation environment (also normalized to match training)
eval_env = make_vec_env(env_id, n_envs=1, env_kwargs={"render_mode": None})
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0, training=False)

log_dir = "/Users/brandon/Documents/thesis/model/logs/ppo_fencing"
os.makedirs(log_dir, exist_ok=True)

# INCREASED: Humanoids need much more training time
total_timesteps = 5_000_000  # Start with 1M, can increase to 5-10M for better performance

# PPO with humanoid-optimized hyperparameters
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log=log_dir + "/tb/",
    # Hyperparameters tuned for humanoid complexity
    learning_rate=3e-4,  # Standard learning rate
    n_steps=2048,  # Steps per update (good default)
    batch_size=64,  # Batch size (can increase to 128-256 for larger networks)
    n_epochs=10,  # PPO epochs per update (default is 10)
    gamma=0.99,  # Discount factor
    gae_lambda=0.95,  # GAE lambda
    clip_range=0.2,  # PPO clip range
    ent_coef=0.01,  # Entropy coefficient (encourages exploration)
    vf_coef=0.5,  # Value function coefficient
    max_grad_norm=0.5,  # Gradient clipping (important for stability)
    # Network architecture (humanoids benefit from larger networks)
    # Note: SB3 v1.8.0+ requires dict format, not list
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])  # Larger networks for complex observations
    )
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir + "/ppo_fencing/",
    log_path=log_dir + "/ppo_fencing/",
    eval_freq=5000,
    deterministic=True,
    render=False,
)

print(f"Starting training for {total_timesteps:,} timesteps...")
model.learn(total_timesteps=total_timesteps, callback=eval_callback)
model.save(os.path.join(log_dir, "fencer_final"))

vec_env.save(os.path.join(log_dir, "vec_normalize.pkl"))

vec_env.close()
eval_env.close()

print("\n--- Testing Best Model ---")
best_model_path = os.path.join(log_dir, "best_model.zip")
norm_stats_path = os.path.join(log_dir, "vec_normalize.pkl")

if os.path.exists(best_model_path):
    print(f"Loading best model from {best_model_path}")
    # Load normalization stats if available
    if os.path.exists(norm_stats_path):
        # Create normalized env for loading (model expects normalized obs)
        test_vec_env = make_vec_env(env_id, n_envs=1, env_kwargs={"render_mode": None})
        test_vec_env = VecNormalize.load(norm_stats_path, test_vec_env)
        test_vec_env.training = False
        best_model = PPO.load(best_model_path, env=test_vec_env)
        use_normalization = True
        # Note: test_vec_env will be closed later, but we keep it for now
        # since the model might reference it
    else:
        best_model = PPO.load(best_model_path)
        use_normalization = False
        test_vec_env = None
else:
    print("Best model not found, using final model")
    best_model = model
    use_normalization = True
    test_vec_env = None

# Create unnormalized environment for visualization
test_env = gym.make(env_id, render_mode="human")
obs, info = test_env.reset()

# Load normalization wrapper for manual normalization if needed
norm_wrapper = None
if use_normalization and os.path.exists(norm_stats_path):
    # Load VecNormalize wrapper (it's saved as a VecNormalize object, not a dict)
    test_vec_env_for_norm = make_vec_env(env_id, n_envs=1, env_kwargs={"render_mode": None})
    norm_wrapper = VecNormalize.load(norm_stats_path, test_vec_env_for_norm)
    norm_wrapper.training = False  # Don't update stats during testing
    print("Using normalization stats for testing")
    # Close the temporary env (we only needed it to load the wrapper)
    test_vec_env_for_norm.close()
    
for step in range(1000):
    # Normalize observation manually if model was trained with normalization
    if norm_wrapper is not None:
        # Use VecNormalize's normalize_obs method
        # Convert single obs to batch format (VecNormalize expects batched obs)
        obs_batch = np.array([obs])
        obs_normalized_batch = norm_wrapper.normalize_obs(obs_batch)
        obs_normalized = obs_normalized_batch[0]  # Extract single obs from batch
        action, _ = best_model.predict(obs_normalized, deterministic=True)
    else:
        action, _ = best_model.predict(obs, deterministic=True)
    
    obs, reward, terminated, truncated, info = test_env.step(action)
    sleep(0.01)
    test_env.render()
    
    if terminated or truncated:
        print(f"Episode finished at step {step} with reward: {reward:.2f}")
        obs, info = test_env.reset()

test_env.close()
# Close the test_vec_env if it was created
if 'test_vec_env' in locals() and test_vec_env is not None:
    test_vec_env.close()