import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

register(
    id="FencingBiped-v0",
    entry_point="fencing_env.envs.fencing_env:FencingEnv",
    max_episode_steps=1000,
)

# Paths adjusted to current project layout
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "logs" / "ppo_fencing" / "fencer_final.zip"

N_EPISODES = 5          # how many episodes to simulate
MAX_STEPS = 2000        # cap steps per episode
# turn this to True if you want live Mujoco viewer
DO_LIVE_RENDER = True


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    env = None
    env = gym.make('FencingBiped-v0', render_mode="human")
    obs, info = env.reset()
    model = PPO.load(MODEL_PATH, env=env)

    for ep in range(N_EPISODES):
        obs, info = env.reset()
        ep_return = 0.0
        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)

            if DO_LIVE_RENDER:
                env.render()
                time.sleep(0.01)  # slow down visualization

            if terminated or truncated:
                print(f"Episode {ep+1} finished in {step} steps, reward {ep_return:.3f}")
                break

    env.close()
    print("\nDone watching.")
 
if __name__ == "__main__":
    main()