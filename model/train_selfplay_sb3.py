"""
Self-play PPO training using Stable-Baselines3 (no Ray).

Both agents share a single policy and learn by competing against themselves.
Uses SuperSuit to convert PettingZoo -> SB3 VecEnv.

Usage:
    python model/train_selfplay_sb3.py
"""
import os
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
for p in (script_dir, script_dir.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import LinearSchedule

from fencing_env.envs.selfplay_env import make_selfplay_vec_env

OUTPUT_DIR = script_dir / "logs" / "ppo_selfplay_sb3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Config
SEED = 42
TOTAL_TIMESTEPS = 500_000_000
NUM_ENVS = 8  # 8 copies = 16 agents (more experience per update)
EVAL_FREQ = 50000
CHECKPOINT_FREQ = 2_500_000


def main():
    np.random.seed(SEED)

    env = make_selfplay_vec_env(num_envs=NUM_ENVS, render_mode=None, normalize=True)
    # 8 envs × 2 agents = 16 parallel; n_steps×16 = batch
    n_rollout = 2048 * env.num_envs
    batch_size = min(2048, n_rollout // 4)  # 4 minibatches per epoch

    eval_callback = EvalCallback(
        env,
        best_model_save_path=str(OUTPUT_DIR),
        log_path=str(OUTPUT_DIR),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(OUTPUT_DIR),
        name_prefix="ppo_selfplay",
    )

    # Linear decay: starts at 3e-4, ends at 1e-5 over full training. Reduces overshooting late in training.
    lr_schedule = LinearSchedule(3e-4, 1e-5, end_fraction=1.0)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=SEED,
        tensorboard_log=str(OUTPUT_DIR / "tb"),
        learning_rate=lr_schedule,
        n_steps=2048,
        batch_size=batch_size,
        n_epochs=5,  # Reduced from 10 to limit overfitting to current batch
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )

    print(f"Training: {NUM_ENVS} envs ({NUM_ENVS * 2} agents), {TOTAL_TIMESTEPS} steps")
    print(f"Logs -> {OUTPUT_DIR}")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=CallbackList([eval_callback, checkpoint_callback]),
    )

    model.save(str(OUTPUT_DIR / "selfplay_final"))
    env.save(str(OUTPUT_DIR / "vec_normalize.pkl"))
    print(f"Done. Model + vec_normalize.pkl saved to {OUTPUT_DIR}")
    print(f"For best performance, load best_model.zip (saved by EvalCallback)")


if __name__ == "__main__":
    main()
