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
from stable_baselines3.common.vec_env import VecMonitor

from fencing_env.envs.selfplay_env import make_selfplay_vec_env

OUTPUT_DIR = script_dir / "logs" / "ppo_selfplay_sb3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Config
SEED = 42
TOTAL_TIMESTEPS = 25_000_000
NUM_ENVS = 4
EVAL_FREQ = 50000
CHECKPOINT_FREQ = 2_500_000


def main():
    np.random.seed(SEED)

    env = make_selfplay_vec_env(
        num_envs=NUM_ENVS,
        render_mode=None,
        normalize=True,
        zero_sum=True,
    )
    env = VecMonitor(env)
    env.render_mode = None
    eval_env = make_selfplay_vec_env(
        num_envs=2,
        render_mode=None,
        normalize=True,
        zero_sum=True,
    )
    eval_env = VecMonitor(eval_env)
    eval_env.render_mode = None
    n_rollout = 2048 * env.num_envs
    batch_size = min(2048, n_rollout // 4)  # 4 minibatches per epoch

    eval_callback = EvalCallback(
        eval_env,
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


    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=SEED,
        tensorboard_log=str(OUTPUT_DIR / "tb"),
        learning_rate=1e-4,
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

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=CallbackList([eval_callback, checkpoint_callback]),
    )

    # Save the best model as final (EvalCallback saves best_model.zip during training)
    best_path = OUTPUT_DIR / "best_model.zip"
    if best_path.exists():
        best_model = PPO.load(str(best_path))
        best_model.save(str(OUTPUT_DIR / "selfplay_final"))
        print(f"Saved best model as selfplay_final.zip")
    else:
        model.save(str(OUTPUT_DIR / "selfplay_final"))
        print(f"No eval yet; saved last model as selfplay_final.zip")

    env.venv.save(str(OUTPUT_DIR / "vec_normalize.pkl"))
    eval_env.close()
    print(f"Done. Model + vec_normalize.pkl saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
