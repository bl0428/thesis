"""
Run and visualize a trained self-play PPO model (Stable-Baselines3).

Uses the same env setup as train_selfplay_sb3 (VecNormalize, clip_obs, etc.)
so visualization accurately reflects trained behavior.

Usage:
    mjpython model/run_selfplay_sb3.py   # Required on macOS for MuJoCo viewer
    python model/run_selfplay_sb3.py      # Linux (or if viewer works)
"""
import os
import re
import sys
import time
from pathlib import Path

script_dir = Path(__file__).resolve().parent
for p in (script_dir, script_dir.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from stable_baselines3 import PPO

from fencing_env.envs.selfplay_env import make_eval_env

OUTPUT_DIR = script_dir / "logs" / "ppo_selfplay_sb3"
VECNORM_PATH = OUTPUT_DIR / "vec_normalize.pkl"
N_EPISODES = 5
# Match training eval: episodes avg ~3000 steps; use 5000 to avoid truncation
MAX_STEPS = 5000
DO_LIVE_RENDER = True
MUJOCO_TIMESTEP = 0.005  # Match dual_humanoid_fencing.xml; for real-time sync

MODEL_PATHS = [
    OUTPUT_DIR / "best_model.zip",
    OUTPUT_DIR / "selfplay_final.zip",
]


def main():
    if not OUTPUT_DIR.exists():
        print(f"Output dir not found: {OUTPUT_DIR}")
        print("Run train_selfplay_sb3.py first.")
        return

    model_path = next((p for p in MODEL_PATHS if p.exists()), None)
    if not model_path:
        # Fallback: latest ppo_selfplay_* checkpoint
        def step_num(p):
            m = re.search(r"(\d+)_steps", p.stem)
            return int(m.group(1)) if m else 0
        checkpoints = sorted(OUTPUT_DIR.glob("ppo_selfplay_*_steps.zip"), key=step_num)
        model_path = checkpoints[-1] if checkpoints else None
    if not model_path:
        print(f"No model found in {OUTPUT_DIR}")
        return

    try:
        env = make_eval_env(VECNORM_PATH, render_mode="human" if DO_LIVE_RENDER else None)
    except FileNotFoundError as e:
        print(e)
        return

    model = PPO.load(str(model_path), env=env)

    # With num_envs=1, vec env has 2 rows (fencer_a, fencer_b). rewards shape = (2,).
    n_agents = 2
    agent_names = ["fencer_a", "fencer_b"]

    print(f"Running {model_path.name}... (total return = sum of both agents, matches training)")
    if DO_LIVE_RENDER:
        print("Rendering at real-time speed (sync to simulation).")

    for ep in range(N_EPISODES):
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        ep_returns = [0.0] * n_agents  # per-agent returns
        step_start = time.time()

        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            # rewards shape (2,) for 2 agents; index 0=fencer_a, 1=fencer_b
            for i in range(n_agents):
                ep_returns[i] += float(rewards[i])

            if DO_LIVE_RENDER:
                # Use base env for reliable MuJoCo viewer sync (vec env render may not propagate)
                if hasattr(env, "_base_env"):
                    env._base_env.render()
                else:
                    env.render()
                # Sync to real-time: sleep if we're ahead of sim
                sim_time = (step + 1) * MUJOCO_TIMESTEP
                elapsed = time.time() - step_start
                if sim_time > elapsed:
                    time.sleep(sim_time - elapsed)

            if dones.any():
                total_return = sum(ep_returns)
                per_agent = dict(zip(agent_names, [round(r, 2) for r in ep_returns]))
                print(f"Episode {ep + 1} | {step} steps | Return: {total_return:.2f} | {per_agent}")
                break
        else:
            # Reached MAX_STEPS without termination
            total_return = sum(ep_returns)
            per_agent = dict(zip(agent_names, [round(r, 2) for r in ep_returns]))
            print(f"Episode {ep + 1} | {MAX_STEPS} steps (max) | Return: {total_return:.2f} | {per_agent}")

    # Ensure MuJoCo viewer is closed
    if hasattr(env, "_base_env") and hasattr(env._base_env, "close"):
        env._base_env.close()
    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
