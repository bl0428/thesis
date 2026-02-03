import ray
import time
import sys
from pathlib import Path
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from fencing_env.envs.fencing_duel_env import FencingDuelEnv

# Add project root to sys.path so workers can find fencing_env
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 1. Re-register the environment (required for loading checkpoints)
def env_creator(config):
    return ParallelPettingZooEnv(FencingDuelEnv())

register_env("fencing_duel_v0", env_creator)

# The training script saves to root/model/logs/ppo_fencing_marl
# We use SCRIPT_DIR (model/) to find logs
RAY_RESULTS_DIR = SCRIPT_DIR / "logs" / "ppo_fencing_marl"

N_EPISODES = 5          # how many episodes to simulate
MAX_STEPS = 1000        # cap steps per episode
# turn this to True if you want live Mujoco viewer
DO_LIVE_RENDER = True


def find_latest_checkpoint(results_dir: Path) -> Path:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found at {results_dir}")

    # Find all checkpoint directories
    candidates = list(results_dir.glob("**/checkpoint_*"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under: {results_dir}")

    # Sort by checkpoint number (the integer part of 'checkpoint_XXXXXX')
    def get_checkpoint_num(p: Path):
        try:
            return int(p.name.split("_")[1])
        except (IndexError, ValueError):
            return -1

    candidates.sort(key=get_checkpoint_num)
    latest = candidates[-1]
    return latest


def main():
    ray.init(ignore_reinit_error=True)
    
    try:
        checkpoint_path = find_latest_checkpoint(RAY_RESULTS_DIR)
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        # Load the algorithm
        algo = Algorithm.from_checkpoint(checkpoint_path)
        
        env = FencingDuelEnv(render_mode="human" if DO_LIVE_RENDER else None)

        for ep in range(N_EPISODES):
            observations, infos = env.reset()
            ep_return = {agent: 0.0 for agent in env.possible_agents}

            for step in range(MAX_STEPS):
                actions = {}
                for agent, obs in observations.items():
                    # Use algo.compute_single_action to ensure observation filters are applied
                    action = algo.compute_single_action(
                        obs,
                        policy_id="shared_policy",
                        explore=False
                    )
                    actions[agent] = action

                observations, rewards, terminations, truncations, infos = env.step(actions)
                for agent, reward in rewards.items():
                    ep_return[agent] += float(reward)

                if DO_LIVE_RENDER:
                    env.render()
                    # Slow down slightly to match real-time
                    time.sleep(env.model.opt.timestep) 

                if any(terminations.values()) or any(truncations.values()):
                    print(
                        f"Episode {ep + 1} finished in {step} steps, "
                        f"reward A {ep_return.get('fencer_a', 0.0):.3f}, "
                        f"reward B {ep_return.get('fencer_b', 0.0):.3f}"
                    )
                    break

        env.close()
        algo.stop()
    finally:
        ray.shutdown()
    print("\nDone watching.")


if __name__ == "__main__":
    main()