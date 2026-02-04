import ray
import os
import sys
from pathlib import Path

# Add project root to sys.path so workers can find fencing_env
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))
if str(script_dir.parent) not in sys.path:
    sys.path.append(str(script_dir.parent))

from ray import tune
from ray.train import CheckpointConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from fencing_env.envs.fencing_duel_env import FencingDuelEnv

# 1. Register the environment so RLLib can find it
def env_creator(config):
    return ParallelPettingZooEnv(FencingDuelEnv())

register_env("fencing_duel_v0", env_creator)

if __name__ == "__main__":
    # Ensure registration is also inside __main__ for the driver
    register_env("fencing_duel_v0", env_creator)
    ray.init(ignore_reinit_error=True)

    # 2. Define the Multi-Agent Policy Mapping
    # Both fencers will share the same neural network weights (Self-Play)
    test_env = FencingDuelEnv()
    obs_space = test_env.observation_spaces["fencer_a"]
    act_space = test_env.action_spaces["fencer_a"]

    config = (
        PPOConfig()
        .environment("fencing_duel_v0")
        .framework("torch")
        # Use stable Legacy API stack for easier checkpoint loading/visualization
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .resources(num_gpus=1 if ray.cluster_resources().get("GPU") else 0)
        .env_runners(num_env_runners=4, observation_filter="MeanStdFilter")
        .multi_agent(
            policies={
                "shared_policy": (None, obs_space, act_space, {})
            },
            # Handle any combination of arguments Ray might pass
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .training(
            gamma=0.99,
            lr=1e-5, # Significantly lower LR for long-term stability
            train_batch_size=32000, # Larger batch size for smoother gradients
            minibatch_size=1024,
            num_epochs=20, # More epochs to squeeze more out of the larger batch
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01, # Lower entropy to allow convergence after discovery
            kl_coeff=0.2, # Explicit KL control to prevent catastrophic forgetting
            kl_target=0.01,
        )
    )

    # 3. Run Training
    stop = {
        "training_iteration": 10000
    }

    checkpoint_config = CheckpointConfig(
        num_to_keep=3,
        checkpoint_score_attribute="episode_reward_mean",
        checkpoint_score_order="max",
    )
    
    results = tune.run(
        "PPO",
        config=config.to_dict(),
        stop=stop,
        checkpoint_config=checkpoint_config,
        checkpoint_freq=10,
        checkpoint_at_end=True,
        storage_path=str(script_dir / "logs" / "ppo_fencing_marl")
    )

    ray.shutdown()