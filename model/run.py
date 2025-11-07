from stable_baselines3 import PPO
import gymnasium as gym

model = PPO.load("logs/ppo_biped/best_model.zip")  # or ppo_biped_final.zip
# If you used VecNormalize during training, load and wrap env exactly the same and load the saved VecNormalize:
# vec_norm = VecNormalize.load("logs/ppo_biped/vecnormalize.pkl", env=None)
# env = make_vec_env(...); env = VecNormalize(env, training=False); env.obs_rms = vec_norm.obs_rms

env = gym.make("CustomBiped-v0", render_mode="human")
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()
env.close()
