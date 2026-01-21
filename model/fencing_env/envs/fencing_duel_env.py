import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

class FencingDuelEnv(gym.Env):
    """
    Two fencers on a 14m x 2m piste (see dual_humanoid_fencing.xml).
    Single control vector over all actuators (A + B). Reward encourages each
    fencer to hit the opponent torso with its epee tip and stay upright.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        xml_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dual_humanoid_fencing.xml")
        )
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo model not found at: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None

        # Body IDs
        self.torso_body_id_A = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.torso_body_id_B = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_B")

        # Site IDs (tips)
        self.epee_tip_A = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "epee_tip")
        try:
            self.epee_tip_B = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "epee_tip_B")
        except ValueError:
            # Fallback: if only one name exists, reuse (but rewards may not be meaningful)
            self.epee_tip_B = self.epee_tip_A

        # Dimensions
        self.num_actuators = self.model.nu
        obs_size = self.model.nq + self.model.nv + 6  # qpos+qvel + two tip positions (3+3)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actuators,), dtype=np.float64
        )

        self.dt = self.model.opt.timestep

    def _get_obs(self):
        state_data = np.concatenate([self.data.qpos, self.data.qvel])
        tip_a = self.data.site_xpos[self.epee_tip_A, :]
        tip_b = self.data.site_xpos[self.epee_tip_B, :]
        return np.concatenate([state_data, tip_a, tip_b])

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, -1, 1)
        mujoco.mj_step(self.model, self.data)

        torso_height_A = self.data.xipos[self.torso_body_id_A][2]
        torso_height_B = self.data.xipos[self.torso_body_id_B][2]
        tip_a = self.data.site_xpos[self.epee_tip_A, :]
        tip_b = self.data.site_xpos[self.epee_tip_B, :]
        torso_A = self.data.xipos[self.torso_body_id_A, :]
        torso_B = self.data.xipos[self.torso_body_id_B, :]

        # Distances tip->opponent torso
        dist_a_to_B = float(np.linalg.norm(tip_a - torso_B))
        dist_b_to_A = float(np.linalg.norm(tip_b - torso_A))

        reward = 0.0
        # Shaping toward opponent torso
        reward -= dist_a_to_B
        reward -= dist_b_to_A

        hit_a = dist_a_to_B < 0.05
        hit_b = dist_b_to_A < 0.05
        if hit_a:
            reward += 5.0
        if hit_b:
            reward += 5.0

        # Control cost
        reward -= 0.001 * np.sum(np.square(self.data.ctrl[: self.model.nu]))

        # Termination if either fencer falls
        done = bool((torso_height_A < 0.4) or (torso_height_B < 0.4))
        if done:
            reward -= 5.0

        return self._get_obs(), reward, done, False, {
            "dist_a_to_B": dist_a_to_B,
            "dist_b_to_A": dist_b_to_A,
            "hit_a": hit_a,
            "hit_b": hit_b,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                import sys
                try:
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                except RuntimeError as e:
                    if "mjpython" in str(e).lower() or sys.platform == "darwin":
                        print(f"Warning: Rendering not available on macOS without mjpython: {e}")
                        print("To enable rendering, run your script with: mjpython your_script.py")
                        self.viewer = None
                        return
                    else:
                        raise
            if self.viewer is not None:
                self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

