import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco_viewer


class FencingTargetEnv(gym.Env):
    """
    Single fencer + stationary target. Reward encourages placing the epee tip
    near the target position. Uses the same humanoid model as other fencer envs.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, target_pos=None):
        # Resolve MuJoCo XML path relative to package
        xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fencer_model.xml"))
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo model not found at: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None

        # Root joint indices
        self.root_jnt_id = self.model.joint(name='root').id
        self.torso_z_pos_id = self.model.jnt_qposadr[self.root_jnt_id] + 2

        # Sites / sensors
        self.epee_tip_site_id = self.model.site(name='epee_tip').id
        self.target_site_id = self.model.site(name='target_site').id
        self.target_touch_sensor_id = self.model.sensor(name='target_hit').id

        # Spaces
        obs_size = self.model.nq + self.model.nv + 3 + 2  # qpos + qvel + epee_tip (3) + [epee_touch_placeholder, target_touch]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float64)

        self.dt = self.model.opt.timestep
        self.target_pos = np.array(target_pos if target_pos is not None else [1.0, 0.0, 1.0], dtype=np.float64)

    def _get_obs(self):
        state_data = np.concatenate([self.data.qpos, self.data.qvel])
        epee_tip_pos = self.data.site_xpos[self.epee_tip_site_id, :]
        epee_tip_touch = np.array([0.0])  # placeholder to align with other envs
        target_touch = np.array([self.data.sensordata[self.target_touch_sensor_id]])
        return np.concatenate([state_data, epee_tip_pos, epee_tip_touch, target_touch])

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, -1, 1)
        mujoco.mj_step(self.model, self.data)

        torso_height = self.data.qpos[self.torso_z_pos_id]
        epee_tip_pos = self.data.site_xpos[self.epee_tip_site_id, :]

        # Distance-based reward to target
        dist = float(np.linalg.norm(epee_tip_pos - self.target_pos))
        reward = -dist

        # Contact-based bonus using touch sensor
        touch_val = float(self.data.sensordata[self.target_touch_sensor_id])
        hit = (touch_val > 0) or (dist < 0.05)
        if hit:
            reward += 5.0

        # Control penalty
        reward -= 0.001 * np.sum(np.square(self.data.ctrl[:self.model.nu]))

        # Fall termination
        done = bool(torso_height < 0.38)
        if done:
            reward -= 5.0

        return self._get_obs(), reward, done, False, {"distance": dist, "hit": hit}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Stand pose if available
        stand_key_id = self.model.key(name='stand_on_left_leg').id
        if stand_key_id != -1:
            self.data.qpos[:] = self.model.key_qpos[stand_key_id]
            self.data.qvel[:] = self.model.key_qvel[stand_key_id]
        else:
            squat_key_id = self.model.key(name='squat').id
            if squat_key_id != -1:
                self.data.qpos[:] = self.model.key_qpos[squat_key_id]
                self.data.qvel[:] = self.model.key_qvel[squat_key_id]

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

