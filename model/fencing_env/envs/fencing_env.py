import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os
import mujoco_viewer

class FencingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        # Resolve model path relative to package
        xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fencer_model.xml"))
        
        # Check if the XML file exists before proceeding
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo model not found at: {xml_path}. Please ensure 'fencer_model.xml' is present.")
            
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None

        # 1. Root and Torso IDs
        self.root_jnt_id = self.model.joint(name='root').id 
        self.torso_x_vel_id = self.model.jnt_dofadr[self.root_jnt_id] # X-translation velocity
        self.torso_z_pos_id = self.model.jnt_qposadr[self.root_jnt_id] + 2 # Z-translation position

        # 2. Sensor and Site IDs
        self.epee_tip_site_id = self.model.site(name='epee_tip').id
        # Optional target site/sensor (for hit reward); be robust to missing entries
        self.target_site_id = None
        try:
            self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'target_site')
        except ValueError:
            self.target_site_id = None
        self.target_touch_sensor_id = None
        try:
            self.target_touch_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'target_hit')
        except ValueError:
            self.target_touch_sensor_id = None
        # NOTE: sensor data is returned as a 1D array, even for a single value
        self.epee_touch_sensor_id = self.model.sensor(name='epee_tip_touch').id
        
        # 3. Actuator/Control Information
        self.num_actuators = self.model.nu 
        
        # 4. Define Spaces (Fixed Observation Space Calculation)
        
        # Observation Space size = (qpos) + (qvel) + (epee_tip_pos: 3D) + (epee_tip_touch: 1D)
        obs_size = self.model.nq + self.model.nv + 3 + 1 # Corrected: 29 + 28 + 3 + 1 = 61
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)

        # Action Space: Based on the number of motors/actuators (21)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_actuators,), dtype=np.float64)
        
        self.dt = self.model.opt.timestep


    def _get_obs(self):
        # 1. QPOS and QVEL (size 57)
        state_data = np.concatenate([self.data.qpos, self.data.qvel])
        
        # 2. Epee Tip Position (3D array)
        epee_tip_pos = self.data.site_xpos[self.epee_tip_site_id, :]

        # 3. Epee Tip Touch Force (1D array from sensordata)
        # Note: sensordata is a flat array, the sensor value is indexed by the sensor ID
        # Since 'touch' reports a force, the value will be a scalar. We access it as an array for concatenation.
        epee_tip_touch = np.array([self.data.sensordata[self.epee_touch_sensor_id]])
        
        # 4. Concatenate all data (size 29 + 28 + 3 + 1 = 61)
        return np.concatenate([state_data, epee_tip_pos, epee_tip_touch])

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, -1, 1)
        mujoco.mj_step(self.model, self.data)

        reward = 0.0

        torso_z_vel = self.data.qvel[self.torso_x_vel_id + 2]  # Z-velocity (vertical)
        torso_height = self.data.qpos[self.torso_z_pos_id]
        
        # Get torso orientation (quaternion) to check if upright
        root_qpos_start = self.model.jnt_qposadr[self.root_jnt_id]
        torso_quat = self.data.qpos[root_qpos_start + 3:root_qpos_start + 7]  # Quaternion [w, x, y, z]
        
        # Termination condition: fell down
        done = bool(torso_height < 0.38)
        
        # A. Height Reward: Encourage reaching and maintaining standing height
        # Target height: ~1.1m (between squat 0.6m and stand 1.22m)
        target_height = 1.1
        min_height = 0.6  # Minimum acceptable height (squat level)
        
        if torso_height < 0.38:
            # Stronger penalty for falling
            reward -= 45.0
            done = True
        elif torso_height < min_height:
            # Penalty for being too low (but not fallen)
            reward -= 15 * (min_height - torso_height) / min_height
        else:
            # Reward for height, with peak at target_height
            # Use a smooth reward that peaks at target_height
            height_diff = abs(torso_height - target_height)
            height_reward = 2.0 * np.exp(-height_diff / 0.2)  # Gaussian-like reward
            reward += height_reward
        
        # B. Orientation Reward: Encourage upright posture
        # Quaternion [w, x, y, z] - upright means w should be close to 1
        # For upright: quat should be close to [1, 0, 0, 0]
        upright_score = torso_quat[0]  # w component (1 = upright, 0 = horizontal)
        orientation_reward = 1.5 * (upright_score - 0.7)  # Reward if w > 0.7
        if upright_score > 0.7:
            reward += orientation_reward
        else:
            reward -= 0.5 * (0.7 - upright_score)  # Small penalty for tilting

        # Additional tilt penalty to discourage pitching over early
        if upright_score < 0.9:
            reward -= 1.0 * (0.9 - upright_score)
        
        # C. Stability Reward: Encourage staying still while standing
        # Penalize excessive vertical velocity (wobbling)
        stability_reward = -0.5 * abs(torso_z_vel)
        reward += stability_reward
        
        # Optional: Small penalty for horizontal movement (if you want standing still)
        # Uncomment if you want agent to stand in place:
        # reward -= 0.1 * abs(torso_x_vel)
        
        # D. Control Cost: Penalize high energy expenditure
        control_cost = 0.001 * np.sum(np.square(self.data.ctrl[:self.model.nu]))
        reward -= control_cost

        # E. Touch/Epee Interaction Reward / Target hit
        hit_target = False
        target_dist = None
        if self.target_site_id is not None:
            epee_tip_pos = self.data.site_xpos[self.epee_tip_site_id, :]
            target_pos = self.data.site_xpos[self.target_site_id, :]
            target_dist = float(np.linalg.norm(epee_tip_pos - target_pos))
            # Shaped reward: stronger pull near target, mild penalty far away
            reward -= (target_dist ** 2)
            # reward += 1.0 * np.exp(-6.0 * target_dist)
            
        if self.target_touch_sensor_id is not None:
            touch_val = float(self.data.sensordata[self.target_touch_sensor_id])
            if touch_val > 0:
                reward += 15.0
                hit_target = True
                done = True

        return self._get_obs(), reward, done, False, {"hit_target": hit_target, "target_distance": target_dist}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # Reset to a default standing pose (use keyframe if available)
        stand_key_id = self.model.key(name='stand_on_left_leg').id
        if stand_key_id != -1:
            self.data.qpos[:] = self.model.key_qpos[stand_key_id]
            self.data.qvel[:] = self.model.key_qvel[stand_key_id]
        else:
            # Fallback to squat if stand keyframe is unavailable
            squat_key_id = self.model.key(name='squat').id
            if squat_key_id != -1:
                self.data.qpos[:] = self.model.key_qpos[squat_key_id]
                self.data.qvel[:] = self.model.key_qvel[squat_key_id]
        
        # Re-initialize the data after setting qpos/qvel
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                import sys
                # Passive viewer is non-blocking and great for Gym/RL
                # On macOS, launch_passive requires mjpython, so we catch the error
                try:
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                except RuntimeError as e:
                    if "mjpython" in str(e).lower() or sys.platform == "darwin":
                        # On macOS, passive viewer requires mjpython
                        # Skip rendering or use alternative method
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