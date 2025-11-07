import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os

class FencingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(__file__), "fencer_model.xml"))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None

        # Observation space: joint positions + velocities
        # The root freejoint is the FIRST joint in the model (ID 0)
        self.root_jnt_id = self.model.joint(name='root').id 

        # Torso X-velocity index in qvel (dofadr[0] is the index of the first DOF, which is X-translation)
        # Note: qvel[0] is typically root_x_vel
        self.torso_x_vel_id = self.model.jnt_dofadr[self.root_jnt_id] 
        
        # Torso Height (Z-position) index in qpos (qposadr[2] is the index of the Z-translation)
        # Note: qpos[2] is typically root_z_pos
        self.torso_z_pos_id = self.model.jnt_qposadr[self.root_jnt_id] + 2 

        # 3. Sensor IDs (for Stage 2/3 Epee Control and Touch)
        
        # ID for the 'epee_tip_touch' sensor you added
        # This sensor measures contact force/activation
        self.epee_touch_sensor_id = self.model.sensor(name='epee_tip_touch').id
        
        # ID for the 'epee_tip' site (to get its 3D position)
        # Site names are unique ways to track positions/orientations
        self.epee_tip_site_id = self.model.site(name='epee_tip').id
        
        # 4. Actuator/Control Information
        self.num_actuators = self.model.nu # Number of actuators (21 total for this Humanoid)
        
        # 5. Define Spaces (Continuous control, large state space)
        
        # Observation Space: Needs to be defined based on model.nq (qpos) and model.nv (qvel)
        # Add a few extra dimensions if you include things like epee tip position (3D)
        obs_size = self.model.nq + self.model.nv + 3 # qpos + qvel + epee tip position (mocap/site data)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)

        # Action Space: Based on the number of motors/actuators
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_actuators,), dtype=np.float64)
        
        self.dt = self.model.opt.timestep

    def _get_obs(self):
        # 1. Get QPOS and QVEL (size 55)
        state_data = np.concatenate([self.data.qpos, self.data.qvel])
        
        # 2. Get Epee Tip Position (3D array from site_xpos)
        # Note: data.site_xpos is a (num_sites, 3) array. Indexing with the site ID is correct.
        epee_tip_pos = self.data.site_xpos[self.epee_tip_site_id, :]
        
        # 3. Concatenate all data (size 55 + 3 = 58)
        return np.concatenate([state_data, epee_tip_pos])

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, -1, 1)
        mujoco.mj_step(self.model, self.data)

        # Reward: height of torso + forward velocity
        done = False
        reward = 0.0

        # Retrieve relevant MuJoCo data (needs body IDs for fencer A's torso and feet)
        # Note: You'll need to define these IDs in __init__
        torso_x_vel = self.data.qvel[self.torso_x_vel_id]  
        torso_height = self.data.qpos[self.torso_z_pos_id]
        
        # A. Goal Reward: Encourage forward movement (Torso X-velocity)
        reward += 1.0 * torso_x_vel 

        # B. Maintenance Reward: Encourage staying upright
        # Punish falling (torso too low)
        min_height = 0.8  # Example threshold
        if torso_height < min_height:
            reward -= 5.0
            done = True # End episode if the agent falls
        else:
            reward += 0.1 # Small bonus for staying up

        # C. Control Cost: Penalize high energy expenditure
        control_cost = 0.001 * np.sum(np.square(self.data.ctrl[:self.model.nu])) # Only Fencer A's actuators
        reward -= control_cost

        done = bool(self.data.qpos[2] < 0.4)  # fell down
        return self._get_obs(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
