import os
import numpy as np
from gymnasium import spaces
import mujoco
from pettingzoo import ParallelEnv

class FencingDuelEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        xml_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dual_humanoid_fencing.xml")
        )
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo model not found at: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.agents = ["fencer_a", "fencer_b"]
        self.possible_agents = self.agents[:]
        
        # Split actuators: 21 for A, 21 for B (total 42)
        if self.model.nu % 2 != 0:
            raise ValueError(f"Expected even number of actuators, got {self.model.nu}")
        num_actuators_per_agent = self.model.nu // 2
        
        # Observations: local qpos/qvel + global tip & opponent torso pos
        # obs_size: (nq/2) + (nv/2) + 3 + 3 = 61
        if (self.model.nq % 2) != 0 or (self.model.nv % 2) != 0:
            raise ValueError(
                f"Expected even nq/nv for two agents, got nq={self.model.nq}, nv={self.model.nv}"
            )
        obs_size = (self.model.nq // 2) + (self.model.nv // 2) + 4 
        print(f"DEBUG: Initialized FencingDuelEnv with obs_size={obs_size}")
        
        self.observation_spaces = {
            a: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_size,),
                dtype=np.float32,
            )
            for a in self.agents
        }
        self.action_spaces = {
            a: spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(num_actuators_per_agent,),
                dtype=np.float32,
            )
            for a in self.agents
        }

        self.render_mode = render_mode
        self.viewer = None

        # IDs for lookup
        self.body_ids = {
            "fencer_a": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"),
            "fencer_b": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_B"),
        }
        self.tip_ids = {
            "fencer_a": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "epee_tip"),
            "fencer_b": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "epee_tip_B"),
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _get_obs(self, agent):
        idx = 0 if agent == "fencer_a" else 1
        start_q, end_q = idx * (self.model.nq//2), (idx+1) * (self.model.nq//2)
        start_v, end_v = idx * (self.model.nv//2), (idx+1) * (self.model.nv//2)
        
        qpos = self.data.qpos[start_q:end_q]
        qvel = self.data.qvel[start_v:end_v]
        
        # 1. Translation invariance: Exclude root X and Y (qpos[0], qpos[1])
        # Keep root Z, orientation, and joints. Size: 28 - 2 = 26.
        obs_q = qpos[2:] 
        obs_v = qvel # Size: 27.
        
        # 2. Relative vectors (translation invariant)
        my_torso = self.data.xipos[self.body_ids[agent]]
        my_tip = self.data.site_xpos[self.tip_ids[agent]]
        opp_agent = "fencer_b" if agent == "fencer_a" else "fencer_a"
        opp_torso = self.data.xipos[self.body_ids[opp_agent]]
        
        rel_tip_opp = opp_torso - my_tip # Size: 3.
        rel_torso_opp = opp_torso - my_torso # Size: 3.
        
        # Total size: 26 + 27 + 3 + 3 = 59.
        return np.concatenate([
            obs_q, obs_v, 
            rel_tip_opp, rel_torso_opp
        ]).astype(np.float32)

    def step(self, actions):
        # actions is a dict: {"fencer_a": [...], "fencer_b": [...]}
        joint_actions = np.concatenate([actions["fencer_a"], actions["fencer_b"]])
        self.data.ctrl[:] = np.clip(joint_actions, -1, 1)
        
        mujoco.mj_step(self.model, self.data)

        # Calculate rewards/terminations
        rewards = {}
        terminations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        for agent in self.agents:
            opp = "fencer_b" if agent == "fencer_a" else "fencer_a"
            idx = 0 if agent == "fencer_a" else 1
            start_v = idx * (self.model.nv // 2)

            # Get body/site positions
            my_torso_pos = self.data.xipos[self.body_ids[agent]]
            my_tip_pos = self.data.site_xpos[self.tip_ids[agent]]
            opp_torso_pos = self.data.xipos[self.body_ids[opp]]
            
            # 1. Survival Bonus (Balanced to not drown out penalties)
            reward_alive = 1.0 
            
            # 2. Distance Reward (Shaping)
            dist = np.linalg.norm(my_tip_pos - opp_torso_pos)
            reward_dist = -0.1 * (dist ** 2)
            
            # 3. Velocity Reward (The "Engine" of movement)
            # Calculate velocity of torso along the vector to the opponent
            my_torso_vel = self.data.qvel[start_v:start_v+3]
            vec_to_opp = (opp_torso_pos - my_torso_pos)
            dist_torso = np.linalg.norm(vec_to_opp)
            vec_to_opp_unit = vec_to_opp / (dist_torso + 1e-6)
            vel_towards_opp = np.dot(my_torso_vel, vec_to_opp_unit)
            reward_vel = 0.5 * vel_towards_opp 
            
            # 4. Balance Reward (Stay upright)
            my_height = my_torso_pos[2]
            reward_upright = -3.0 * abs(my_height - 1.28) 
            
            # 5. Control/Energy Penalty (Encourage efficient movement)
            reward_ctrl = -0.001 * np.square(actions[agent]).sum()
            
            # 6. Hit Bonus (Simplified for density)
            reward_hit = 0.0
            if dist < 0.1:
                reward_hit = 20.0
            elif dist < 0.5:
                reward_hit = 1.0 / (dist + 0.1)
            
            # 7. Fall Penalty & Termination
            reward_fall = 0.0
            if my_height < 0.7: 
                reward_fall = -100.0
                terminations = {a: True for a in self.agents}
            
            # 8. Out-of-piste penalty
            reward_piste = 0.0
            if abs(my_torso_pos[0]) > 7.5 or abs(my_torso_pos[1]) > 1.5:
                reward_piste = -100.0
                terminations = {a: True for a in self.agents}

            rewards[agent] = (reward_alive + reward_dist + reward_vel + 
                              reward_upright + reward_ctrl + reward_hit + 
                              reward_fall + reward_piste)

        observations = {a: self._get_obs(a) for a in self.agents}
        truncations = {a: False for a in self.agents}
        
        if all(terminations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                import sys
                try:
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                except RuntimeError as exc:
                    if "mjpython" in str(exc).lower() or sys.platform == "darwin":
                        print(f"Warning: Rendering not available on macOS without mjpython: {exc}")
                        print("To enable rendering, run your script with: mjpython your_script.py")
                        self.viewer = None
                        return
                    raise
            if self.viewer is not None:
                self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None