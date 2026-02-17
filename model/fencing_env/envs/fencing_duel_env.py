import os
import numpy as np
from gymnasium import spaces
import mujoco
from pettingzoo import ParallelEnv

class FencingDuelEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, zero_sum=True):
        """
        Args:
            render_mode: "human" or None.
            zero_sum: If True, hit/got-hit rewards are symmetric. Agent who
                lands hit gets +HIT_REWARD, opponent gets -HIT_REWARD.
        """
        xml_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dual_humanoid_fencing.xml")
        )
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo model not found at: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.zero_sum = zero_sum
        self.agents = ["fencer_a", "fencer_b"]
        self.possible_agents = self.agents[:]
        
        # Split actuators: 21 for A, 21 for B (total 42)
        if self.model.nu % 2 != 0:
            raise ValueError(f"Expected even number of actuators, got {self.model.nu}")
        num_actuators_per_agent = self.model.nu // 2
        
        # obs_size: (nq/2-2) + (nv/2) + 3 + 3 = 26 + 27 + 3 + 3 = 59
        obs_size = (self.model.nq // 2) - 2 + (self.model.nv // 2) + 6
        
        # Use float32 throughout to satisfy Gymnasium passive_env_checker
        low = np.full((obs_size,), np.float32(-np.inf), dtype=np.float32)
        high = np.full((obs_size,), np.float32(np.inf), dtype=np.float32)
        self.observation_spaces = {
            a: spaces.Box(low=low, high=high, dtype=np.float32) for a in self.agents
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
        # Store previous step values for delta-based rewards
        self._prev_actions = {a: np.zeros(num_actuators_per_agent, dtype=np.float32) for a in self.agents}
        self._prev_state = {}  # dist, dist_torso, engagement, facing, posture, height per agent

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _get_obs(self, agent):
        idx = 0 if agent == "fencer_a" else 1
        start_q, end_q = idx * (self.model.nq//2), (idx+1) * (self.model.nq//2)
        start_v, end_v = idx * (self.model.nv//2), (idx+1) * (self.model.nv//2)
        
        qpos = self.data.qpos[start_q:end_q]
        qvel = self.data.qvel[start_v:start_v+3] # Root linear vel
        qvel_ang = self.data.qvel[start_v+3:start_v+6] # Root angular vel
        qvel_joints = self.data.qvel[start_v+6:end_v] # Joint vels
        
        # 1. Torso Orientation (to convert global to local)
        my_torso_id = self.body_ids[agent]
        my_torso_xmat = self.data.xmat[my_torso_id].reshape(3, 3)
        
        # 2. Translation invariance: Exclude root X and Y (qpos[0], qpos[1])
        # Keep root Z, orientation, and joints. Size: 28 - 2 = 26.
        obs_q = qpos[2:] 
        
        # 3. Relative positions in LOCAL frame
        my_torso_pos = self.data.xipos[my_torso_id]
        my_tip_pos = self.data.site_xpos[self.tip_ids[agent]]
        
        opp_agent = "fencer_b" if agent == "fencer_a" else "fencer_a"
        opp_torso_pos = self.data.xipos[self.body_ids[opp_agent]]
        
        # Global relative vectors
        rel_torso_opp_glob = opp_torso_pos - my_torso_pos
        rel_tip_opp_glob = opp_torso_pos - my_tip_pos
        
        # Project into local torso frame (Torso local forward/side/up)
        rel_torso_opp_loc = my_torso_xmat.T @ rel_torso_opp_glob
        rel_tip_opp_loc = my_torso_xmat.T @ rel_tip_opp_glob
        
        # 4. Velocities in LOCAL frame
        obs_v_loc = my_torso_xmat.T @ qvel
        obs_v_ang_loc = my_torso_xmat.T @ qvel_ang
        
        # Total size: 26 + 3 + 3 + 21 + 3 + 3 = 59. Ensure float32 for Gymnasium.
        out = np.concatenate([
            obs_q,
            obs_v_loc, obs_v_ang_loc, qvel_joints,
            rel_torso_opp_loc, rel_tip_opp_loc,
        ])
        out = np.asarray(out, dtype=np.float32)
        # Prevent NaNs/Infs from unstable physics (falls, explosions)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, -100.0, 100.0)
        return out

    def _get_state_scalars(self, agent):
        """Compute scalar state values for delta-based rewards."""
        opp = "fencer_b" if agent == "fencer_a" else "fencer_a"
        my_torso_pos = self.data.xipos[self.body_ids[agent]]
        my_tip_pos = self.data.site_xpos[self.tip_ids[agent]]
        opp_torso_pos = self.data.xipos[self.body_ids[opp]]
        my_torso_xmat = self.data.xmat[self.body_ids[agent]].reshape(3, 3)
        rel_opp_glob = opp_torso_pos - my_torso_pos
        dist_torso = np.linalg.norm(rel_opp_glob) + 1e-6
        rel_opp_loc = my_torso_xmat.T @ rel_opp_glob
        dist = np.linalg.norm(my_tip_pos - opp_torso_pos)
        # Engagement score: higher when in 1-2.5m range
        if dist_torso < 1.0:
            engagement = -(1.0 - dist_torso)
        elif dist_torso <= 2.5:
            engagement = 1.0
        elif dist_torso <= 4.0:
            engagement = 0.3
        else:
            engagement = -0.5
        facing = np.clip(rel_opp_loc[0] / dist_torso, 0, 1)
        posture = np.clip(my_torso_xmat[2, 2], 0, 1)
        height_err = abs(my_torso_pos[2] - 1.28)
        return dict(dist=dist, dist_torso=dist_torso, engagement=engagement,
                    facing=facing, posture=posture, height_err=height_err)

    def step(self, actions):
        # actions is a dict: {"fencer_a": [...], "fencer_b": [...]}
        # Save previous state before physics step (for delta-based rewards)
        prev_state = {}
        for a in self.agents:
            prev_state[a] = self._prev_state.get(a, self._get_state_scalars(a))

        joint_actions = np.concatenate([actions["fencer_a"], actions["fencer_b"]])
        self.data.ctrl[:] = np.clip(joint_actions, -1, 1)

        mujoco.mj_step(self.model, self.data)

        rewards = {}
        terminations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        for agent in self.agents:
            opp = "fencer_b" if agent == "fencer_a" else "fencer_a"
            prev = prev_state[agent]
            curr = self._get_state_scalars(agent)
            self._prev_state[agent] = dict(curr)

            my_torso_pos = self.data.xipos[self.body_ids[agent]]
            dist = curr["dist"]

            # Delta-based shaping: reward = scale * (improvement)
            # 1. Distance: closer tip-to-torso is better (supports hitting) → reward (prev - curr)
            reward_dist = 0.25 * (prev["dist"] - dist)

            # 2. Engagement: moved toward better zone → reward (curr - prev)
            reward_engagement = 0.2 * (curr["engagement"] - prev["engagement"])

            # 3. Facing: turned more toward opponent → reward (curr - prev)
            reward_facing = 0.1 * (curr["facing"] - prev["facing"])

            # 4. Posture: more upright → reward (curr - prev)
            reward_posture = 0.15 * (curr["posture"] - prev["posture"])

            # 5. Height: moved toward target height → reward (prev_err - curr_err)
            reward_height = 0.1 * (prev["height_err"] - curr["height_err"])

            # 6. Survival bonus (small; hitting should dominate)
            reward_alive = 0.02

            # 7. Control penalty + action smoothness
            reward_ctrl = -0.001 * np.square(actions[agent]).sum()
            action_delta = np.abs(actions[agent] - self._prev_actions[agent])
            reward_smooth = -0.02 * action_delta.mean()
            self._prev_actions[agent] = np.array(actions[agent], dtype=np.float32)

            # 8. Hit bonus (primary goal: landing a hit)
            HIT_REWARD = 15.0
            reward_hit = 0.0
            if dist < 0.1:
                reward_hit = HIT_REWARD
                terminations = {a: True for a in self.agents}
            elif dist < 0.5:
                reward_hit = 0.4 / (dist + 0.1)  # Stronger shaping when approaching to hit

            # 9. Got-hit penalty. Zero-sum: opponent's hit = my loss (same magnitude)
            opp_tip_pos = self.data.site_xpos[self.tip_ids[opp]]
            dist_got_hit = np.linalg.norm(opp_tip_pos - my_torso_pos)
            reward_got_hit = 0.0
            if dist_got_hit < 0.1:
                reward_got_hit = -HIT_REWARD if self.zero_sum else -5.0
                terminations = {a: True for a in self.agents}

            # 10. Fall penalty & termination
            my_height = my_torso_pos[2]
            reward_fall = 0.0
            if my_height < 0.7:
                reward_fall = -1.0
                terminations = {a: True for a in self.agents}
            elif my_height < 0.38:
                reward_fall = -0.1

            # 11. Out-of-piste penalty
            reward_piste = 0.0
            if abs(my_torso_pos[0]) > 7.5 or abs(my_torso_pos[1]) > 1.5:
                reward_piste = -1.0
                terminations = {a: True for a in self.agents}

            r = (reward_alive + reward_dist + reward_engagement + reward_facing +
                 reward_posture + reward_height + reward_ctrl + reward_smooth +
                 reward_hit + reward_got_hit + reward_fall + reward_piste)
            rewards[agent] = np.clip(float(r), -15.0, 20.0)  # Allow hit bonus to pass through

        observations = {a: self._get_obs(a) for a in self.agents}
        truncations = {a: False for a in self.agents}
        
        if all(terminations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._prev_state = {}
        for a in self.agents:
            self._prev_actions[a][:] = 0

        # Proper Gymnasium seeding
        if seed is not None:
            np.random.seed(seed)
        
        mujoco.mj_resetData(self.model, self.data)
        
        # Add small random noise to initial joint positions to help exploration
        noise_q = np.random.uniform(-0.01, 0.01, size=((self.model.nq//2)-7))
        self.data.qpos[7:self.model.nq//2] += noise_q
        self.data.qpos[self.model.nq//2 + 7:] += noise_q
        
        mujoco.mj_forward(self.model, self.data)
        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def render(self, mode=None):
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