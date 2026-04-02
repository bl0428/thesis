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
            zero_sum: Ignored (kept for API compatibility). Reward uses old non-zero-sum logic.
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
        # Tip geom IDs — only the small sphere at the tip scores a valid touch
        self.epee_tip_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "epee_tip_geom")
        self.epee_tip_geom_B_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "epee_tip_geom_B")
        for _name, _gid in (
            ("epee_tip_geom", self.epee_tip_geom_id),
            ("epee_tip_geom_B", self.epee_tip_geom_B_id),
        ):
            if _gid < 0:
                raise RuntimeError(f"MuJoCo name not found: {_name} (mj_name2id -> {_gid})")

        torso_a = self.body_ids["fencer_a"]
        torso_b = self.body_ids["fencer_b"]
        epee_a = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "epee")
        epee_b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "epee_B")
        if epee_a < 0 or epee_b < 0:
            raise RuntimeError(f"epee body not found (epee={epee_a}, epee_B={epee_b})")

        # Full-body valid target (épée rules): every geom on that humanoid under torso / torso_B,
        # excluding the weapon subtree only. Use one BFS per root (not repeated parent walks per
        # geom) so init stays fast and cannot spin on a bad parent chain.
        in_a = self._kinematic_subtree_body_mask(self.model, torso_a)
        in_b = self._kinematic_subtree_body_mask(self.model, torso_b)
        in_epee_a = self._kinematic_subtree_body_mask(self.model, epee_a)
        in_epee_b = self._kinematic_subtree_body_mask(self.model, epee_b)
        self._a_target_geom_ids = set()
        self._b_target_geom_ids = set()
        for gid in range(self.model.ngeom):
            bid = int(self.model.geom_bodyid[gid])
            if bid < 0:
                continue
            if in_b[bid] and not in_epee_b[bid]:
                self._b_target_geom_ids.add(gid)
            if in_a[bid] and not in_epee_a[bid]:
                self._a_target_geom_ids.add(gid)

        # Map agent -> geom ids on that agent's body that count as hit targets for the opponent.
        self._target_geom_ids_by_agent = {
            "fencer_a": frozenset(self._a_target_geom_ids),
            "fencer_b": frozenset(self._b_target_geom_ids),
        }

        # Precompute per-agent joint indices and limits for joint-limit penalty.
        # Skip the 7 root DOFs (3 pos + 4 quat for the freejoint) per humanoid.
        njnt = self.model.njnt
        self._joint_qpos_slices = {}
        self._joint_ranges = {}
        for agent_name, jnt_offset in (("fencer_a", 0), ("fencer_b", njnt // 2)):
            indices = []
            lo_list, hi_list = [], []
            for j in range(jnt_offset, jnt_offset + njnt // 2):
                if not self.model.jnt_limited[j]:
                    continue
                if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    continue
                adr = int(self.model.jnt_qposadr[j])
                indices.append(adr)
                lo_list.append(float(self.model.jnt_range[j, 0]))
                hi_list.append(float(self.model.jnt_range[j, 1]))
            self._joint_qpos_slices[agent_name] = np.array(indices, dtype=np.intp)
            self._joint_ranges[agent_name] = (
                np.array(lo_list, dtype=np.float64),
                np.array(hi_list, dtype=np.float64),
            )

    @staticmethod
    def _kinematic_subtree_body_mask(model, root_body_id):
        """
        Boolean mask length nbody: True for root_body_id and all descendant bodies.
        Built via adjacency from body_parentid (O(nbody)); avoids parent-chain loops.
        """
        if root_body_id < 0:
            return np.zeros(model.nbody, dtype=bool)
        n = int(model.nbody)
        children = [[] for _ in range(n)]
        parent = np.asarray(model.body_parentid, dtype=np.int32).ravel()
        for b in range(n):
            p = int(parent[b])
            if p >= 0:
                children[p].append(b)
        mask = np.zeros(n, dtype=bool)
        stack = [int(root_body_id)]
        while stack:
            b = stack.pop()
            if b < 0 or b >= n or mask[b]:
                continue
            mask[b] = True
            stack.extend(children[b])
        return mask

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

    def step(self, actions):
        joint_actions = np.concatenate([actions["fencer_a"], actions["fencer_b"]])
        self.data.ctrl[:] = np.clip(joint_actions, -1, 1)

        mujoco.mj_step(self.model, self.data)

        rewards = {}
        terminations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        # Hit detection: only the tip geom (small sphere at blade end) vs opponent body counts.
        # Full blade contacts are physics-only; they don't score.
        hit_by_a = False  # A's tip touched B's body
        hit_by_b = False  # B's tip touched A's body
        for i in range(self.data.ncon):
            g1, g2 = int(self.data.contact[i].geom1), int(self.data.contact[i].geom2)
            if g1 == self.epee_tip_geom_id and g2 in self._b_target_geom_ids:
                hit_by_a = True
            elif g2 == self.epee_tip_geom_id and g1 in self._b_target_geom_ids:
                hit_by_a = True
            elif g1 == self.epee_tip_geom_B_id and g2 in self._a_target_geom_ids:
                hit_by_b = True
            elif g2 == self.epee_tip_geom_B_id and g1 in self._a_target_geom_ids:
                hit_by_b = True
        # No distance fallback — rely solely on the tip geom contact. The dedicated tip sphere
        # (r=0.025) is large enough to avoid tunneling; proximity alone should not score.
        if hit_by_a or hit_by_b:
            terminations = {a: True for a in self.agents}

        for agent in self.agents:
            opp = "fencer_b" if agent == "fencer_a" else "fencer_a"

            my_torso_pos = self.data.xipos[self.body_ids[agent]]
            my_tip_pos = self.data.site_xpos[self.tip_ids[agent]]
            opp_torso_pos = self.data.xipos[self.body_ids[opp]]
            dist = np.linalg.norm(my_tip_pos - opp_torso_pos)

            # Reward terms use tanh for smooth bounded scaling.

            # 1. Survival bonus (small positive)
            reward_alive = 0.10

            # 2. Distance shaping (closer tip→opponent is better)
            reward_dist = -0.15 * float(np.tanh(dist / 3.0))

            # 3. Engagement distance (keep ideal range ~1.5m)
            dist_torso = float(np.linalg.norm(opp_torso_pos - my_torso_pos))
            target_dist = 1.5
            dist_err = abs(dist_torso - target_dist)
            reward_engagement = -0.15 * float(np.tanh(dist_err / 2.0))

            # 4. Velocity-direction alignment (cosine similarity)
            idx = 0 if agent == "fencer_a" else 1
            start_v = idx * (self.model.nv // 2)
            my_torso_vel = self.data.qvel[start_v : start_v + 3]
            vel_norm = float(np.linalg.norm(my_torso_vel)) + 1e-8
            vec_to_opp = opp_torso_pos - my_torso_pos
            dir_to_opp = vec_to_opp / (float(np.linalg.norm(vec_to_opp)) + 1e-8)
            cos_sim = float(np.dot(my_torso_vel, dir_to_opp)) / vel_norm
            reward_vel_align = 0.1 * cos_sim

            # 5. Balance: height + orientation (stay upright, don't tilt)
            my_height = my_torso_pos[2]
            reward_height = -0.2 * float(np.tanh(abs(my_height - 1.05) / 0.5))
            my_torso_xmat = self.data.xmat[self.body_ids[agent]].reshape(3, 3)
            upright_score = float(my_torso_xmat[2, 2])
            reward_orientation = 0.15 * (upright_score - 0.7) if upright_score > 0.7 else -0.15 * (0.7 - upright_score)
            ang_vel = self.data.qvel[start_v + 3 : start_v + 6]
            reward_stability = -0.06 * float(np.tanh(np.linalg.norm(ang_vel)))
            reward_upright = reward_height + reward_orientation + reward_stability

            # 6. Control penalty
            reward_ctrl = -0.10 * float(np.tanh(np.square(actions[agent]).sum() / 21.0))

            # 6b. Joint-limit penalty: penalize joints in the outer 10% of their range
            jnt_idx = self._joint_qpos_slices[agent]
            jnt_lo, jnt_hi = self._joint_ranges[agent]
            jnt_pos = self.data.qpos[jnt_idx]
            jnt_range = jnt_hi - jnt_lo
            margin = 0.1 * jnt_range
            lo_violation = np.maximum(jnt_lo + margin - jnt_pos, 0.0) / (margin + 1e-8)
            hi_violation = np.maximum(jnt_pos - (jnt_hi - margin), 0.0) / (margin + 1e-8)
            violation = np.maximum(lo_violation, hi_violation)
            reward_joint_limit = -0.15 * float(np.mean(violation))

            # 7. Hit bonus (capped to stay in range); use hit detection from above
            reward_hit = 0.0
            if agent == "fencer_a" and hit_by_a:
                reward_hit = 1.0
            elif agent == "fencer_b" and hit_by_b:
                reward_hit = 1.0
            elif not (hit_by_a or hit_by_b) and dist < 0.5:
                reward_hit = 0.2 * (1.0 - dist / 0.5)  # Shaping when approaching

            # 8. Fall penalty & termination
            reward_fall = 0.0
            if my_height < 0.38:
                reward_fall = -1.0
                terminations = {a: True for a in self.agents}
            elif my_height < 0.7:
                reward_fall = -0.1

            # 9. Out-of-piste penalty
            reward_piste = 0.0
            if abs(my_torso_pos[0]) > 7.5 or abs(my_torso_pos[1]) > 1.5:
                reward_piste = -1.0
                terminations = {a: True for a in self.agents}

            r = (
                reward_alive
                + reward_dist
                + reward_engagement
                + reward_vel_align
                + (reward_upright * 10.0)
                + reward_ctrl
                + reward_joint_limit
                + reward_hit
                + reward_fall
                + reward_piste
            ) * 0.5
            rewards[agent] = float(r)

        observations = {a: self._get_obs(a) for a in self.agents}
        truncations = {a: False for a in self.agents}

        for a in self.agents:
            infos[a]["hit_by_a"] = hit_by_a
            infos[a]["hit_by_b"] = hit_by_b
        
        if all(terminations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]

        # Proper Gymnasium seeding
        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0

        # Add small random noise to initial joint positions to help exploration
        noise_q = np.random.uniform(-0.01, 0.01, size=((self.model.nq // 2) - 7))
        self.data.qpos[7 : self.model.nq // 2] += noise_q
        self.data.qpos[self.model.nq // 2 + 7 :] += noise_q

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