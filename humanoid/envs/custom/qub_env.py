# SPDX-License-Identifier: BSD-3-Clause

import torch

from humanoid.envs.base.legged_robot_config import LeggedRobotCfg
from humanoid.envs.base.legged_robot import LeggedRobot
from humanoid.envs.custom.humanoid_env import XBotLFreeEnv


class QUBEnv(XBotLFreeEnv):
    """QUB-specific humanoid environment.

    The base XBot environment is reused for the common humanoid rewards, while
    QUB-specific DOF indexing and observation sizes are handled here.
    """

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self._dof_name_to_idx = {}
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()
        self._dof_name_to_idx = {name: i for i, name in enumerate(self.dof_names)}

    def _set_ref_joint(self, name, value):
        idx = self._dof_name_to_idx.get(name)
        if idx is not None:
            self.ref_dof_pos[:, idx] = value

    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1

        sin_pos_l[sin_pos_l > 0] = 0
        self._set_ref_joint('L_Hip_pitch_joint', sin_pos_l * scale_1)
        self._set_ref_joint('L_Thigh_pitch_3_joint', sin_pos_l * scale_2)
        self._set_ref_joint('L_Ankle_pitch_joint', -sin_pos_l * scale_1)

        sin_pos_r[sin_pos_r < 0] = 0
        self._set_ref_joint('R_Hip_pitch_joint', sin_pos_r * scale_1)
        self._set_ref_joint('R_Thigh_pitch_3_joint', sin_pos_r * scale_2)
        self._set_ref_joint('R_Ankle_pitch_joint', -sin_pos_r * scale_1)

        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0
        self.ref_action = 2 * self.ref_dof_pos

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        num_actions = self.cfg.env.num_actions

        noise_vec[0:5] = 0.
        noise_vec[5:5 + num_actions] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[5 + num_actions:5 + 2 * num_actions] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[5 + 2 * num_actions:5 + 3 * num_actions] = 0.
        noise_vec[5 + 3 * num_actions:5 + 3 * num_actions + 3] = noise_scales.ang_vel * self.obs_scales.ang_vel
        noise_vec[5 + 3 * num_actions + 3:5 + 3 * num_actions + 6] = noise_scales.quat * self.obs_scales.quat
        return noise_vec

    def step(self, actions):
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action
        actions = torch.clip(actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        delay = torch.rand((self.num_envs, 1), device=self.device) * self.cfg.domain_rand.action_delay
        actions = (1 - delay) * actions + delay * self.actions
        actions += self.cfg.domain_rand.action_noise * torch.randn_like(actions) * actions
        return LeggedRobot.step(self, actions)

    def _reward_default_joint_pos(self):
        joint_diff = self.dof_pos - self.default_joint_pd_target
        yaw_roll_names = (
            'L_Hip_roll_joint',
            'L_Hip_yaw_joint',
            'R_Hip_roll_joint',
            'R_Hip_yaw_joint',
            'Torso_yaw_joint',
        )
        yaw_roll_indices = [
            self._dof_name_to_idx[name]
            for name in yaw_roll_names
            if name in self._dof_name_to_idx
        ]
        if yaw_roll_indices:
            yaw_roll = torch.norm(joint_diff[:, yaw_roll_indices], dim=1)
        else:
            yaw_roll = torch.zeros(self.num_envs, device=self.device)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)
