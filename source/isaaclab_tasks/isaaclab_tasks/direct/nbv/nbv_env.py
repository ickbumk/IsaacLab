# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warp as wp

from pxr import UsdGeom, Gf

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_conjugate, sample_uniform
from isaaclab.sim.utils.stage import get_current_stage


from .utils import get_random_usd, replace_random_usd, get_camera
import numpy as np



from .nbv_env_cfg import NBVEnvCfg  # noqa: F401

class NBVEnv(DirectRLEnv):
    cfg: NBVEnvCfg

    def __init__(self, cfg: NBVEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # UR10e joint limits
        self.robot_dof_lower_limits = (
            self._robot.data.soft_joint_pos_limits.torch[0, :, 0]
            .to(device=self.device)
        )
        self.robot_dof_upper_limits = (
            self._robot.data.soft_joint_pos_limits.torch[0, :, 1]
            .to(device=self.device)
        )

        # Action is a delta joint-position command
        self.robot_dof_targets = torch.zeros(
            (self.num_envs, self._robot.num_joints),
            device=self.device,
        )

    def _setup_scene(self):
        # Robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        stage = get_current_stage()

        self._camera = get_camera(stage)



        
        self._objects = []
        self._cameras = []
        for env_id in range(self.num_envs):
            prim_path = f"/World/envs/env_{env_id}/MyModel"
            obj = get_random_usd(
                stage,
                "/home/asclab/projects/NBV/datasets/gso_usd",
                prim_path,
            )
            
            self._objects.append(obj)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(
                global_prim_paths=[self.cfg.terrain.prim_path]
            )

        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        )
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)

        targets = (
            self.robot_dof_targets
            + self.cfg.action_scale * self.dt * self.actions
        )

        self.robot_dof_targets[:] = torch.clamp(
            targets,
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits,
        )

    def _apply_action(self):
        self._robot.set_joint_position_target_index(
            target=self.robot_dof_targets
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # No NBV reward yet.
        # This is only a robot-control smoke test.
        return torch.zeros(
            self.num_envs,
            device=self.device,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(
                self.num_envs,
                device=self.device,
                dtype=torch.long,
            )

        super()._reset_idx(env_ids)

        stage = get_current_stage()
        for env_id in env_ids.tolist():
            replace_random_usd(
                stage,
                "/home/asclab/projects/NBV/datasets/gso_usd",
                self._objects[env_id],
            )

        joint_pos = self._robot.data.default_joint_pos.torch[env_ids]
        joint_vel = torch.zeros_like(joint_pos)

        self._robot.set_joint_position_target_index(
            target=joint_pos,
            env_ids=env_ids,
        )

        self._robot.write_joint_position_to_sim_index(
            position=joint_pos,
            env_ids=env_ids,
        )

        self._robot.write_joint_velocity_to_sim_index(
            velocity=joint_vel,
            env_ids=env_ids,
        )

        self.robot_dof_targets[env_ids] = joint_pos

    def _get_observations(self) -> dict:
        # Joint positions + velocities
        obs = torch.cat(
            (
                self._robot.data.joint_pos.torch,
                self._robot.data.joint_vel.torch,
            ),
            dim=-1,
        )

        return {
            "policy": obs,
        }
