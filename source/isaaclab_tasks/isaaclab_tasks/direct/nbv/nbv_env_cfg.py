# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

import os
import numpy as np


@configclass
class NBVEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2

    action_space = 6
    observation_space = 12
    state_space = 0

    action_scale = 1.0
    dof_velocity_scale = 0.1

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, 
        env_spacing=3.0, 
        replicate_physics=False, 
        clone_in_fabric=True
    )

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/UR10e",
        spawn=sim_utils.UsdFileCfg(
            usd_path=(
                f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10e/ur10e.usd"
            ),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": 1.157,
                "shoulder_lift_joint": -1.066,
                "elbow_joint": -0.155,
                "wrist_1_joint": -2.239,
                "wrist_2_joint": -1.841,
                "wrist_3_joint": 1.003,
            },
            pos=(0.5, 0.0, 0.0),
            rot=(0.0, 0.0, 1.0, 0.0),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "shoulder_pan_joint", 
                    "shoulder_lift_joint", 
                    "elbow_joint", 
                    "wrist_1_joint", 
                    "wrist_2_joint", 
                    "wrist_3_joint",
                ],
                effort_limit_sim=150.0,
                stiffness=400.0,
                damping=4.0,
            ),
        },
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # random object
    # usd_parent = "/home/asclab/projects/NBV/datasets/gso_usd"
    # models = os.listdir(usd_parent)
    # model_name = np.random.choice(models)
    # model_path = os.path.join(usd_parent, model_name, model_name + ".usd")

    # random_object = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/MyModel",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=model_path,
    #         activate_contact_sensors=False,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             disable_gravity=False,
    #             max_depenetration_velocity=5.0,
    #         ),
    #     ),
    #     init_state = RigidObjectCfg.InitialStateCfg(
    #         pos=(0.0, 0.0, 0.0),
    #         rot=(0.0, 0.0, 1.0, 0.0)
    #     ),
    # )

    action_scale = 7.5
    dof_velocity_scale = 0.1

    # print(self.robot.num_joints)
    # print(self.robot.joint_names)
