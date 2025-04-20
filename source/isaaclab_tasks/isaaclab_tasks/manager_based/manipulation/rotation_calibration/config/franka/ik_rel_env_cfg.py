# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class FrankaRotationCalibrationEnvCfg(joint_pos_env_cfg.FrankaRotationCalibrationEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = self.scene.robot.replace(
            spawn=sim_utils.UsdFileCfg(
        usd_path=f"Rotation_Calibration_Robot.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
            )
        # articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        #     enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        # ),
        # ),
        #     init_state=ArticulationCfg.InitialStateCfg(
        #     joint_pos={
        #         "panda_joint1": 0.0,
        #         "panda_joint2": -0.569,
        #         "panda_joint3": 0.0,
        #         "panda_joint4": -2.810,
        #         "panda_joint5": 0.0,
        #         "panda_joint6": 3.037,
        #         "panda_joint7": 0.741,
        #     },
        # ),

        #     actuators={
        # "panda_shoulder": ImplicitActuatorCfg(
        #     joint_names_expr=["panda_joint[1-4]"],
        #     effort_limit=87.0,
        #     velocity_limit=2.175,
        #     stiffness=80.0,
        #     damping=4.0,
        # ),
        # "panda_forearm": ImplicitActuatorCfg(
        #     joint_names_expr=["panda_joint[5-7]"],
        #     effort_limit=12.0,
        #     velocity_limit=2.61,
        #     stiffness=80.0,
        #     damping=4.0,
        # )
        # }
        )


        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )


@configclass
class FrankaRotationCalibrationEnvCfg_PLAY(FrankaRotationCalibrationEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False