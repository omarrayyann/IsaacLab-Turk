# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym


##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Rotation-Calibration-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaRotationCalibrationEnvCfg",
    },
)

gym.register(
    id="Isaac-Rotation-Calibration-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaRotationCalibrationEnvCfg_PLAY",
    },
)


##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Rotation-Calibration-Franka-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaRotationCalibrationEnvCfg",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Rotation-Calibration-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:FrankaRotationCalibrationEnvCfg",
    },
    disable_env_checker=True,
)

##
# Operational Space Control
##

gym.register(
    id="Isaac-Rotation-Calibration-Franka-OSC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.osc_env_cfg:FrankaRotationCalibrationEnvCfg",
    },
)

gym.register(
    id="Isaac-Rotation-Calibration-Franka-OSC-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.osc_env_cfg:FrankaRotationCalibrationEnvCfg_PLAY",
    },
)
