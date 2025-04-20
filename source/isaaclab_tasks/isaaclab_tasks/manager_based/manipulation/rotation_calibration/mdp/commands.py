from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique
import isaaclab.sim as sim_utils
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformPoseRotationCommand(CommandTerm):
    """Command generator for generating pose commands uniformly with spherical sampling for position,
    centered around a specified point.
    """
    cfg: CommandTermCfg

    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # Create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)

        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

        # Cache the center in a tensor for convenience
        self.center_b = torch.tensor(cfg.ranges.center, dtype=torch.float, device=self.device)
        # (Note: shape is (3,). We'll broadcast it as needed.)

    def __str__(self) -> str:
        msg = "UniformPoseRotationCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position (x,y,z),
        followed by the quaternion orientation (w,x,y,z).
        """
        return self.pose_command_b

    def _update_metrics(self):
        # Transform command from base frame to world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        # Compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_state_w[:, self.body_idx, :3],
            self.robot.data.body_state_w[:, self.body_idx, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        """
        Samples a new random position in a spherical shell [r_min, r_max] around
        self.center_b, and a random orientation (roll/pitch/yaw).
        """
        # 1) Position
        radii = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.ranges.radius)
        directions = torch.randn(len(env_ids), 3, device=self.device)
        directions /= torch.norm(directions, dim=1, keepdim=True)  # Make it unit length

        # Flip directions that have a negative x-component so x is always positive
        negative_x_mask = directions[:, 0] < 0
        directions[negative_x_mask] *= -1
        
        offsets = (radii.unsqueeze(-1) * directions)  # shape: (num_envs, 3)

        # center_b is shape (3,) so we expand it for the env_ids:
        center = self.center_b.unsqueeze(0)  # shape: (1, 3)
        pos = center + offsets  # shape: (num_envs, 3)

        self.pose_command_b[env_ids, 0:3] = pos

        # 2) Orientation
        euler_angles = torch.zeros(len(env_ids), 3, device=self.device)
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)   # roll
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)  # pitch
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)    # yaw
        quat = quat_from_euler_xyz(
            euler_angles[:, 0],
            euler_angles[:, 1],
            euler_angles[:, 2]
        )
        # Optionally ensure real part is positive (unique quaternion)
        if self.cfg.make_quat_unique:
            quat = quat_unique(quat)
        self.pose_command_b[env_ids, 3:] = quat

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        # Update the markers
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        body_link_state_w = self.robot.data.body_state_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_link_state_w[:, :3], body_link_state_w[:, 3:7])


@configclass
class UniformPoseRotationCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator with spherical radius sampling.

    We now have a `center` parameter indicating the center point of the spherical distribution.
    """

    class_type: type = UniformPoseRotationCommand

    asset_name: str = MISSING
    body_name: str = MISSING
    make_quat_unique: bool = False

    @configclass
    class Ranges:
        """
        Ranges for spherical position and roll/pitch/yaw orientation,
        plus center point of the distribution.
        """
        radius: tuple[float, float] = MISSING
        """(r_min, r_max): minimum and maximum radius from `center` to sample."""

        center: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """(cx, cy, cz): center point for spherical sampling."""

        roll: tuple[float, float] = MISSING
        pitch: tuple[float, float] = MISSING
        yaw: tuple[float, float] = MISSING

    ranges: Ranges = MISSING

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_pose",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"goal_sphere.usd",
                scale=(0.5, 0.5, 0.5),
            )
    }
)
    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )

    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
