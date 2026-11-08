"""Stand/sit task: one policy that switches between standing and sitting on a binary command.

Reuses the gap study's randomized, realistic setup (``Go2FlatRandEnvCfg``) and swaps the velocity
command and rewards for a pose-mode command and pose rewards.

Sit pose, derived from the Go2's geometry (thigh = calf = 0.213 m, hips 0.193 m fore/aft of the body
center): body pitched nose-up by ``SIT_PITCH``; front feet straight under the shoulders (front hips
~0.39 m high); rear legs folded so the rear hips drop to ~0.21 m with the knees just above the floor.
All angles sit inside the 0.9 soft joint limits.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity import mdp

from .envs import Go2FlatRandEnvCfg, _push

SIT_PITCH = 0.48  # rad, nose up
SIT_JOINT_POS = {
    "F[L,R]_thigh_joint": 0.98,
    "F[L,R]_calf_joint": -1.0,
    "R[L,R]_thigh_joint": 1.04,
    "R[L,R]_calf_joint": -2.06,
}  # hips keep their default (+-0.1)


class PoseModeCommand(CommandTerm):
    """Binary command: 0 = stand (default pose, level body), 1 = sit (``SIT_JOINT_POS``, pitched body)."""

    cfg: PoseModeCommandCfg

    def __init__(self, cfg: PoseModeCommandCfg, env):
        """Build the stand and sit targets from the robot's defaults and ``cfg``."""
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.mode = torch.zeros(self.num_envs, 1, device=self.device)
        self.stand_pose = self.robot.data.default_joint_pos.torch[0].clone()
        self.sit_pose = self.stand_pose.clone()
        for expr, value in cfg.sit_joint_pos.items():
            ids, _ = self.robot.find_joints(expr)
            self.sit_pose[ids] = value
        self.stand_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        self.sit_gravity = torch.tensor([-math.sin(cfg.sit_pitch), 0.0, -math.cos(cfg.sit_pitch)], device=self.device)
        self.metrics["pose_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tilt_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sit_fraction"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Current mode of each env, shape ``(num_envs, 1)``; this is what the policy observes."""
        return self.mode

    @property
    def sitting(self) -> torch.Tensor:
        """Boolean mask of the envs currently commanded to sit."""
        return self.mode[:, 0] > 0.5

    @property
    def target_joint_pos(self) -> torch.Tensor:
        """Target joint positions of each env under its current command."""
        return torch.where(self.sitting[:, None], self.sit_pose, self.stand_pose)

    @property
    def target_gravity(self) -> torch.Tensor:
        """Target gravity direction in the body frame of each env under its current command."""
        return torch.where(self.sitting[:, None], self.sit_gravity, self.stand_gravity)

    def _resample_command(self, env_ids: Sequence[int] | slice):
        """Draw a new command for the given envs: sit with probability ``cfg.sit_probability``."""
        draw = (torch.rand(self.num_envs, device=self.device) < self.cfg.sit_probability).float()
        self.mode[env_ids, 0] = draw[env_ids]

    def _update_command(self):
        """Do nothing: the command only changes when it is resampled."""
        pass

    def _update_metrics(self):
        """Add this step's pose error, tilt error and sit share to the episode means."""
        # means over the episode (metrics are logged and cleared at reset)
        steps = self._env.max_episode_length
        q = self.robot.data.joint_pos.torch
        g = self.robot.data.projected_gravity_b.torch
        self.metrics["pose_error"] += (q - self.target_joint_pos).abs().mean(dim=1) / steps
        self.metrics["tilt_error"] += torch.linalg.norm(g - self.target_gravity, dim=1) / steps
        self.metrics["sit_fraction"] += self.sitting.float() / steps


@configclass
class PoseModeCommandCfg(CommandTermCfg):
    """Config for ``PoseModeCommand``: resampling time, sit probability and sit pose."""

    class_type: type = PoseModeCommand
    asset_name: str = "robot"
    resampling_time_range: tuple[float, float] = (4.0, 6.0)
    sit_probability: float = 0.5
    sit_pitch: float = SIT_PITCH
    sit_joint_pos: dict = SIT_JOINT_POS


def _pose_term(env, command_name: str) -> PoseModeCommand:
    """Return the environment's ``PoseModeCommand`` term."""
    return env.command_manager.get_term(command_name)


def pose_tracking_exp(env, command_name: str, std: float) -> torch.Tensor:
    """Reward in [0, 1] that rises as the joints approach the commanded pose."""
    term = _pose_term(env, command_name)
    # mean over joints, so std is a per-joint RMS error; a sum over 12 joints saturates to 0
    err = torch.mean((term.robot.data.joint_pos.torch - term.target_joint_pos) ** 2, dim=1)
    return torch.exp(-err / std**2)


def tilt_tracking_exp(env, command_name: str, std: float) -> torch.Tensor:
    """Reward in [0, 1] that rises as the body's gravity direction approaches the commanded tilt."""
    term = _pose_term(env, command_name)
    err = torch.sum((term.robot.data.projected_gravity_b.torch - term.target_gravity) ** 2, dim=1)
    return torch.exp(-err / std**2)


def feet_off_ground(env, sensor_cfg: SceneEntityCfg, sitting_only_command: str | None = None) -> torch.Tensor:
    """Number of lifted feet; with ``sitting_only_command``, counted only while that command says sit."""
    sensor = env.scene.sensors[sensor_cfg.name]
    lifted = (sensor.data.current_air_time.torch[:, sensor_cfg.body_ids] > 0.0).float().sum(dim=1)
    if sitting_only_command is not None:
        lifted *= _pose_term(env, sitting_only_command).sitting.float()
    return lifted


def base_lin_vel_xy_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Squared horizontal body speed; penalized so the robot stays in place."""
    return torch.sum(base_mdp.base_lin_vel(env, asset_cfg)[:, :2] ** 2, dim=1)


@configclass
class SitRewardsCfg:
    """Reward terms and weights of the stand/sit task."""

    pose_tracking = RewTerm(func=pose_tracking_exp, weight=1.5, params={"command_name": "pose_mode", "std": 0.5})
    tilt_tracking = RewTerm(func=tilt_tracking_exp, weight=1.0, params={"command_name": "pose_mode", "std": 0.25})
    feet_off_ground = RewTerm(
        func=feet_off_ground, weight=-0.5, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")}
    )
    base_lin_vel_xy = RewTerm(func=base_lin_vel_xy_l2, weight=-1.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)


@configclass
class Go2SitEnvCfg(Go2FlatRandEnvCfg):
    """Stand/sit on a binary command, on top of the randomized walking setup."""

    def __post_init__(self):
        """Swap the velocity command and walking rewards for the pose command and pose rewards."""
        super().__post_init__()
        self.commands.base_velocity = None
        self.commands.pose_mode = PoseModeCommandCfg()
        self.observations.policy.velocity_commands = None
        self.observations.policy.pose_mode = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_mode"})
        self.rewards = SitRewardsCfg()
        self.events.push_robot = _push(0.5, (5.0, 10.0))


@configclass
class Go2SitPushEnvCfg(Go2SitEnvCfg):
    """Stand/sit hardened against shoves: random 0-1.5 m/s pushes instead of a fixed 0.5 m/s."""

    def __post_init__(self):
        """Replace the training pushes with a random 0-1.5 m/s range."""
        super().__post_init__()
        self.events.push_robot = _push((0.0, 1.5), (5.0, 10.0))


@configclass
class Go2SitStepEnvCfg(Go2SitPushEnvCfg):
    """Push-hardened stand/sit that may step to recover while standing.

    Under hard pushes the standing robot fell more than the sitting one: the lifted-feet penalty
    kept it from taking recovery steps. Here that penalty applies only while sitting.
    """

    def __post_init__(self):
        """Make the lifted-feet penalty apply only while the command says sit."""
        super().__post_init__()
        self.rewards.feet_off_ground.params["sitting_only_command"] = "pose_mode"
