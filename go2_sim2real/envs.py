"""Environment configs for the gap study.

Both training variants share a realistic interface: joint-position targets at 50 Hz through the
Go2 DC-motor PD model, and policy inputs limited to what a real Go2 can sense (IMU angular
velocity, gravity direction, joint states, last action, command). True body linear velocity is
removed from the observations.

- ``Go2FlatNoRandEnvCfg``: an "ideal" simulator. Fixed friction and mass, no pushes, no sensor
  noise, no action delay.
- ``Go2FlatRandEnvCfg``: domain randomization over friction, payload, center of mass, PD gains,
  pushes, sensor noise and 0-2 steps of action delay.

``apply_eval_change`` turns the ideal config into one "fake real" test world with a single change.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.core.velocity import mdp
from isaaclab_tasks.core.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg

NOMINAL_STATIC_FRICTION = 0.8
NOMINAL_DYNAMIC_FRICTION = 0.6
# Stock observation noise (uniform half-widths) for the IMU terms.
ANG_VEL_NOISE = 0.2
GRAVITY_NOISE = 0.05


class DelayedJointPositionAction(JointPositionAction):
    """Joint-position action applied ``d`` policy steps late, with ``d`` drawn per env at reset.

    The policy's own last-action observation still sees the action it sent, as on a real robot.
    """

    cfg: DelayedJointPositionActionCfg

    def __init__(self, cfg: DelayedJointPositionActionCfg, env):
        """Create the action history and draw a first delay for every env."""
        super().__init__(cfg, env)
        lo, hi = cfg.delay_range
        self._history = torch.zeros(hi + 1, self.num_envs, self.action_dim, device=self.device)
        self._delay = torch.randint(lo, hi + 1, (self.num_envs,), device=self.device)
        self._env_ids = torch.arange(self.num_envs, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        """Store the newest action and apply the one sent ``d`` steps ago in each env."""
        self._history = torch.roll(self._history, 1, dims=0)
        self._history[0] = actions
        super().process_actions(self._history[self._delay, self._env_ids])

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear the history of the given envs and draw a new delay for each."""
        super().reset(env_ids)
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = self._env_ids
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._history[:, env_ids] = 0.0
        lo, hi = self.cfg.delay_range
        self._delay[env_ids] = torch.randint(lo, hi + 1, (len(env_ids),), device=self.device)


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """Config for ``DelayedJointPositionAction``."""

    class_type: type = DelayedJointPositionAction
    delay_range: tuple[int, int] = (0, 0)
    """Inclusive range of the per-episode delay, in policy steps (20 ms each)."""


def push_fixed_speed(
    env, env_ids: torch.Tensor, speed: float | tuple[float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """Add a horizontal velocity in a random direction: exactly ``speed`` m/s, or uniform in a ``(min, max)`` range."""
    asset = env.scene[asset_cfg.name]
    vel_w = asset.data.root_vel_w.torch[env_ids].clone()
    n = vel_w.shape[0]
    theta = torch.rand(n, device=vel_w.device) * 2 * math.pi
    if isinstance(speed, (tuple, list)):
        speed = speed[0] + (speed[1] - speed[0]) * torch.rand(n, device=vel_w.device)
    vel_w[:, 0] += speed * torch.cos(theta)
    vel_w[:, 1] += speed * torch.sin(theta)
    asset.write_root_velocity_to_sim_index(root_velocity=vel_w, env_ids=env_ids)


def _friction(static_range, dynamic_range) -> EventTerm:
    """Startup event that sets robot-body friction, drawn uniformly from the given ranges."""
    return EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": static_range,
            "dynamic_friction_range": dynamic_range,
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )


def _set_ground_friction(cfg, value: float) -> None:
    """Set the ground's static and dynamic friction to ``value``."""
    material = cfg.scene.terrain.physics_material
    material.static_friction = value
    material.dynamic_friction = value
    cfg.sim.physics_material = material


def _payload(kg_range) -> EventTerm:
    """Startup event that adds a mass in ``kg_range`` to the robot's base."""
    return EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": kg_range,
            "operation": "add",
            "distribution": "uniform",
        },
    )


def _motor_gains(scale_range) -> EventTerm:
    """Startup event that scales PD stiffness and damping by a factor in ``scale_range``."""
    return EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": scale_range,
            "damping_distribution_params": scale_range,
            "operation": "scale",
            "distribution": "uniform",
        },
    )


def _push(speed: float | tuple[float, float], interval_s: tuple[float, float]) -> EventTerm:
    """Interval event that applies ``push_fixed_speed`` every ``interval_s`` seconds."""
    return EventTerm(func=push_fixed_speed, mode="interval", interval_range_s=interval_s, params={"speed": speed})


def _set_imu_noise(cfg, scale: float) -> None:
    """Noise only on the IMU terms, ``scale`` times the stock level; 0 disables corruption."""
    policy = cfg.observations.policy
    for term in ("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel"):
        getattr(policy, term).noise = None
    if scale > 0:
        policy.base_ang_vel.noise = Unoise(n_min=-ANG_VEL_NOISE * scale, n_max=ANG_VEL_NOISE * scale)
        policy.projected_gravity.noise = Unoise(n_min=-GRAVITY_NOISE * scale, n_max=GRAVITY_NOISE * scale)
    policy.enable_corruption = scale > 0


def _realistic_interface(cfg) -> None:
    """Remove true body velocity from the observations and install the delayable action term."""
    cfg.observations.policy.base_lin_vel = None
    cfg.actions.joint_pos = DelayedJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=cfg.actions.joint_pos.scale, use_default_offset=True
    )


@configclass
class Go2FlatNoRandEnvCfg(UnitreeGo2FlatEnvCfg):
    """Walking in the ideal simulator: fixed physics, no pushes, noise or delay."""

    def __post_init__(self):
        """Apply the realistic interface and turn every randomization off."""
        super().__post_init__()
        _realistic_interface(self)
        self.events.physics_material = _friction((NOMINAL_STATIC_FRICTION,) * 2, (NOMINAL_DYNAMIC_FRICTION,) * 2)
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.push_robot = None
        self.observations.policy.enable_corruption = False


@configclass
class Go2FlatRandEnvCfg(UnitreeGo2FlatEnvCfg):
    """Walking with domain randomization over physics, pushes, sensor noise and delay."""

    def __post_init__(self):
        """Apply the realistic interface and set the randomization ranges."""
        super().__post_init__()
        _realistic_interface(self)
        # Contact friction is max(ground, foot): a low-friction ground lets the foot range below take effect.
        _set_ground_friction(self, 0.3)
        self.events.physics_material = _friction((0.3, 1.25), (0.25, 1.0))
        self.events.add_base_mass = _payload((-1.0, 5.0))
        # base_com keeps the stock +-5 cm / +-1 cm shift
        self.events.motor_gains = _motor_gains((0.6, 1.1))
        self.events.push_robot = _push(1.0, (5.0, 10.0))
        self.observations.policy.enable_corruption = True  # stock noise on all sensor terms
        self.actions.joint_pos.delay_range = (0, 2)


# axis -> (values tested, nominal value). Units: friction coefficient, kg, PD gain scale,
# push speed m/s, multiple of stock IMU noise, policy steps (20 ms).
EVAL_AXES = {
    # Effective contact friction (ground and feet both set). The ideal simulator's is 1.0: its
    # ground is 1.0 and Newton takes the larger of the two surfaces' values.
    "friction": ([0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.25], 1.0),
    "payload_kg": ([0.0, 2.0, 4.0, 6.0, 8.0, 10.0], 0.0),
    "motor_gain": ([1.0, 0.8, 0.6, 0.5, 0.4, 0.3], 1.0),
    "push_mps": ([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], 0.0),
    "imu_noise_x": ([0.0, 1.0, 2.0, 4.0, 8.0], 0.0),
    "delay_steps": ([0, 1, 2, 3, 4], 0),
}


def _make_ideal(cfg) -> None:
    """Strip every randomization so the config matches the ideal simulator."""
    _set_ground_friction(cfg, 1.0)
    cfg.events.physics_material = _friction((NOMINAL_STATIC_FRICTION,) * 2, (NOMINAL_DYNAMIC_FRICTION,) * 2)
    cfg.events.add_base_mass = None
    cfg.events.base_com = None
    cfg.events.motor_gains = None
    cfg.events.push_robot = None
    cfg.actions.joint_pos.delay_range = (0, 0)
    _set_imu_noise(cfg, 0.0)


def apply_eval_change(cfg, axis: str, value: float):
    """Turn a resolved task config into a test world: the ideal simulator with one property set to ``value``."""
    _make_ideal(cfg)
    for term in vars(cfg.commands).values():
        if hasattr(term, "debug_vis"):
            term.debug_vis = False
    if axis == "friction":
        # Newton (MuJoCo) uses the larger of the two surfaces' friction, so the ground must change too
        _set_ground_friction(cfg, value)
        cfg.events.physics_material = _friction((value, value), (0.75 * value, 0.75 * value))
    elif axis == "payload_kg":
        if value:
            cfg.events.add_base_mass = _payload((value, value))
    elif axis == "motor_gain":
        if value != 1.0:
            cfg.events.motor_gains = _motor_gains((value, value))
    elif axis == "push_mps":
        if value:
            cfg.events.push_robot = _push(value, (4.0, 4.0))
    elif axis == "imu_noise_x":
        _set_imu_noise(cfg, value)
    elif axis == "delay_steps":
        cfg.actions.joint_pos.delay_range = (int(value), int(value))
    else:
        raise ValueError(f"unknown axis {axis!r}; choose from {list(EVAL_AXES)}")
    return cfg
