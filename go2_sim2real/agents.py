"""PPO settings: the stock Go2 flat config, with separate log folders per variant."""

from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2FlatPPORunnerCfg


@configclass
class Go2FlatNoRandPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    """PPO settings for walking trained in the ideal world (no randomization)."""

    def __post_init__(self):
        """Set the iteration budget and log folder."""
        super().__post_init__()
        self.max_iterations = 500
        self.experiment_name = "go2_flat_norand"


@configclass
class Go2SitPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    """PPO settings for the stand/sit task trained with gentle 0.5 m/s pushes."""

    def __post_init__(self):
        """Set the iteration budget and log folder."""
        super().__post_init__()
        self.max_iterations = 1000
        self.experiment_name = "go2_sit"


@configclass
class Go2SitPushPPORunnerCfg(Go2SitPPORunnerCfg):
    """PPO settings for stand/sit trained with random 0-1.5 m/s pushes."""

    def __post_init__(self):
        """Raise the iteration budget and set the log folder."""
        super().__post_init__()
        self.max_iterations = 1500
        self.experiment_name = "go2_sit_push"


@configclass
class Go2SitStepPPORunnerCfg(Go2SitPushPPORunnerCfg):
    """PPO settings for push-hardened stand/sit that may step while standing."""

    def __post_init__(self):
        """Set the log folder; the iteration budget is inherited."""
        super().__post_init__()
        self.experiment_name = "go2_sit_step"


@configclass
class Go2FlatRandPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    """PPO settings for walking trained with domain randomization."""

    def __post_init__(self):
        """Set the default iteration budget and log folder."""
        super().__post_init__()
        self.max_iterations = 500
        self.experiment_name = "go2_flat_rand"
