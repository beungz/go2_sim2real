"""Go2 sim-to-real gap study: custom Isaac Lab tasks.

Load into the Isaac Lab CLI with ``--external_callback go2_sim2real.register`` and
``PYTHONPATH`` pointing at this repository.
"""

from __future__ import annotations

import sys

TASKS = {
    # task id: (env cfg class, agent cfg class)
    "Go2-Flat-NoRand": ("Go2FlatNoRandEnvCfg", "Go2FlatNoRandPPORunnerCfg"),
    "Go2-Flat-Rand": ("Go2FlatRandEnvCfg", "Go2FlatRandPPORunnerCfg"),
    "Go2-Sit": ("sit:Go2SitEnvCfg", "Go2SitPPORunnerCfg"),
    "Go2-Sit-Push": ("sit:Go2SitPushEnvCfg", "Go2SitPushPPORunnerCfg"),
    "Go2-Sit-Step": ("sit:Go2SitStepEnvCfg", "Go2SitStepPPORunnerCfg"),
}


def register() -> list[str]:
    """Register the study's tasks with Gym; returns the Hydra-style ``key=value`` args to keep."""
    import gymnasium as gym

    for task_id, (env_cfg, agent_cfg) in TASKS.items():
        if task_id in gym.registry:
            continue
        env_module, _, env_cfg = env_cfg.rpartition(":")
        env_module = env_module or "envs"
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"go2_sim2real.{env_module}:{env_cfg}",
                "rsl_rl_cfg_entry_point": f"go2_sim2real.agents:{agent_cfg}",
                "default_agent": "rsl_rl",
            },
        )
    return [a for a in sys.argv[1:] if "=" in a and not a.startswith("-")]
