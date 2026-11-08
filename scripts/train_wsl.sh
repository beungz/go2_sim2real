#!/usr/bin/env bash
# Train a Go2 policy in WSL. TASK picks the task (default: stock flat walking); the study's own
# tasks (Go2-Flat-NoRand, Go2-Flat-Rand) are registered from go2_sim2real.
# Extra args are passed to `isaaclab train` (e.g. --max_iterations 500 --num_envs 8192).
# Logs land in ~/IsaacLab/logs/rsl_rl/<experiment>.
here="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="$HOME/.isaaclab_cache"
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"
cd "$HOME/IsaacLab"
exec uv run isaaclab train --rl_library rsl_rl --task "${TASK:-Isaac-Velocity-Flat-UnitreeGo2}" \
  --external_callback go2_sim2real.register physics=newton_mjwarp --viz none "$@"
