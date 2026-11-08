#!/usr/bin/env bash
# Record a playback video of the latest checkpoint in WSL. TASK picks the task as in train_wsl.sh.
# Extra args are passed to `isaaclab play` (e.g. --checkpoint <path> --num_envs 4).
# Videos land in the run folder under ~/IsaacLab/logs/rsl_rl/<experiment>/<run>/videos/play.
here="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="$HOME/.isaaclab_cache"
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"
cd "$HOME/IsaacLab"
exec uv run --extra video isaaclab play --rl_library rsl_rl --task "${TASK:-Isaac-Velocity-Flat-UnitreeGo2}" \
  --external_callback go2_sim2real.register physics=newton_mjwarp \
  --checkpoint latest --num_envs 16 --video --video_length 500 "$@"
