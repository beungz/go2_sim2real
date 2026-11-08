#!/usr/bin/env bash
# Short Go2 training run inside WSL to check the install and asset downloads.
# Assets are cached under TMPDIR, so keep it on disk (WSL clears /tmp on shutdown).
export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="$HOME/.isaaclab_cache"
mkdir -p "$TMPDIR"
cd "$HOME/IsaacLab"
uv run isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-UnitreeGo2 \
  --num_envs 256 --max_iterations 3 physics=newton_mjwarp --viz none > "$HOME/smoke.log" 2>&1
echo "exit=$?"
grep -E "Steps per second|Training time" "$HOME/smoke.log"
find "$TMPDIR" -type f | grep -v isaaclab_usd_ | head -50
du -sh "$TMPDIR"
