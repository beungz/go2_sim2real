#!/usr/bin/env bash
# Run bench.py inside WSL against the Linux-side Isaac Lab checkout.
export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="$HOME/.isaaclab_cache"
here="$(cd "$(dirname "$0")" && pwd)"
"$HOME/IsaacLab/.venv/bin/python" "$here/bench.py" --isaaclab "$HOME/IsaacLab" --label wsl "$@"
