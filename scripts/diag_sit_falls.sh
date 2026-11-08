#!/usr/bin/env bash
# When do stand/sit policies fall under pushes? Runs each push level under three command scripts
# (stand/sit/stand sequence, always stand, always sit) and appends per-phase fall shares to
# results/<POLICY>_fall_diag.jsonl. EXP/POLICY pick the policy as in eval_sit_wsl.sh (or MODEL=<absolute path to a .pt>); PUSHES the levels.
here="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="$HOME/.isaaclab_cache"
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"
cd "$HOME/IsaacLab"
EXP="${EXP:-go2_sit_push}"
POLICY="${POLICY:-sit_push}"
M="${MODEL:-$(ls "$(ls -td logs/rsl_rl/$EXP/*/ | head -1)"/model_*.pt | sort -V | tail -1)}"
echo "$POLICY: $M"
for push in ${PUSHES:-1.5 2.0}; do
  for script in sequence stand sit; do
    .venv/bin/python "$here/scripts/eval_gap.py" --task sit --script $script --axis push_mps --value $push \
      --policy "$POLICY=$M" | grep '^RESULT ' | cut -c8- | tee -a "$here/results/${POLICY}_fall_diag.jsonl"
  done
done
