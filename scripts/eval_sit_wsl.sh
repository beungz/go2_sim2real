#!/usr/bin/env bash
# Sweep a stand/sit policy over every fake-real world in WSL.
# EXP picks the experiment (default go2_sit; go2_sit_push for the push-hardened one) and POLICY the
# policy label in the results; the latest run is used unless SIT points at a model file or run folder.
# Results go to results/<POLICY>_results.jsonl. Extra args are passed to eval_gap.py (e.g. --axes friction).
here="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="$HOME/.isaaclab_cache"
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"
cd "$HOME/IsaacLab"
EXP="${EXP:-go2_sit}"
POLICY="${POLICY:-sit}"
SIT="${SIT:-$(ls -td logs/rsl_rl/$EXP/*/ | head -1)}"
[ -d "$SIT" ] && SIT="$(ls "$SIT"/model_*.pt | sort -V | tail -1)"
echo "$POLICY: $SIT"
mkdir -p "$here/results"
exec .venv/bin/python "$here/scripts/eval_gap.py" --task sit --sweep --out "$here/results/${POLICY}_results.jsonl" \
  --policy "$POLICY=$SIT" "$@"
