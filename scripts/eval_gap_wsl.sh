#!/usr/bin/env bash
# Sweep both gap-study policies over every fake-real world in WSL.
# Uses the latest run of each experiment unless NORAND / RAND point at a model file.
# Extra args are passed to eval_gap.py (e.g. --axes friction,payload_kg --num_envs 2048).
here="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="$HOME/.isaaclab_cache"
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"
cd "$HOME/IsaacLab"
NORAND="${NORAND:-$(ls -td logs/rsl_rl/go2_flat_norand/*/ | head -1)}"
RAND="${RAND:-$(ls -td logs/rsl_rl/go2_flat_rand/*/ | head -1)}"
[ -d "$NORAND" ] && NORAND="$(ls "$NORAND"/model_*.pt | sort -V | tail -1)"
[ -d "$RAND" ] && RAND="$(ls "$RAND"/model_*.pt | sort -V | tail -1)"
echo "norand: $NORAND"
echo "rand:   $RAND"
mkdir -p "$here/results"
exec .venv/bin/python "$here/scripts/eval_gap.py" --sweep --out "$here/results/gap_results.jsonl" \
  --policy "norand=$NORAND" --policy "rand=$RAND" "$@"
