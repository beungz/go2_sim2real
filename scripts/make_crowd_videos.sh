#!/usr/bin/env bash
# Many-robot push videos for one stand/sit policy (default: v4, may step while standing).
# Same scenario as make_progress_videos.sh, but N robots (default 16) in view at once, each pushed
# in its own random direction, with fall and off-pose counters.
# Output: results/progress/crowd_<name>/push_<speed>/{clip_0000.mp4,events.json,annotated.mp4}
set -e
here="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="$HOME/.isaaclab_cache"
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"
cd "$HOME/IsaacLab"
VERSION="${VERSION:-v4_may_step}"
MODEL="${MODEL:-$here/runs/sit_step_1500/model_1499.pt}"
N="${N:-16}"
OUT="$here/results/progress/crowd_$VERSION"
for push in ${PUSHES:-1.5 2.0}; do
  dir="$OUT/videos/push_$push"
  echo "== $VERSION, $N robots, push $push m/s: $MODEL"
  uv run --extra video python "$here/scripts/record_push_video.py" --task Go2-Sit --policy "$MODEL" \
    --out_dir "$OUT" --name "push_$push" --push "$push" --num_envs "$N" 2>&1 | grep -E "^wrote|Error" || true
  .venv/bin/python -I "$here/scripts/annotate_video.py" "$dir/clip_0000.mp4" "$dir/annotated.mp4" \
    --events "$dir/events.json" --title "$VERSION: $N robots, $push m/s pushes"
done
