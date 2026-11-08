#!/usr/bin/env bash
# Progress videos for the stand/sit task: every version replayed in the same scenario (ideal world,
# PUSH m/s pushes every 4 s, stand 0-6 s / sit 6-14 s / stand 14-20 s, same seed, following camera),
# annotated with the command timeline and falls, plus a 2x2 comparison.
# Output: results/progress/push_<PUSH>/videos/<version>/{clip_0000.mp4,events.json,annotated.mp4} and
#         results/progress/push_<PUSH>/sit_progress_2x2.mp4 (PUSH defaults to 1.5 m/s)
set -e
here="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export TMPDIR="$HOME/.isaaclab_cache"
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"
cd "$HOME/IsaacLab"
PUSH="${PUSH:-1.5}"
OUT="$here/results/progress/push_$PUSH"
FF="$(ls .venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg* | head -1)"

# version | checkpoint | caption
VERSIONS=(
  "v1_pose_reward_bug|$here/runs/sit_1000/model_999.pt|v1: pose reward too strict, learned tilt only (1000 it)"
  "v2_reward_fixed|$here/runs/sit_v2_1000/model_999.pt|v2: pose reward fixed, 0.5 m/s training pushes (1000 it)"
  "v3_push_hardened|$here/runs/sit_push_1500/model_1499.pt|v3: trained with 0-1.5 m/s pushes (1500 it)"
  "v4_may_step|$here/runs/sit_step_1500/model_1499.pt|v4: may step to recover while standing (1500 it)"
)

clips=()
for entry in "${VERSIONS[@]}"; do
  IFS="|" read -r name glob title <<< "$entry"
  model=$(ls $glob | tail -1)
  dir="$OUT/videos/$name"
  echo "== $name: $model"
  uv run --extra video python "$here/scripts/record_push_video.py" --task Go2-Sit --policy "$model" \
    --out_dir "$OUT" --name "$name" --push "$PUSH" 2>&1 | grep -E "fell|wrote|Error" || true
  .venv/bin/python -I "$here/scripts/annotate_video.py" "$dir/clip_0000.mp4" "$dir/annotated.mp4" \
    --events "$dir/events.json" --title "$title"
  clips+=("$dir/annotated.mp4")
done

"$FF" -loglevel error -y -i "${clips[0]}" -i "${clips[1]}" -i "${clips[2]}" -i "${clips[3]}" -filter_complex \
  "[0:v]scale=960:540[a];[1:v]scale=960:540[b];[2:v]scale=960:540[c];[3:v]scale=960:540[d];\
[a][b][c][d]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" \
  -map "[v]" -c:v libx264 -pix_fmt yuv420p -crf 20 "$OUT/sit_progress_2x2.mp4"
echo "wrote $OUT/sit_progress_2x2.mp4"
