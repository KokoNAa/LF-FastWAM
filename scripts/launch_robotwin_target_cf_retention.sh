#!/usr/bin/env bash
# Data collection only. Does not start instances, train, or consume test scenes.
set -Eeuo pipefail
if [[ $# != 1 ]]; then
  echo "Usage: bash scripts/launch_robotwin_target_cf_retention.sh NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH="$repo_dir/src:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH=/root/gpufree-data/fastwam/FastWAM/checkpoints
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export OMP_NUM_THREADS=2
phase_output=$1
base_manifest=/root/gpufree-data/LF-FastWAM/runs/robotwin_eraf_fg/20260906-warm400-v1/bank_full_v2/manifest.json
teacher=/root/gpufree-data/LF-FastWAM/runs/robotwin_cf_dense/20260906-native-retention-v1/repair-shared-decisions/step_000400.pt
robotwin_root=/root/gpufree-data/LF-FastWAM/third_party/RoboTwin
test -f "$base_manifest"
test -f "$teacher"
test -d "$robotwin_root/assets"
command -v timeout >/dev/null
# Preserve the original campaign cutoff, even when this command starts late.
collection_budget=$(python - <<'PY'
from datetime import datetime, timezone
deadline = datetime.fromisoformat('2026-09-07T11:30:00+08:00')
remaining = int((deadline - datetime.now(timezone.utc)).total_seconds()) - 45
if remaining < 900:
    raise SystemExit('Insufficient time before the original 11:30 HKT work cutoff.')
print(min(4200, remaining))
PY
)
gpu_inventory=$(nvidia-smi --query-gpu=index --format=csv,noheader)
if [[ $(wc -l <<<"$gpu_inventory" | tr -d ' ') != 6 ]]; then
  echo 'This collection requires the authorized six-GPU instance.' >&2
  exit 1
fi
compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$compute_pids" ]]; then
  echo 'GPU compute jobs are already running; no collectors were started.' >&2
  exit 1
fi
mkdir "$phase_output"
pids=()
cleanup() {
  trap - INT TERM
  # These PIDs are the timeout supervisors created by this script. Each owns
  # its child's process group; no global process matching or GPU reset.
  for pid in $(jobs -pr); do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  exit 130
}
trap cleanup INT TERM
for gpu in 0 1 2 3 4 5; do
  if (( gpu < 3 )); then
    task=place_a2b_left
    seed=$((80800000 + gpu * 1000))
  else
    task=blocks_ranking_rgb
    seed=$((80810000 + (gpu - 3) * 1000))
  fi
  timeout --signal=TERM --kill-after=30s "${collection_budget}s" \
    python -u scripts/collect_robotwin_cf_retention.py \
      --manifest "$base_manifest" --checkpoint "$teacher" --robotwin-root "$robotwin_root" \
      --task "$task" --condition counterfactual --start-seed "$seed" \
      --scenes 4 --max-attempts 40 --max-seconds "$((collection_budget - 60))" \
      --gpu "$gpu" --output "$phase_output/shard$gpu" >"$phase_output/shard$gpu.log" 2>&1 &
  pids+=("$!")
done
failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[index]}"; then code=0; else code=$?; failed=1; fi
  printf '%s\n' "$code" >"$phase_output/shard$index.exit"
done
trap - INT TERM
python - "$phase_output" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
report = {'purpose': 'training data collection, not an evaluation result',
          'complete': True, 'hash_scans': False, 'optimizer_updates': 0, 'shards': []}
seen = set()
counts = {}
for shard in range(6):
    path = root / f'shard{shard}' / 'manifest.json'
    data = json.loads(path.read_text()) if path.exists() else {}
    code = int((root / f'shard{shard}.exit').read_text())
    task = 'place_a2b_left' if shard < 3 else 'blocks_ranking_rgb'
    keys = {(r['source_task'], r['task_config'], r['scene_seed']) for r in data.get('states', [])}
    good = (code == 0 and data.get('complete') is True and data.get('source_task') == task
            and data.get('condition') == 'counterfactual' and data.get('successful_scenes') == 4
            and len(keys) == 4 and not keys & seen and all(k[0] == task for k in keys))
    report['complete'] &= good
    report['shards'].append({'shard': shard, 'exit_code': code, 'manifest': str(path),
                             'successful_scenes': len(keys), 'complete': good})
    seen.update(keys)
    counts[task] = counts.get(task, 0) + len(keys)
report['successful_training_scenes'] = counts
(root / 'collection_result.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
if not report['complete']:
    raise SystemExit('Incomplete collection; preserve partial results and inspect shard logs.')
PY
exit "$failed"
