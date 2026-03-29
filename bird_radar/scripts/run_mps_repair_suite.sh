#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/sgrisshk/Desktop/AI-task"
PY="$ROOT/.venv/bin/python"
FORCE_RETRAIN="${FORCE_RETRAIN:-1}"

configs=(
  "bird_radar/redesign_config_mps_repair.yaml"
  "bird_radar/redesign_config_mps_repair_tab_only.yaml"
  "bird_radar/redesign_config_mps_repair_seq_only.yaml"
)

cd "$ROOT"

for cfg in "${configs[@]}"; do
  OUT_DIR=$("$PY" - <<PY
import yaml
cfg=yaml.safe_load(open("$cfg"))
print(cfg["paths"]["output_dir"])
PY
)
  if [ "$FORCE_RETRAIN" = "1" ] && [ -d "$OUT_DIR/artifacts" ]; then
    rm -f "$OUT_DIR"/artifacts/*_oof.npy "$OUT_DIR"/artifacts/*_test.npy "$OUT_DIR"/oof_summary.json
  fi
  echo "=== TRAIN: $cfg ==="
  "$PY" bird_radar/run_redesign.py --config "$cfg" --mode train
  "$PY" - <<PY
import json, yaml
cfg = yaml.safe_load(open("$cfg"))
out = cfg["paths"]["output_dir"]
s = json.load(open(f"{out}/oof_summary.json"))
print('output_dir:', out)
for name, meta in s.get('models', {}).items():
    print(name, 'macro_map=', round(float(meta.get('macro_map', 0.0)), 6))
print('device_used:', s.get('device'))
print()
PY
done

cat <<'TXT'
=== NEXT ===
1) If full > tab_only and full > seq_only: keep sequence branch and tune aug/domain gradually.
2) If tab_only >= full: sequence branch currently hurts; keep tabular baseline path for submissions.
3) If seq_only is competitive: prioritize sequence feature/arch tuning.
TXT
