#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/sgrisshk/Desktop/AI-task"
PY="$ROOT/.venv/bin/python"
CACHE="$ROOT/bird_radar/artifacts/cache"
PSEUDO="$ROOT/bird_radar/artifacts/pseudo_labels/pseudo_teacher047_p95_top40.csv"

run_exp () {
  local name="$1"
  shift
  local out_dir="$ROOT/bird_radar/artifacts/$name"
  echo "=== RUN: $name ==="
  MPLCONFIGDIR=/tmp/.mpl "$PY" "$ROOT/bird_radar/train_temporal_lgbm.py" \
    --data-dir "$ROOT" \
    --cache-dir "$CACHE" \
    --output-dir "$out_dir" \
    --force-config reg \
    --drop-k-grid 10 \
    --blacklist-mode none \
    --seed 2026 \
    --oof-strategy forward_cv \
    --oof-forward-splits 5 \
    --oof-forward-complete backcast_last \
    "$@"

  "$PY" - <<'PY' "$out_dir"
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "report.json"
j = json.loads(p.read_text())
print("temporal_macro_map_best=", j.get("temporal_macro_map_best"))
print("adversarial_auc_mean=", j.get("adversarial_auc_mean"))
print("domain_reweight=", j.get("domain_reweight"))
print("pseudo_label_info=", j.get("pseudo_label_info"))
PY
}

if [[ ! -f "$PSEUDO" ]]; then
  echo "Missing pseudo labels: $PSEUDO"
  exit 1
fi

run_exp "temporal_reweight_seed2026" \
  --domain-reweight odds \
  --domain-reweight-clip-min 0.5 \
  --domain-reweight-clip-max 3.0

run_exp "temporal_pseudo_seed2026" \
  --domain-reweight none \
  --pseudo-label-csv "$PSEUDO" \
  --pseudo-weight 0.2

run_exp "temporal_pseudo_reweight_seed2026" \
  --domain-reweight odds \
  --domain-reweight-clip-min 0.5 \
  --domain-reweight-clip-max 3.0 \
  --pseudo-label-csv "$PSEUDO" \
  --pseudo-weight 0.2

echo "=== DONE: shift adapt suite ==="
