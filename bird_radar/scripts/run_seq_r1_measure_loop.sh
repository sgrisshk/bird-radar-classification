#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/sgrisshk/Desktop/AI-task"
PY="$ROOT/.venv/bin/python"

configs=(
  "bird_radar/redesign_config_mps_seq_onefold_r1b_train.yaml"
  "bird_radar/redesign_config_mps_seq_onefold_r1_logdt.yaml"
  "bird_radar/redesign_config_mps_seq_onefold_r1_pool_meanmax.yaml"
)

cd "$ROOT"

for cfg in "${configs[@]}"; do
  out_dir=$($PY - <<PY
import yaml
c=yaml.safe_load(open("$cfg"))
print(c["paths"]["output_dir"])
PY
)

  echo "=== TRAIN: $cfg ==="
  "$PY" bird_radar/run_redesign.py --config "$cfg" --mode train

  echo "=== METRICS: $cfg ==="
  "$PY" - <<PY
import json
from pathlib import Path

p = Path("$out_dir") / "oof_summary.json"
s = json.load(open(p))
for name, meta in s.get("models", {}).items():
    if not name.startswith("deep_"):
        continue
    val = float(meta.get("macro_map", 0.0))
    tr_list = meta.get("fold_train_scores", [])
    tr = float(tr_list[0]) if tr_list else float("nan")
    gap = tr - val if tr == tr else float("nan")
    print(f"model={name}")
    print(f"  train_macro_map={tr:.6f}" if tr == tr else "  train_macro_map=nan")
    print(f"  val_macro_map={val:.6f}")
    print(f"  gap={gap:.6f}" if gap == gap else "  gap=nan")
    diag = (meta.get("fold_val_pred_diag") or [{}])[0]
    if diag:
        print(
            "  pred_diag="
            f"mean={diag.get('pred_mean', float('nan')):.6f}, "
            f"std={diag.get('pred_std', float('nan')):.6f}, "
            f"p95={diag.get('pred_p95', float('nan')):.6f}, "
            f"gt0.2={diag.get('pred_frac_gt_0p2', float('nan')):.6f}, "
            f"gt0.5={diag.get('pred_frac_gt_0p5', float('nan')):.6f}"
        )
    per = meta.get("per_class_ap", {})
    if per:
        items = sorted(per.items(), key=lambda kv: kv[1])
        worst = items[:10]
        best = items[-10:][::-1]
        print("  worst10=" + ", ".join(f"{k}:{v:.4f}" for k, v in worst))
        print("  best10=" + ", ".join(f"{k}:{v:.4f}" for k, v in best))
PY
  echo

done
