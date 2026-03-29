#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/sgrisshk/Desktop/AI-task"
PY="$ROOT/.venv/bin/python"
CFG="bird_radar/redesign_config_mps_seq_onefold_r1b_multiseed.yaml"

cd "$ROOT"

echo "=== TRAIN: $CFG ==="
"$PY" bird_radar/run_redesign.py --config "$CFG" --mode train

out_dir=$($PY - <<PY
import yaml
c=yaml.safe_load(open("$CFG"))
print(c["paths"]["output_dir"])
PY
)

echo "=== SUMMARY: $out_dir/oof_summary.json ==="
"$PY" - <<PY
import json, math
import numpy as np
from pathlib import Path

p = Path("$out_dir") / "oof_summary.json"
s = json.load(open(p))
vals=[]
for name, meta in sorted(s.get("models", {}).items()):
    if not name.startswith("deep_"):
        continue
    val=float(meta.get("macro_map", 0.0))
    tr_list=meta.get("fold_train_scores", [])
    tr=float(tr_list[0]) if tr_list else float("nan")
    gap=tr-val if tr==tr else float("nan")
    vals.append(val)
    print(f"{name}: train={tr:.6f} val={val:.6f} gap={gap:.6f}" if tr==tr else f"{name}: train=nan val={val:.6f} gap=nan")
    diag=(meta.get("fold_val_pred_diag") or [{}])[0]
    if diag:
        print("  pred_diag="
              f"mean={diag.get('pred_mean', float('nan')):.6f}, "
              f"std={diag.get('pred_std', float('nan')):.6f}, "
              f"p95={diag.get('pred_p95', float('nan')):.6f}, "
              f"gt0.2={diag.get('pred_frac_gt_0p2', float('nan')):.6f}, "
              f"gt0.5={diag.get('pred_frac_gt_0p5', float('nan')):.6f}")
    per=meta.get('per_class_ap', {})
    if per:
      items=sorted(per.items(), key=lambda kv: kv[1])
      print("  worst10=" + ", ".join(f"{k}:{v:.4f}" for k,v in items[:10]))
if vals:
    print(f"mean_val={np.mean(vals):.6f}")
    print(f"std_val={np.std(vals):.6f}")
PY
