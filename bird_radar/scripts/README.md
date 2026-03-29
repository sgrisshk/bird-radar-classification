# Scripts Guide

## Core Entrypoints (recommended)
- `train_lgbm_ovr_forward.py` main tabular OvR training with forward CV.
- `eval_lb_proxy_month_holdout.py` OOF-based leaderboard proxy.
- `blend_submissions.py` submission-level blending utilities.
- `train_xgb_ovr_forward.py` XGBoost OvR baseline/alternative.

## Auxiliary Training Pipelines
- `train_catboost_*` CatBoost variants.
- `train_tcn_trajectory_forward.py` trajectory TCN sequence model.
- `train_lgbm_ovr_rank_forward.py` ranking-objective OvR variant.

## Diagnostics / Utilities
- `analyze_*`, `d1_d2_diagnostics.py`, `optimize_blend_penalties.py`
- `build_*` scripts for masks, banks, pseudo-label artifacts, and weights.

## Note
This directory includes many experimental scripts from iterative research cycles.  
For reproducible public runs, start from the core entrypoints listed above.

