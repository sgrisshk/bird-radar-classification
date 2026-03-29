# bird_radar Package Notes

This folder contains the project implementation.

## Key Modules
- `src/feature_engineering.py` trajectory feature extraction and engineered features.
- `src/lgbm_model.py` LightGBM OVR wrappers and training utilities.
- `scripts/train_lgbm_ovr_forward.py` primary temporal forward-CV training script.
- `scripts/eval_lb_proxy_month_holdout.py` leaderboard-proxy metrics on OOF.
- `scripts/blend_submissions.py` blend utilities for submission-level experiments.
- `configs/redesign` archived redesign configuration variants.

## Conventions
- Experiments write to `artifacts/<run_name>`.
- Final candidate files are saved in `submissions/`.
- Long-running exploratory files should be moved under `archive/` when no longer active.
- `run_redesign.py` defaults to `redesign_config.yaml`; extra redesign configs are grouped in `configs/redesign/`.
