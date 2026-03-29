# Project Structure

This repository is organized for public presentation and reproducible local runs.

## Root
- `README.md` project overview and quickstart.
- `requirements.txt` Python dependencies.
- `docs/LINKS.md` public links (LinkedIn, challenge pages).
- `train.csv`, `test.csv`, `sample_submission.csv` local task files (not tracked in git).
- `validate_submission.py` validation helper.

## Core Package: `bird_radar/`
- `src/` feature engineering, model code, utilities.
- `scripts/` runnable experiment scripts.
- `configs/` grouped configuration files.
  - `configs/redesign/` archived redesign config variants.
- `data/` local derived parquet features.
- `submissions/` generated local submissions.
- `artifacts/` local experiment outputs.
- `archive/` archived logs and legacy helper files.

## Primary Entrypoints
- `bird_radar/scripts/train_lgbm_ovr_forward.py`
- `bird_radar/scripts/eval_lb_proxy_month_holdout.py`
- `bird_radar/scripts/blend_submissions.py`

## Notes
- `artifacts/` and `submissions/` are intentionally gitignored for clean publishing.
- Historical notes are stored in `docs/archive/`.

