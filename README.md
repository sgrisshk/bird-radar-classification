# Bird Radar Classification Project

## Overview
This repository contains a full experimentation workflow for multiclass bird-track classification from radar trajectories.

The project includes:
- tabular feature extraction from trajectories,
- one-vs-rest gradient boosting pipelines,
- sequence-model experiments,
- ensemble and submission tooling,
- diagnostics for temporal shift and leaderboard proxy analysis.

## Public References
- Competition hub: https://www.teamepoch.ai/ai-cup-2026/
- Implementation track: https://www.teamepoch.ai/ai-cup-2026/implementation-track/
- Kaggle page: https://www.kaggle.com/competitions/ai-cup-2026-performance
- LinkedIn write-up: https://www.linkedin.com/posts/grishkova-sofya-borisovna_turbine-uncertainty-meet-ai-this-answers-activity-7444008472049963008-Ol6a

## Repository Layout
- `bird_radar/src` core feature engineering, models, and utilities.
- `bird_radar/scripts` training, evaluation, blending, and diagnostic scripts.
- `bird_radar/configs` configuration files (`redesign` experiment configs are grouped here).
- `bird_radar/data` local data helpers and intermediate data assets.
- `bird_radar/submissions` generated submission files.
- `bird_radar/artifacts` local experiment outputs (large, mostly ignored for GitHub).
- `bird_radar/archive` archived local logs, legacy scripts, and old root-level submissions.
- `docs` public links (task page, LinkedIn post, repository links).
- `docs/PROJECT_STRUCTURE.md` concise structure index.

## Environment
Use Python 3.11+ and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Placement
Expected local files at repository root:
- `train.csv`
- `test.csv`
- `sample_submission.csv`

If you do not have the original task files, see `docs/LINKS.md` and add your task URL there.

## Common Commands
Train an OvR LightGBM forward-CV run:

```bash
.venv/bin/python bird_radar/scripts/train_lgbm_ovr_forward.py \
  --train-csv train.csv \
  --test-csv test.csv \
  --sample-submission sample_submission.csv \
  --output-dir bird_radar/artifacts/proposal/local_run
```

Evaluate leaderboard proxy on OOF predictions:

```bash
.venv/bin/python bird_radar/scripts/eval_lb_proxy_month_holdout.py \
  --train-csv train.csv \
  --oof-csv bird_radar/artifacts/proposal/local_run/oof.csv \
  --output-json bird_radar/artifacts/proposal/local_run/lb_proxy.json
```

## Publication Notes
- The repository is cleaned for publishing: transient logs and root-level experiment outputs were moved under `bird_radar/archive`.
- Large local artifacts and dataset files are ignored via `.gitignore`.
- Add your public references in `docs/LINKS.md` before publishing.
- Archived internal notes are in `docs/archive/`.

## Reproducibility and Scope
This codebase is research-oriented and contains many experimental branches.  
For GitHub presentation, the recommended entrypoint is:
- `bird_radar/scripts/train_lgbm_ovr_forward.py`
- `bird_radar/scripts/eval_lb_proxy_month_holdout.py`
- `bird_radar/scripts/blend_submissions.py`

## License
See `LICENSE`.
