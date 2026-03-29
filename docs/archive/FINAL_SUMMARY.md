# Bird Radar — Final Summary

Date: 2026-03-14

## Final Decision
- Competition work is closed.
- Final selected submission: `bird_radar/submissions/sub_blend_softmax_w80_sizeprior_b10.csv`
- Active marker in repo: `bird_radar/submissions/ACTIVE_SUBMISSION.txt`

## Ground Truth About The Data
- Train months: `1, 4, 9, 10` with counts `{1:221, 4:473, 9:467, 10:1440}`
- Test months: `2, 5, 9, 10, 12` with counts `{2:176, 5:303, 9:457, 10:803, 12:133}`
- OOD test share (months 2/5/12): `612 / 1872 = 32.7%`
- Severe class imbalance in train:
  - `Gulls=1503`, `Songbirds=483`, `Waders=120`, `Birds of Prey=108`, `Geese=83`, `Ducks=58`, `Cormorants=40`

## Baseline OOF Structure (stack9 forward complete)
Source: `bird_radar/artifacts/oof_csv_for_per_class/oof_temporal_stack9_forward_complete.csv`

- Macro AP: `0.4507`
- Per-class AP:
  - `Clutter 0.7662`
  - `Cormorants 0.1001`
  - `Pigeons 0.4831`
  - `Ducks 0.2460`
  - `Geese 0.4510`
  - `Gulls 0.8332`
  - `Birds of Prey 0.4528`
  - `Waders 0.1681`
  - `Songbirds 0.5562`

Main bottleneck classes: `Cormorants`, `Waders`, `Ducks`.

## What Was Tried (and Result)

### 1) Blending / stacking / weighting sweeps
- Multiple blend families, convex scans, teacher/TCN/Transformer mixes, classwise overrides, meta-stack variants.
- OOF often improved locally; LB did not show stable conversion above the selected 0.51 plateau.

### 2) Pairwise specialists (Waders/Ducks/BOP/Cormorants/Geese/Songbirds/Pigeons)
- Strong OOF lifts on some classes (especially Waders) in controlled experiments.
- Conversion to LB was unstable; Cormorants remained fragile and repeatedly regressed.

### 3) Postprocessing interventions (calibration/priors/LP)
- Label propagation temporal-forward looked promising in OOF:
  - `k=20, beta=0.30`: fold mean delta `+0.0152`
  - files: `bird_radar/artifacts/proposal/label_prop_temporal_forward_summary.csv`,
    `bird_radar/artifacts/proposal/label_prop_temporal_forward_report.json`
- But LB outcomes were strongly negative:
  - LP submission: ~`0.46`
  - logit eco-prior: ~`0.43`
  - gull-penalty eco-prior: ~`0.44`
- Conclusion: postprocessing consistently harmed LB despite OOF gains.

### 4) Feature engineering
- Tried: structural ACF/turning/distribution features, dt-stats, quantile profiles, segment consistency, GIS, weather, interaction blocks, month soft features, etc.
- Notable gain:
  - `w80_dt_on_orig_teacher_tcn_75_25`: covered ref `0.4529` (from `0.4464`)
  - file: `bird_radar/artifacts/proposal/w80_dt_on_orig_teacher_tcn_75_25_metrics.json`
- But improvements were not robust across LB; several additions increased domain separability and hurt generalization.

### 5) Sequence models and redesign runs
- TCN/Transformer families, SSL pretrain/fine-tune, multiple MPS runs, heavy/tiny variants, ranking/distill variants.
- Best sequence-only directions remained below the tabular stack in stable setting.

### 6) Domain adaptation / adversarial controls
- Domain reweighting, feature blacklists, month-conditional calibration, monotone constraints, etc.
- Did not produce reliable LB gains.

### 7) Data-cleaning / sample weighting for rare classes
- Manual Cormorant audit and targeted reweight test:
  - dropped `2658934`, downweighted `2628038` to `0.25`
  - result: Cormorants AP delta `-0.0139` (worse)
  - artifacts:
    - `bird_radar/artifacts/proposal/sample_weights_cormorant_audit_v1.parquet`
    - `bird_radar/artifacts/temporal_model0_reg_drop10_orig146dt_cormaudit_v1/report.json`

## Why We Stop
- OOF↔LB mismatch is structural on this dataset setup, not just tuning noise.
- 32.7% OOD test months + tiny rare classes (`Cormorants=40`, `Ducks=58`, `Waders=120`) create a Gull-attractor failure mode.
- New ideas repeatedly produced one of two patterns:
  - OOF gains with LB drop (most postprocessing/domain tricks),
  - or no meaningful gain on bottleneck classes.

## Final Status
- Keep `sub_blend_softmax_w80_sizeprior_b10.csv` as final selected submission.
- Do not spend more submissions/iterations without new labeled data or a truly new external signal with proven no-leak behavior.

## If Reopened Later
- Only high-value path: new labeled data / transfer data targeted at winter/OOD months for rare classes.
- Without new data, expected ROI of further local experimentation is very low.
