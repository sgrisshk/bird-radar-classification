# Configuration Layout

## Active Default
- `bird_radar/redesign_config.yaml` is kept as the default config used by:
  - `bird_radar/run_redesign.py` (when `--config` is omitted).

## Experiment Config Archive
- Historical redesign configs are stored in:
  - `bird_radar/configs/redesign/`

`run_redesign.py` supports backward compatibility:
- if an old config path is passed and not found at its original location,
- it will automatically try `bird_radar/configs/redesign/<basename>.yaml`.

