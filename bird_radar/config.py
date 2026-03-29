from __future__ import annotations

import os
from pathlib import Path

CLASSES = [
    "Clutter",
    "Cormorants",
    "Pigeons",
    "Ducks",
    "Geese",
    "Gulls",
    "Birds of Prey",
    "Waders",
    "Songbirds",
]

CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

SEQUENCE_FEATURES = [
    "x",
    "y",
    "altitude",
    "rcs",
    "speed",
    "vertical_speed",
    "acceleration",
    "curvature",
    "dt",
]

RADAR_BIRD_SIZE_CATEGORIES = ["Small bird", "Medium bird", "Large bird", "Flock"]
RADAR_BIRD_SIZE_TO_RCS = {
    "Small bird": 0.0,
    "Medium bird": 0.5,
    "Large bird": 1.0,
    "Flock": 1.5,
}

DEFAULT_DATA_DIR = Path(os.getenv("BIRD_RADAR_DATA_DIR", "/Users/sgrisshk/Desktop/ai-cup-2026-performance"))
TRAIN_CSV = DEFAULT_DATA_DIR / "train.csv"
TEST_CSV = DEFAULT_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = DEFAULT_DATA_DIR / "sample_submission.csv"

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CACHE_DIR = ARTIFACTS_DIR / "cache"
SEQUENCE_DIR = ARTIFACTS_DIR / "sequence"
LGBM_DIR = ARTIFACTS_DIR / "lgbm"
ENSEMBLE_DIR = ARTIFACTS_DIR / "ensemble"

N_FOLDS = 5
GROUP_COLUMN_CANDIDATES = ["primary_observation_id", "observation_id"]

MAX_LEN = 512
BATCH_SIZE = 16
SEQUENCE_EPOCHS = 50
SEQUENCE_PATIENCE = 10
SEQUENCE_LR = 1e-4
SEQUENCE_WEIGHT_DECAY = 1e-4
SEQUENCE_LABEL_SMOOTHING = 0.05
SEQUENCE_DROPOUT = 0.3
SEQUENCE_SEEDS = [2026, 3407, 777]

LGBM_SEED = 2026


def ensure_dirs() -> None:
    for path in [ARTIFACTS_DIR, CACHE_DIR, SEQUENCE_DIR, LGBM_DIR, ENSEMBLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)

