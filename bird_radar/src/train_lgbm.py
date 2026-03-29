from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def run_lgbm_experiment(
    project_root: str | Path,
    data_dir: str | Path,
    cache_dir: str | Path,
    artifact_dir: str | Path,
    seed: int,
    hyperparams: dict[str, Any],
    python_executable: str | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifact_dir / "train.log"
    py = python_executable or sys.executable

    cmd = [
        py,
        str(project_root / "train_lgbm.py"),
        "--data-dir",
        str(data_dir),
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(artifact_dir),
        "--seed",
        str(int(seed)),
        "--learning-rate",
        str(float(hyperparams.get("learning_rate", 0.02))),
        "--num-leaves",
        str(int(hyperparams.get("num_leaves", 31))),
        "--feature-fraction",
        str(float(hyperparams.get("feature_fraction", 0.8))),
        "--n-estimators",
        str(int(hyperparams.get("n_estimators", 8000))),
    ]
    started = time.time()
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write("CMD: " + " ".join(cmd) + "\n\n")
        logf.flush()
        proc = subprocess.run(
            cmd,
            cwd=project_root,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
            check=False,
        )
    duration = time.time() - started

    result = {
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": int(proc.returncode),
        "duration_sec": float(duration),
        "artifact_dir": str(artifact_dir),
        "log_path": str(log_path),
        "oof_path": str(artifact_dir / "oof_calibrated.npy"),
        "test_path": str(artifact_dir / "test_calibrated.npy"),
        "scores_path": str(artifact_dir / "scores.json"),
        "model_type": "lgbm",
    }
    scores_path = artifact_dir / "scores.json"
    if scores_path.exists():
        try:
            with scores_path.open("r", encoding="utf-8") as f:
                result["scores"] = json.load(f)
        except Exception:
            pass
    return result

