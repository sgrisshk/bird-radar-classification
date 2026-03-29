from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _is_oom(log_text: str) -> bool:
    s = log_text.lower()
    markers = [
        "cuda out of memory",
        "outofmemoryerror",
        "cublas_status_alloc_failed",
        "mps backend out of memory",
        "out of memory",
    ]
    return any(m in s for m in markers)


def run_seq_experiment(
    project_root: str | Path,
    data_dir: str | Path,
    cache_dir: str | Path,
    artifact_dir: str | Path,
    seed: int,
    device: str,
    hyperparams: dict[str, Any],
    max_len: int,
    batch_size: int,
    python_executable: str | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    py = python_executable or sys.executable

    bs_ladder = [int(batch_size), 12, 8, 6]
    bs_ladder = [x for i, x in enumerate(bs_ladder) if x > 0 and x not in bs_ladder[:i]]
    ml_ladder = [int(max_len), 384, 256]
    ml_ladder = [x for i, x in enumerate(ml_ladder) if x > 0 and x not in ml_ladder[:i]]
    attempts: list[tuple[int, int]] = []
    for bs in bs_ladder:
        for ml in ml_ladder:
            attempts.append((bs, ml))
    attempts = attempts[:3] if str(device).lower().startswith("cuda") else [(int(batch_size), int(max_len))]

    last_log_text = ""
    started_all = time.time()
    for attempt_idx, (bs, ml) in enumerate(attempts, start=1):
        log_path = artifact_dir / f"train_attempt_{attempt_idx}.log"
        cmd = [
            py,
            str(project_root / "train_sequence.py"),
            "--data-dir",
            str(data_dir),
            "--cache-dir",
            str(cache_dir),
            "--output-dir",
            str(artifact_dir),
            "--device",
            str(device),
            "--epochs",
            str(int(hyperparams.get("epochs", 12))),
            "--seeds",
            str(int(seed)),
            "--batch-size",
            str(int(bs)),
            "--max-len",
            str(int(ml)),
            "--dropout",
            str(float(hyperparams.get("dropout", 0.3))),
            "--weight-decay",
            str(float(hyperparams.get("weight_decay", 1e-4))),
            "--lr",
            str(float(hyperparams.get("lr", 1e-4))),
        ]
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
        try:
            last_log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            last_log_text = ""
        if proc.returncode == 0:
            duration = time.time() - started_all
            result = {
                "status": "ok",
                "returncode": 0,
                "duration_sec": float(duration),
                "artifact_dir": str(artifact_dir),
                "log_path": str(log_path),
                "attempts": attempt_idx,
                "used_batch_size": int(bs),
                "used_max_len": int(ml),
                "oof_path": str(artifact_dir / "oof_sequence_mean_calibrated.npy"),
                "test_path": str(artifact_dir / "test_sequence_mean_calibrated.npy"),
                "scores_path": str(artifact_dir / "summary.json"),
                "model_type": "seq",
            }
            scores_path = artifact_dir / "summary.json"
            if scores_path.exists():
                try:
                    with scores_path.open("r", encoding="utf-8") as f:
                        result["scores"] = json.load(f)
                except Exception:
                    pass
            return result
        if not (str(device).lower().startswith("cuda") and _is_oom(last_log_text)):
            break

    duration = time.time() - started_all
    fail_log = str(artifact_dir / f"train_attempt_{min(len(attempts), max(1, len(attempts)))}.log")
    return {
        "status": "failed",
        "returncode": int(proc.returncode) if "proc" in locals() else -1,
        "duration_sec": float(duration),
        "artifact_dir": str(artifact_dir),
        "log_path": fail_log,
        "attempts": len(attempts),
        "oom_detected": bool(_is_oom(last_log_text)),
        "model_type": "seq",
    }

