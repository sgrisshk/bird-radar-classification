from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import CLASSES
from src.redesign.utils import dump_json


def export_submissions(cfg: dict[str, Any], blend_report: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(cfg["paths"]["output_dir"]).resolve()
    blend_dir = out_dir / "blends"
    sub_dir = out_dir / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)

    test_ids = np.load(out_dir / "test_track_ids.npy").astype(np.int64)

    artifacts: list[dict[str, Any]] = []
    for item in blend_report["top_blends"]:
        pred = np.load(item["test_path"]).astype(np.float32)
        pred = np.clip(pred, 0.0, 1.0)

        sub = pd.DataFrame(pred, columns=CLASSES)
        sub.insert(0, "track_id", test_ids)

        sub_path = sub_dir / f"{item['name']}.csv"
        sub.to_csv(sub_path, index=False)

        val_report_path = sub_dir / f"{item['name']}_validation.txt"
        cmd = [
            ".venv/bin/python",
            "validate_submission.py",
            "--test-csv",
            str((Path(cfg["paths"]["data_dir"]).resolve() / "test.csv")),
            "--submission-csv",
            str(sub_path),
        ]
        proc = subprocess.run(cmd, cwd=Path(cfg["paths"]["project_root"]).resolve().parent, capture_output=True, text=True)
        val_report_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")

        artifacts.append(
            {
                "name": item["name"],
                "submission_csv": str(sub_path.resolve()),
                "validation_report": str(val_report_path.resolve()),
                "macro_map_oof": float(item["macro_map"]),
            }
        )

    report = {
        "submissions": artifacts,
        "blend_report_path": str((blend_dir / "blend_report.json").resolve()),
    }
    dump_json(sub_dir / "submission_report.json", report)
    return report
