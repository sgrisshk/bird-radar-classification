from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_validation(
    test_csv: str | Path,
    submission_csv: str | Path,
    report_path: str | Path,
    train_csv: str | Path | None = None,
    oof_predictions_csv: str | Path | None = None,
) -> dict[str, object]:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "validate_submission.py",
        here.parents[1] / "validate_submission.py",
    ]
    script = next((p for p in candidates if p.exists()), None)
    if script is None:
        text = "validate_submission.py not found\n"
        report_path.write_text(text, encoding="utf-8")
        return {"status": "failed", "returncode": 127, "report_path": str(report_path)}

    cmd = [sys.executable, str(script), "--test-csv", str(test_csv), "--submission-csv", str(submission_csv)]
    if train_csv is not None:
        cmd.extend(["--train-csv", str(train_csv)])
    if oof_predictions_csv is not None:
        cmd.extend(["--oof-predictions-csv", str(oof_predictions_csv)])

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    report_path.write_text(proc.stdout or "", encoding="utf-8")
    return {"status": "ok" if proc.returncode == 0 else "failed", "returncode": int(proc.returncode), "report_path": str(report_path)}

