from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class OvernightState:
    run_id: str
    output_dir: str
    started_at_unix: float
    completed_experiments: dict[str, dict[str, Any]] = field(default_factory=dict)
    failed_experiments: dict[str, dict[str, Any]] = field(default_factory=dict)
    best_scores: dict[str, dict[str, Any]] = field(default_factory=dict)
    time_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    submissions_exported: int = 0
    last_successful_step: str = ""
    experiment_order: list[str] = field(default_factory=list)
    blend_history: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def create(run_id: str, output_dir: str, experiment_order: list[str]) -> "OvernightState":
        return OvernightState(
            run_id=run_id,
            output_dir=output_dir,
            started_at_unix=time.time(),
            experiment_order=list(experiment_order),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "output_dir": self.output_dir,
            "started_at_unix": self.started_at_unix,
            "completed_experiments": self.completed_experiments,
            "failed_experiments": self.failed_experiments,
            "best_scores": self.best_scores,
            "time_stats": self.time_stats,
            "submissions_exported": self.submissions_exported,
            "last_successful_step": self.last_successful_step,
            "experiment_order": self.experiment_order,
            "blend_history": self.blend_history,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "OvernightState":
        return OvernightState(
            run_id=str(payload.get("run_id", "")),
            output_dir=str(payload.get("output_dir", "")),
            started_at_unix=float(payload.get("started_at_unix", time.time())),
            completed_experiments=dict(payload.get("completed_experiments", {})),
            failed_experiments=dict(payload.get("failed_experiments", {})),
            best_scores=dict(payload.get("best_scores", {})),
            time_stats=dict(payload.get("time_stats", {})),
            submissions_exported=int(payload.get("submissions_exported", 0)),
            last_successful_step=str(payload.get("last_successful_step", "")),
            experiment_order=list(payload.get("experiment_order", [])),
            blend_history=list(payload.get("blend_history", [])),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=True)

    @staticmethod
    def load(path: str | Path) -> "OvernightState":
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return OvernightState.from_dict(payload)

    def is_done(self, experiment_name: str) -> bool:
        return experiment_name in self.completed_experiments or experiment_name in self.failed_experiments

    def record_completed(self, experiment_name: str, payload: dict[str, Any]) -> None:
        self.completed_experiments[experiment_name] = payload
        self.last_successful_step = f"experiment:{experiment_name}"

    def record_failed(self, experiment_name: str, payload: dict[str, Any]) -> None:
        self.failed_experiments[experiment_name] = payload

    def update_time_stats(self, model_type: str, duration_sec: float) -> None:
        ts = self.time_stats.setdefault(model_type, {"count": 0, "total_sec": 0.0, "avg_sec": None, "recent_sec": []})
        ts["count"] = int(ts.get("count", 0)) + 1
        ts["total_sec"] = float(ts.get("total_sec", 0.0)) + float(duration_sec)
        ts["avg_sec"] = float(ts["total_sec"]) / max(int(ts["count"]), 1)
        recent = list(ts.get("recent_sec", []))
        recent.append(float(duration_sec))
        ts["recent_sec"] = recent[-10:]

    def estimate_duration_sec(self, model_type: str, default_sec: float) -> float:
        ts = self.time_stats.get(model_type)
        if not ts:
            return float(default_sec)
        recent = list(ts.get("recent_sec", []))
        if recent:
            return float(sum(recent) / len(recent))
        avg = ts.get("avg_sec")
        if avg is not None:
            return float(avg)
        return float(default_sec)

