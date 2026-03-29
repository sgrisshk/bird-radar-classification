from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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


def average_precision_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int8)
    y_score = np.asarray(y_score).astype(np.float64)
    pos = int(y_true.sum())
    if pos <= 0:
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    ap = float(np.sum(precision[y_sorted == 1]) / pos)
    return ap


def per_class_ap(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str] | None = None) -> dict[str, float]:
    class_names = class_names or CLASSES
    out: dict[str, float] = {}
    for i, c in enumerate(class_names):
        out[c] = average_precision_binary(y_true[:, i], y_pred[:, i])
    return out


def macro_map(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str] | None = None) -> float:
    pcs = per_class_ap(y_true, y_pred, class_names=class_names)
    return float(np.mean(list(pcs.values()))) if pcs else 0.0


class Scoreboard:
    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.columns = [
            "timestamp",
            "experiment_name",
            "model_type",
            "seed",
            "params_json",
            "macro_map",
            "per_class_ap_json",
            "duration_sec",
            "status",
            "artifact_dir",
        ]
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.columns)
                writer.writeheader()

    def append(self, row: dict[str, Any]) -> None:
        safe_row = {k: row.get(k, "") for k in self.columns}
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(safe_row)

    def load_df(self) -> pd.DataFrame:
        return pd.read_csv(self.csv_path)

    def top_k(self, k: int = 5) -> pd.DataFrame:
        df = self.load_df()
        if df.empty:
            return df
        ok = df[df["status"] == "ok"].copy()
        if ok.empty:
            return ok
        ok["macro_map"] = pd.to_numeric(ok["macro_map"], errors="coerce")
        ok = ok.sort_values("macro_map", ascending=False)
        return ok.head(k)

    def best_by_type(self) -> dict[str, dict[str, Any]]:
        df = self.load_df()
        if df.empty:
            return {}
        df = df[df["status"] == "ok"].copy()
        if df.empty:
            return {}
        df["macro_map"] = pd.to_numeric(df["macro_map"], errors="coerce")
        out: dict[str, dict[str, Any]] = {}
        for model_type, g in df.groupby("model_type"):
            g = g.sort_values("macro_map", ascending=False)
            if len(g):
                out[str(model_type)] = g.iloc[0].to_dict()
        return out

    @staticmethod
    def json_dumps(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=True, sort_keys=True)

