from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score

from config import CLASSES

EPS = 1e-8


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2)


def config_hash(cfg: dict[str, Any]) -> str:
    s = json.dumps(cfg, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def macro_map(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    vals: list[float] = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        if yt.sum() <= 0:
            vals.append(0.0)
        else:
            vals.append(float(average_precision_score(yt, yp)))
    return float(np.mean(vals))


def per_class_ap(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, cls in enumerate(CLASSES):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        out[cls] = float(average_precision_score(yt, yp) if yt.sum() > 0 else 0.0)
    return out


def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def apply_temperature_to_probs(probs: np.ndarray, temps: np.ndarray) -> np.ndarray:
    logits = to_logit(probs)
    scaled = logits / temps.reshape(1, -1)
    return np.clip(sigmoid(scaled), 0.0, 1.0)


def optimize_temperature_per_class(
    y_true: np.ndarray,
    probs: np.ndarray,
    t_min: float = 0.5,
    t_max: float = 3.0,
) -> tuple[np.ndarray, float, float]:
    raw = macro_map(y_true, probs)
    temps = np.ones(probs.shape[1], dtype=np.float64)

    for c in range(probs.shape[1]):
        yt = y_true[:, c]
        if yt.sum() == 0:
            temps[c] = 1.0
            continue
        logit_c = to_logit(probs[:, c])

        def obj(t: float) -> float:
            pred = sigmoid(logit_c / max(float(t), 1e-6))
            return -float(average_precision_score(yt, pred))

        res = minimize_scalar(obj, bounds=(t_min, t_max), method="bounded", options={"xatol": 1e-3})
        temps[c] = float(res.x if res.success else 1.0)

    cal = apply_temperature_to_probs(probs, temps)
    cal_score = macro_map(y_true, cal)
    return temps.astype(np.float32), float(raw), float(cal_score)


def topk_weak_classes(per_class: dict[str, float], k: int = 3) -> list[str]:
    return [x[0] for x in sorted(per_class.items(), key=lambda z: z[1])[:k]]
