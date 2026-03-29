from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Experiment:
    name: str
    model_type: str
    seed: int
    hyperparams: dict[str, Any]
    max_len: int | None = None
    batch_size: int | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _lgbm_variants() -> list[dict[str, Any]]:
    return [
        {"variant": "lgbm_v1", "learning_rate": 0.02, "num_leaves": 31, "feature_fraction": 0.80, "n_estimators": 8000},
        {"variant": "lgbm_v2", "learning_rate": 0.03, "num_leaves": 63, "feature_fraction": 0.75, "n_estimators": 6000},
        {"variant": "lgbm_v3", "learning_rate": 0.015, "num_leaves": 47, "feature_fraction": 0.90, "n_estimators": 10000},
    ]


def _seq_variants() -> list[dict[str, Any]]:
    return [
        {"variant": "seq_v1", "dropout": 0.30, "weight_decay": 1e-4, "max_len": 512, "batch_size": 16, "epochs": 12},
        {"variant": "seq_v2", "dropout": 0.35, "weight_decay": 2e-4, "max_len": 512, "batch_size": 12, "epochs": 12},
        {"variant": "seq_v3", "dropout": 0.40, "weight_decay": 3e-4, "max_len": 384, "batch_size": 16, "epochs": 12},
        {"variant": "seq_v4", "dropout": 0.30, "weight_decay": 5e-4, "max_len": 256, "batch_size": 16, "epochs": 12},
    ]


def generate_experiment_queue(
    seeds: list[int] | None = None,
    mode: str = "all",
) -> list[Experiment]:
    seeds = seeds or [42, 2026, 777]
    mode = mode.lower()
    out: list[Experiment] = []

    if mode in {"all", "lgbm"}:
        for seed in seeds:
            for v in _lgbm_variants():
                name = f"{v['variant']}_seed{seed}"
                hp = {
                    "learning_rate": float(v["learning_rate"]),
                    "num_leaves": int(v["num_leaves"]),
                    "feature_fraction": float(v["feature_fraction"]),
                    "n_estimators": int(v["n_estimators"]),
                }
                out.append(
                    Experiment(
                        name=name,
                        model_type="lgbm",
                        seed=int(seed),
                        hyperparams=hp,
                        notes=f"{v['variant']} grouped CV by observation_id",
                    )
                )

    if mode in {"all", "seq"}:
        for seed in seeds:
            for v in _seq_variants():
                name = f"{v['variant']}_seed{seed}"
                hp = {
                    "dropout": float(v["dropout"]),
                    "weight_decay": float(v["weight_decay"]),
                    "epochs": int(v["epochs"]),
                }
                out.append(
                    Experiment(
                        name=name,
                        model_type="seq",
                        seed=int(seed),
                        hyperparams=hp,
                        max_len=int(v["max_len"]),
                        batch_size=int(v["batch_size"]),
                        notes=f"{v['variant']} compact hybrid",
                    )
                )

    return out

