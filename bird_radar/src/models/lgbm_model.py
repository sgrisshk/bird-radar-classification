from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as e:  # pragma: no cover
    raise ImportError("lightgbm is required for train_lgbm.py") from e


@dataclass
class LGBMParams:
    n_estimators: int = 8000
    learning_rate: float = 0.02
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.5
    reg_lambda: float = 1.0
    random_state: int = 2026
    n_jobs: int = -1


class LGBMOVRClassifier:
    def __init__(self, num_classes: int, params: LGBMParams | None = None) -> None:
        self.num_classes = num_classes
        self.params = params or LGBMParams()
        self.models: list[lgb.LGBMClassifier] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_valid: pd.DataFrame,
        y_valid: np.ndarray,
        early_stopping_rounds: int = 300,
    ) -> "LGBMOVRClassifier":
        self.models = []
        for c in range(self.num_classes):
            y_tr = y_train[:, c].astype(int)
            y_va = y_valid[:, c].astype(int)
            pos = max(int(y_tr.sum()), 1)
            neg = max(int(len(y_tr) - y_tr.sum()), 1)
            scale_pos_weight = neg / pos

            model = lgb.LGBMClassifier(
                objective="binary",
                metric="average_precision",
                n_estimators=self.params.n_estimators,
                learning_rate=self.params.learning_rate,
                num_leaves=self.params.num_leaves,
                max_depth=self.params.max_depth,
                min_child_samples=self.params.min_child_samples,
                subsample=self.params.subsample,
                colsample_bytree=self.params.colsample_bytree,
                reg_alpha=self.params.reg_alpha,
                reg_lambda=self.params.reg_lambda,
                random_state=self.params.random_state + c,
                n_jobs=self.params.n_jobs,
                scale_pos_weight=scale_pos_weight,
                verbosity=-1,
            )

            callbacks = [
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ]
            model.fit(
                X_train,
                y_tr,
                eval_set=[(X_valid, y_va)],
                eval_metric="average_precision",
                callbacks=callbacks,
            )
            self.models.append(model)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise RuntimeError("LGBMOVRClassifier is not fitted")
        preds = []
        for model in self.models:
            p = model.predict_proba(X)[:, 1]
            preds.append(p.astype(np.float32))
        return np.stack(preds, axis=1)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "LGBMOVRClassifier":
        with Path(path).open("rb") as f:
            return pickle.load(f)

