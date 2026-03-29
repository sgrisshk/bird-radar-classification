from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import CLASS_TO_INDEX, CLASSES
from src.cv import make_forward_temporal_group_folds
from src.preprocessing import build_track_cache, load_track_cache, save_track_cache
from src.redesign.dataset import SequenceConfig
from src.redesign.features import build_tabular_frame, get_feature_columns
from src.redesign.ssl_pretrain import run_ssl_pretrain
from src.redesign.train import train_one_fold
from src.redesign.utils import dump_json, macro_map, per_class_ap, set_seed


def _one_hot_targets(y_idx: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.zeros((len(y_idx), n_classes), dtype=np.float32)
    y[np.arange(len(y_idx)), y_idx] = 1.0
    return y


def _resolve_device(requested: str) -> tuple[torch.device, str]:
    req = str(requested).lower().strip()
    if req == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda"), req
        return torch.device("cpu"), req
    if req == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps"), req
        return torch.device("cpu"), req
    if req == "cpu":
        return torch.device("cpu"), req
    return torch.device("cpu"), req


def _load_or_build_cache(df: pd.DataFrame, path: Path) -> dict[int, dict[str, Any]]:
    if path.exists():
        return load_track_cache(path)
    cache = build_track_cache(df)
    save_track_cache(cache, path)
    return cache


def _resolve_existing_path(path_value: str, project_root: Path) -> Path:
    p = Path(path_value).expanduser()
    candidates: list[Path] = [p]
    if not p.is_absolute():
        candidates.append(project_root / p)
        candidates.append(project_root.parent / p)
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError(f"path does not exist: {path_value}")


def _load_distill_teacher_map(
    cfg: dict[str, Any],
    project_root: Path,
    train_track_ids: np.ndarray,
) -> tuple[dict[str, Any] | None, dict[int, np.ndarray] | None, str | None, str]:
    distill_raw = cfg.get("distill", {}) or {}
    if not bool(distill_raw.get("enabled", False)):
        return None, None, None, "track_id"

    teacher_csv = _resolve_existing_path(str(distill_raw.get("teacher_oof_csv", "")), project_root=project_root)
    teacher_id_col = str(distill_raw.get("teacher_id_col", "track_id"))
    teacher_cols_cfg = distill_raw.get("teacher_cols", None)
    if teacher_cols_cfg is None:
        teacher_cols = list(CLASSES)
    else:
        teacher_cols = [str(c) for c in teacher_cols_cfg]
        if len(teacher_cols) != len(CLASSES):
            raise ValueError(
                f"distill.teacher_cols must have exactly {len(CLASSES)} entries; got {len(teacher_cols)}"
            )

    usecols = [teacher_id_col, *list(dict.fromkeys(teacher_cols))]
    teacher_df = pd.read_csv(teacher_csv, usecols=usecols)
    missing_cols = [c for c in [teacher_id_col, *teacher_cols] if c not in teacher_df.columns]
    if missing_cols:
        raise ValueError(f"missing distill columns in {teacher_csv}: {missing_cols}")
    if teacher_df[teacher_id_col].duplicated().any():
        dup = teacher_df.loc[teacher_df[teacher_id_col].duplicated(), teacher_id_col].head(5).tolist()
        raise ValueError(f"duplicate teacher ids in {teacher_csv}: {dup}")

    teacher_ids = teacher_df[teacher_id_col].to_numpy(dtype=np.int64)
    teacher_probs = np.clip(teacher_df[teacher_cols].to_numpy(dtype=np.float32), 0.0, 1.0)
    teacher_map = {int(tid): teacher_probs[i] for i, tid in enumerate(teacher_ids)}

    train_set = set(int(x) for x in train_track_ids.tolist())
    missing_ids = sorted(list(train_set.difference(teacher_map.keys())))
    if missing_ids:
        raise ValueError(
            f"distill teacher OOF is missing {len(missing_ids)} train track_ids; first={missing_ids[:10]}"
        )

    alpha_true = float(distill_raw.get("alpha_true", 0.3))
    alpha_soft = float(distill_raw.get("alpha_soft", 0.7))
    if alpha_true < 0.0 or alpha_soft < 0.0 or (alpha_true + alpha_soft) <= 0.0:
        raise ValueError("distill alpha_true/alpha_soft must be >=0 and not both zero")
    alpha_sum = alpha_true + alpha_soft
    alpha_true /= alpha_sum
    alpha_soft /= alpha_sum

    distill_cfg = {
        "enabled": True,
        "teacher_oof_csv": str(teacher_csv),
        "teacher_id_col": teacher_id_col,
        "teacher_cols": teacher_cols,
        "temperature": float(distill_raw.get("temperature", 2.0)),
        "alpha_true": float(alpha_true),
        "alpha_soft": float(alpha_soft),
        "warmup_soft_epochs": int(distill_raw.get("warmup_soft_epochs", 0)),
        "soft_loss": str(distill_raw.get("soft_loss", "bce")).lower().strip(),
        "residual_mode": bool(distill_raw.get("residual_mode", False)),
        "delta_l2": float(distill_raw.get("delta_l2", 0.0)),
        "selective_conf_enabled": bool(distill_raw.get("selective_conf_enabled", False)),
        "selective_conf_threshold": float(distill_raw.get("selective_conf_threshold", 0.8)),
        "selective_conf_mode": str(distill_raw.get("selective_conf_mode", "max")).lower().strip(),
        "pseudo_target_enabled": bool(distill_raw.get("pseudo_target_enabled", False)),
        "alpha_pseudo": float(distill_raw.get("alpha_pseudo", 0.0)),
        "pseudo_threshold_high": float(distill_raw.get("pseudo_threshold_high", 0.95)),
        "pseudo_threshold_low": float(distill_raw.get("pseudo_threshold_low", 0.05)),
        "pseudo_conf_power": float(distill_raw.get("pseudo_conf_power", 1.0)),
        "consistency_lambda": float(distill_raw.get("consistency_lambda", 0.0)),
        "consistency_time_mask_p": float(distill_raw.get("consistency_time_mask_p", 0.0)),
        "consistency_channel_mask_p": float(distill_raw.get("consistency_channel_mask_p", 0.0)),
        "consistency_noise_std": float(distill_raw.get("consistency_noise_std", 0.0)),
        "teacher_rows": int(len(teacher_df)),
    }
    teacher_test_csv_raw = distill_raw.get("teacher_test_csv", None)
    teacher_test_csv = str(teacher_test_csv_raw).strip() if teacher_test_csv_raw is not None else None
    if teacher_test_csv == "":
        teacher_test_csv = None
    return distill_cfg, teacher_map, teacher_test_csv, teacher_id_col


def _load_distill_teacher_test_map(
    teacher_test_csv: str | None,
    teacher_id_col: str,
    teacher_cols: list[str],
    project_root: Path,
    test_track_ids: np.ndarray,
) -> dict[int, np.ndarray] | None:
    if teacher_test_csv is None:
        return None
    test_csv = _resolve_existing_path(str(teacher_test_csv), project_root=project_root)
    usecols = [teacher_id_col, *list(dict.fromkeys(teacher_cols))]
    test_df = pd.read_csv(test_csv, usecols=usecols)
    missing_cols = [c for c in [teacher_id_col, *teacher_cols] if c not in test_df.columns]
    if missing_cols:
        raise ValueError(f"missing distill test columns in {test_csv}: {missing_cols}")
    if test_df[teacher_id_col].duplicated().any():
        dup = test_df.loc[test_df[teacher_id_col].duplicated(), teacher_id_col].head(5).tolist()
        raise ValueError(f"duplicate teacher test ids in {test_csv}: {dup}")

    test_ids = test_df[teacher_id_col].to_numpy(dtype=np.int64)
    test_probs = np.clip(test_df[teacher_cols].to_numpy(dtype=np.float32), 0.0, 1.0)
    test_map = {int(tid): test_probs[i] for i, tid in enumerate(test_ids)}
    missing = sorted(list(set(int(x) for x in test_track_ids.tolist()).difference(test_map.keys())))
    if missing:
        raise ValueError(f"distill teacher_test_csv missing {len(missing)} test track_ids; first={missing[:10]}")
    return test_map


def _class_weights(y_onehot: np.ndarray) -> np.ndarray:
    pos = y_onehot.sum(axis=0)
    total = float(len(y_onehot))
    neg = total - pos
    w = (neg + 1.0) / (pos + 1.0)
    w = w / np.mean(w)
    return w.astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    z = np.clip(z, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _logit(p: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(q / (1.0 - q)).astype(np.float32)


def _load_aligned_teacher_probs(
    *,
    csv_path: str,
    project_root: Path,
    track_ids: np.ndarray,
    id_col: str = "track_id",
    cols: list[str] | None = None,
) -> np.ndarray:
    use_cols = list(cols) if cols is not None else list(CLASSES)
    path = _resolve_existing_path(csv_path, project_root=project_root)
    df = pd.read_csv(path, usecols=[id_col, *use_cols])
    if id_col not in df.columns:
        raise ValueError(f"teacher csv missing id column '{id_col}': {path}")
    if df[id_col].duplicated().any():
        dup = df.loc[df[id_col].duplicated(), id_col].head(5).tolist()
        raise ValueError(f"teacher csv has duplicated ids in {path}: {dup}")
    idx = df.set_index(id_col)
    missing = [int(t) for t in track_ids.tolist() if int(t) not in idx.index]
    if missing:
        raise ValueError(f"teacher csv {path} missing {len(missing)} ids; first={missing[:10]}")
    arr = idx.loc[track_ids.tolist(), use_cols].to_numpy(dtype=np.float32)
    return np.clip(arr, 1e-6, 1.0 - 1e-6)


def _safe_corr_flat(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return 0.0
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    c = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(c):
        return 0.0
    return c


def _teacher_blend_diagnostics(
    y_true: np.ndarray,
    teacher_prob: np.ndarray,
    model_prob: np.ndarray,
) -> dict[str, Any]:
    teacher_prob = np.clip(np.asarray(teacher_prob, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    model_prob = np.clip(np.asarray(model_prob, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    teacher_macro = float(macro_map(y_true, teacher_prob))
    model_macro = float(macro_map(y_true, model_prob))
    corr = _safe_corr_flat(teacher_prob, model_prob)

    w_grid = [0.99, 0.98, 0.97, 0.95, 0.93, 0.90]
    best_macro = teacher_macro
    best_w = 1.0
    best_pred = teacher_prob
    for w in w_grid:
        pred = np.clip(w * teacher_prob + (1.0 - w) * model_prob, 0.0, 1.0)
        m = float(macro_map(y_true, pred))
        if m > best_macro:
            best_macro = m
            best_w = float(w)
            best_pred = pred
    ap_teacher = per_class_ap(y_true, teacher_prob)
    ap_best = per_class_ap(y_true, best_pred)
    ap_delta = {k: float(ap_best[k] - ap_teacher[k]) for k in CLASSES}
    return {
        "teacher_macro": teacher_macro,
        "model_macro": model_macro,
        "corr_with_teacher": float(corr),
        "best_blend_macro": float(best_macro),
        "best_blend_w_teacher": float(best_w),
        "best_blend_gain": float(best_macro - teacher_macro),
        "per_class_delta_ap_vs_teacher": ap_delta,
    }


def _build_temporal_groups(ts: pd.Series, n_groups: int) -> np.ndarray:
    n = len(ts)
    g = int(max(1, n_groups))
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    if g == 1:
        return np.zeros((n,), dtype=np.int64)
    ts_dt = pd.to_datetime(ts, errors="coerce", utc=True)
    if ts_dt.isna().any():
        bad = int(ts_dt.isna().sum())
        raise ValueError(f"timestamp parsing failed for {bad} rows in temporal group builder")
    ts_ns = ts_dt.astype("int64").to_numpy()
    order = np.argsort(ts_ns, kind="mergesort")
    groups = np.zeros((n,), dtype=np.int64)
    for gid, idxs in enumerate(np.array_split(order, g)):
        if len(idxs) == 0:
            continue
        groups[idxs] = int(gid)
    return groups


_SEQ_CHANNEL_NAME_TO_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
    "rcs": 3,
    "speed": 4,
    "vertical_speed": 5,
    "accel": 6,
    "acceleration": 6,
    "curvature": 7,
    "dt": 8,
}


def _parse_seq_channel_indices(values: Any) -> tuple[int, ...]:
    if values is None:
        return tuple()
    if isinstance(values, (str, int)):
        values = [values]
    out: list[int] = []
    for v in values:
        if isinstance(v, str):
            key = v.strip().lower()
            if key in _SEQ_CHANNEL_NAME_TO_INDEX:
                out.append(int(_SEQ_CHANNEL_NAME_TO_INDEX[key]))
            else:
                out.append(int(key))
        else:
            out.append(int(v))
    return tuple(sorted(set(out)))


def _parse_delta_channels(values: Any) -> tuple[str, ...]:
    if values is None:
        return tuple()
    if isinstance(values, str):
        values = [values]
    out: list[str] = []
    for v in values:
        k = str(v).strip().lower()
        if k:
            out.append(k)
    return tuple(out)


def _parse_group_dropout(
    cfg_sequence: dict[str, Any],
) -> tuple[tuple[tuple[int, ...], ...], tuple[float, ...]]:
    groups_raw = cfg_sequence.get("group_dropout_channels", [])
    probs_raw = cfg_sequence.get("group_dropout_probs", [])
    groups: list[tuple[int, ...]] = []
    probs: list[float] = []

    if groups_raw and probs_raw:
        for g_raw, p_raw in zip(groups_raw, probs_raw):
            g_idx = _parse_seq_channel_indices(g_raw)
            if len(g_idx) == 0:
                continue
            p = float(p_raw)
            if p <= 0.0:
                continue
            groups.append(g_idx)
            probs.append(p)
    else:
        speed_p = float(cfg_sequence.get("group_dropout_speed_p", 0.0))
        dt_rcs_p = float(cfg_sequence.get("group_dropout_dt_rcs_p", 0.0))
        if speed_p > 0.0:
            groups.append(_parse_seq_channel_indices(["speed", "vertical_speed", "accel", "curvature"]))
            probs.append(speed_p)
        if dt_rcs_p > 0.0:
            groups.append(_parse_seq_channel_indices(["dt", "rcs"]))
            probs.append(dt_rcs_p)

    return tuple(groups), tuple(probs)


def _compute_global_seq_stats(
    cache: dict[int, dict[str, Any]],
    channels: int = 9,
    log_dt: bool = False,
    dt_eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    chunks: list[np.ndarray] = []
    for item in cache.values():
        raw = np.asarray(item.get("raw_features", []), dtype=np.float32)
        if raw.ndim != 2 or raw.shape[0] == 0:
            continue
        c = min(int(channels), int(raw.shape[1]))
        chunks.append(raw[:, :c])
    if not chunks:
        median = np.zeros((channels,), dtype=np.float32)
        iqr = np.ones((channels,), dtype=np.float32)
        return median, iqr
    x = np.concatenate(chunks, axis=0)
    if x.shape[1] < channels:
        pad = np.zeros((x.shape[0], channels - x.shape[1]), dtype=np.float32)
        x = np.concatenate([x, pad], axis=1)
    if bool(log_dt) and x.shape[1] > 8:
        x[:, 8] = np.log(np.clip(x[:, 8], float(dt_eps), None)).astype(np.float32)
    median = np.median(x, axis=0).astype(np.float32)
    q25 = np.quantile(x, 0.25, axis=0).astype(np.float32)
    q75 = np.quantile(x, 0.75, axis=0).astype(np.float32)
    iqr = (q75 - q25).astype(np.float32)
    iqr = np.where(iqr < 1e-6, 1.0, iqr).astype(np.float32)
    return median, iqr


def _adversarial_prune(
    train_x: np.ndarray,
    test_x: np.ndarray,
    feature_cols: list[str],
    pct_drop: float,
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    if pct_drop <= 0.0:
        return feature_cols, {
            "auc_mean": 0.0,
            "auc_folds": [],
            "drop_count": 0,
            "dropped": [],
        }

    x = np.concatenate([train_x, test_x], axis=0)
    y = np.concatenate(
        [np.zeros(len(train_x), dtype=np.int32), np.ones(len(test_x), dtype=np.int32)]
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs: list[float] = []
    for i, (tr, va) in enumerate(skf.split(x, y)):
        mdl = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=seed + i,
            verbose=False,
        )
        mdl.fit(x[tr], y[tr], eval_set=(x[va], y[va]), use_best_model=True)
        pred = mdl.predict_proba(x[va])[:, 1]
        aucs.append(float(roc_auc_score(y[va], pred)))

    full = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=700,
        learning_rate=0.05,
        depth=6,
        random_seed=seed,
        verbose=False,
    )
    full.fit(x, y)
    imp = full.get_feature_importance(type="FeatureImportance")
    order = np.argsort(-imp)
    drop_count = int(max(0, min(len(feature_cols) - 1, round(len(feature_cols) * pct_drop))))
    dropped = [feature_cols[i] for i in order[:drop_count]]
    keep = [c for c in feature_cols if c not in set(dropped)]

    return keep, {
        "auc_mean": float(np.mean(aucs)),
        "auc_folds": [float(v) for v in aucs],
        "drop_count": int(drop_count),
        "dropped": dropped,
        "top20": [
            {"feature": feature_cols[i], "importance": float(imp[i])}
            for i in order[:20]
        ],
    }


def _train_catboost_ovr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    cfg: dict[str, Any],
    teacher_train_prob: np.ndarray | None = None,
    teacher_test_prob: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n_train, n_classes = y_train.shape
    oof = np.zeros((n_train, n_classes), dtype=np.float32)
    test_folds: list[np.ndarray] = []
    fallback_events: list[dict[str, Any]] = []
    tab_cfg = cfg.get("tabular", {}) or {}
    mode = str(tab_cfg.get("mode", "ovr_logloss")).lower().strip()
    residual_mode = mode in {"residual_delta_logit", "delta_logit_residual", "delta_logit"}
    if residual_mode and (teacher_train_prob is None or teacher_test_prob is None):
        raise ValueError(
            "tabular.mode=residual_delta_logit requires teacher_train_prob and teacher_test_prob"
        )

    residual_clip = float(tab_cfg.get("delta_clip", 1.0))
    residual_target_clip = float(tab_cfg.get("delta_target_clip", 6.0))
    residual_label_smoothing = float(tab_cfg.get("residual_label_smoothing", 0.01))
    residual_eps = float(tab_cfg.get("residual_eps", 1e-5))

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        fold_test = np.zeros((x_test.shape[0], n_classes), dtype=np.float32)
        for c in range(n_classes):
            ytr = y_train[tr_idx, c]
            yva = y_train[va_idx, c]
            if (not residual_mode) and np.unique(ytr).size < 2:
                prior = float(np.mean(ytr))
                oof[va_idx, c] = prior
                fold_test[:, c] = prior
                fallback_events.append(
                    {
                        "fold": int(fold_id),
                        "class": CLASSES[c],
                        "reason": "single_class_train",
                        "prior": prior,
                    }
                )
                continue

            pos = float(ytr.sum())
            neg = float(len(ytr) - ytr.sum())
            pos_weight = float((neg + 1.0) / (pos + 1.0))

            if residual_mode:
                teacher_tr = np.clip(teacher_train_prob[tr_idx, c], residual_eps, 1.0 - residual_eps)
                teacher_va = np.clip(teacher_train_prob[va_idx, c], residual_eps, 1.0 - residual_eps)
                teacher_te = np.clip(teacher_test_prob[:, c], residual_eps, 1.0 - residual_eps)

                ytr_soft = np.where(
                    ytr > 0.5,
                    1.0 - residual_label_smoothing,
                    residual_label_smoothing,
                ).astype(np.float32)
                yva_soft = np.where(
                    yva > 0.5,
                    1.0 - residual_label_smoothing,
                    residual_label_smoothing,
                ).astype(np.float32)
                delta_target = _logit(ytr_soft, eps=residual_eps) - _logit(teacher_tr, eps=residual_eps)
                delta_target = np.clip(delta_target, -residual_target_clip, residual_target_clip).astype(np.float32)
                delta_val_target = _logit(yva_soft, eps=residual_eps) - _logit(teacher_va, eps=residual_eps)
                delta_val_target = np.clip(
                    delta_val_target,
                    -residual_target_clip,
                    residual_target_clip,
                ).astype(np.float32)

                model = CatBoostRegressor(
                    loss_function="RMSE",
                    eval_metric="RMSE",
                    iterations=int(cfg["tabular"]["iterations"]),
                    depth=int(cfg["tabular"]["depth"]),
                    learning_rate=float(cfg["tabular"]["learning_rate"]),
                    l2_leaf_reg=float(cfg["tabular"]["l2_leaf_reg"]),
                    random_seed=seed + fold_id * 31 + c,
                    verbose=False,
                )
                sample_weight = np.where(ytr > 0.5, pos_weight, 1.0)
                model.fit(
                    x_train[tr_idx],
                    delta_target,
                    eval_set=(x_train[va_idx], delta_val_target),
                    use_best_model=False,
                    sample_weight=sample_weight,
                )

                delta_va = np.clip(
                    model.predict(x_train[va_idx]).astype(np.float32),
                    -residual_clip,
                    residual_clip,
                )
                delta_te = np.clip(
                    model.predict(x_test).astype(np.float32),
                    -residual_clip,
                    residual_clip,
                )
                oof[va_idx, c] = _sigmoid(_logit(teacher_va, eps=residual_eps) + delta_va)
                fold_test[:, c] = _sigmoid(_logit(teacher_te, eps=residual_eps) + delta_te)
            else:
                model = CatBoostClassifier(
                    loss_function="Logloss",
                    eval_metric="Logloss",
                    iterations=int(cfg["tabular"]["iterations"]),
                    depth=int(cfg["tabular"]["depth"]),
                    learning_rate=float(cfg["tabular"]["learning_rate"]),
                    l2_leaf_reg=float(cfg["tabular"]["l2_leaf_reg"]),
                    random_seed=seed + fold_id * 31 + c,
                    verbose=False,
                )
                sample_weight = np.where(ytr > 0.5, pos_weight, 1.0)
                if np.unique(yva).size >= 2:
                    model.fit(
                        x_train[tr_idx],
                        ytr,
                        eval_set=(x_train[va_idx], yva),
                        use_best_model=True,
                        sample_weight=sample_weight,
                    )
                else:
                    model.fit(
                        x_train[tr_idx],
                        ytr,
                        use_best_model=False,
                        sample_weight=sample_weight,
                    )
                oof[va_idx, c] = model.predict_proba(x_train[va_idx])[:, 1]
                fold_test[:, c] = model.predict_proba(x_test)[:, 1]
        test_folds.append(fold_test)

    test_pred = np.mean(test_folds, axis=0)
    score = macro_map(y_train, oof)
    per = per_class_ap(y_train, oof)
    return oof, test_pred, {
        "macro_map": float(score),
        "per_class_ap": per,
        "fallback_events": fallback_events,
        "mode": mode,
        "residual_clip": float(residual_clip),
        "residual_target_clip": float(residual_target_clip),
        "residual_label_smoothing": float(residual_label_smoothing),
    }


def run_oof_training(cfg: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(cfg["seed"]))
    task = str(cfg.get("task", "finetune_cls")).lower().strip()
    if task in {"pretrain_masked_recon", "ssl_pretrain"}:
        return run_ssl_pretrain(cfg)
    if task not in {"finetune_cls", "train", ""}:
        raise ValueError(f"unsupported task='{task}'")

    root = Path(cfg["paths"]["project_root"]).resolve()
    data_dir = Path(cfg["paths"]["data_dir"]).resolve()
    cache_dir = Path(cfg["paths"]["cache_dir"]).resolve()
    out_dir = Path(cfg["paths"]["output_dir"]).resolve()

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    train_cache = _load_or_build_cache(train_df, cache_dir / "train_track_cache.pkl")
    test_cache = _load_or_build_cache(test_df, cache_dir / "test_track_cache.pkl")

    tab_train_df = build_tabular_frame(train_df, train_cache)
    tab_test_df = build_tabular_frame(test_df, test_cache)
    feature_cols = get_feature_columns(tab_train_df)

    train_x_raw = tab_train_df[feature_cols].to_numpy(dtype=np.float32)
    test_x_raw = tab_test_df[feature_cols].to_numpy(dtype=np.float32)

    keep_cols, adv_report = _adversarial_prune(
        train_x=train_x_raw,
        test_x=test_x_raw,
        feature_cols=feature_cols,
        pct_drop=float(cfg["shift"]["drop_top_feature_pct"]),
        seed=int(cfg["seed"]),
    )

    train_x = tab_train_df[keep_cols].to_numpy(dtype=np.float32)
    test_x = tab_test_df[keep_cols].to_numpy(dtype=np.float32)

    mu = train_x.mean(axis=0, keepdims=True)
    sd = train_x.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    train_x = (train_x - mu) / sd
    test_x = (test_x - mu) / sd

    y_idx = train_df["bird_group"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    y = _one_hot_targets(y_idx, n_classes=len(CLASSES))

    cv_direction = str((cfg.get("cv", {}) or {}).get("direction", "forward")).strip().lower()
    if cv_direction not in {"forward", "reverse"}:
        raise ValueError(f"cv.direction must be one of ['forward','reverse'], got: {cv_direction}")
    cv_ts = pd.to_datetime(train_df["timestamp_start_radar_utc"], errors="coerce", utc=True)
    if cv_ts.isna().any():
        raise ValueError("timestamp_start_radar_utc contains unparsable values")
    if cv_direction == "reverse":
        ts_ns = cv_ts.astype("int64").to_numpy(dtype=np.int64)
        ts_min = int(np.min(ts_ns))
        ts_max = int(np.max(ts_ns))
        rev_ns = ts_min + (ts_max - ts_ns)
        cv_ts = pd.to_datetime(rev_ns, utc=True)
        print("[redesign][cv] direction=reverse", flush=True)
    cv_df = pd.DataFrame(
        {
            "_cv_ts": cv_ts,
            "_cv_group": train_df["observation_id"].astype(np.int64),
        }
    )
    folds = make_forward_temporal_group_folds(
        cv_df,
        timestamp_col="_cv_ts",
        group_col="_cv_group",
        n_splits=int(cfg["cv"]["n_splits"]),
    )
    debug_cfg = cfg.get("debug", {})
    overfit_n = int(debug_cfg.get("overfit_subset", 0))
    debug_overfit_active = overfit_n > 0
    debug_overfit_meta: dict[str, Any] = {"active": False}
    if debug_overfit_active:
        fold_id = int(debug_cfg.get("overfit_fold", 0))
        fold_id = max(0, min(fold_id, len(folds) - 1))
        tr_idx, va_idx = folds[fold_id]
        n = min(overfit_n, len(tr_idx))
        tr_small = tr_idx[:n]
        if bool(debug_cfg.get("overfit_same_val", True)):
            va_small = tr_small.copy()
        else:
            start = n
            end = min(start + n, len(tr_idx))
            if end - start >= max(8, n // 4):
                va_small = tr_idx[start:end]
            else:
                va_small = va_idx[: min(len(va_idx), max(8, n))]
        folds = [(tr_small.astype(np.int64), va_small.astype(np.int64))]
        debug_overfit_meta = {
            "active": True,
            "overfit_subset": int(overfit_n),
            "fold_id": int(fold_id),
            "overfit_same_val": bool(debug_cfg.get("overfit_same_val", True)),
            "train_size": int(len(tr_small)),
            "val_size": int(len(va_small)),
        }
        print(f"[redesign][overfit] {debug_overfit_meta}", flush=True)

    max_folds = int(debug_cfg.get("max_folds", 0))
    start_fold = int(debug_cfg.get("start_fold", 0))
    if start_fold > 0 and len(folds) > 0:
        s = min(start_fold, len(folds))
        folds = folds[s:]
        print(
            f"[redesign][debug] starting from fold index {s} (skipped {s} earliest folds)",
            flush=True,
        )
    if max_folds > 0 and len(folds) > max_folds:
        folds = folds[:max_folds]
        print(
            f"[redesign][debug] limiting folds to first {max_folds}",
            flush=True,
        )
    partial_cv_active = len(folds) < int(cfg["cv"]["n_splits"])
    covered_idx = np.unique(np.concatenate([va for _, va in folds])).astype(np.int64)

    log_dt = bool(cfg["sequence"].get("log_dt", False))
    dt_eps = float(cfg["sequence"].get("dt_eps", 1e-6))
    seq_median, seq_iqr = _compute_global_seq_stats(train_cache, channels=9, log_dt=log_dt, dt_eps=dt_eps)
    group_dropout_channels, group_dropout_probs = _parse_group_dropout(cfg.get("sequence", {}))
    seq_cfg = SequenceConfig(
        seq_len=int(cfg["sequence"]["seq_len"]),
        time_crop_min=float(cfg["aug"]["time_crop_min"]),
        p_time_reverse=float(cfg["aug"]["time_reverse_prob"]),
        norm_mode=str(cfg["sequence"].get("norm_mode", "global_robust")),
        global_median=seq_median,
        global_iqr=seq_iqr,
        clip_abs=float(cfg["sequence"].get("clip_abs", 30.0)),
        keep_raw_channels=_parse_seq_channel_indices(cfg["sequence"].get("keep_raw_channels", [])) or None,
        robust_channels=_parse_seq_channel_indices(cfg["sequence"].get("robust_channels", [])) or None,
        delta_channels=_parse_delta_channels(cfg["sequence"].get("delta_channels", [])) or None,
        channel_dropout_p=float(cfg["sequence"].get("channel_dropout_p", 0.0)),
        time_dropout_p=float(cfg["sequence"].get("time_dropout_p", 0.0)),
        channel_dropout_candidates=_parse_seq_channel_indices(cfg["sequence"].get("channel_dropout_candidates", [3, 4, 6, 8])) or None,
        group_dropout_channels=group_dropout_channels or None,
        group_dropout_probs=group_dropout_probs or None,
        log_dt=log_dt,
        dt_eps=dt_eps,
        delta_clip_abs=float(cfg["sequence"].get("delta_clip_abs", 10.0)),
    )
    seq_in_channels = 9 + len(seq_cfg.delta_channels or ())
    cfg.setdefault("sequence", {})
    cfg["sequence"]["seq_in_channels"] = int(seq_in_channels)

    device, requested_device = _resolve_device(cfg["train"]["device"])
    if str(device) != requested_device:
        strict_device = bool(cfg.get("train", {}).get("strict_device", False))
        msg = (
            f"[redesign] requested device='{requested_device}' is unavailable, "
            f"resolved device='{device}'"
        )
        if strict_device:
            raise RuntimeError(msg + " and strict_device=true")
        print(msg + " (fallback enabled)", flush=True)

    train_track_ids = train_df["track_id"].to_numpy(dtype=np.int64)
    test_track_ids = test_df["track_id"].to_numpy(dtype=np.int64)
    hard_groups_n = int(max(1, cfg.get("train", {}).get("hard_mining_time_groups", 8)))
    train_time_groups = _build_temporal_groups(
        train_df["timestamp_start_radar_utc"],
        n_groups=hard_groups_n,
    )
    distill_cfg, distill_teacher_map, distill_teacher_test_csv, distill_teacher_id_col = _load_distill_teacher_map(
        cfg=cfg,
        project_root=root,
        train_track_ids=train_track_ids,
    )
    distill_teacher_test_map: dict[int, np.ndarray] | None = None
    if distill_cfg is not None:
        distill_teacher_test_map = _load_distill_teacher_test_map(
            teacher_test_csv=distill_teacher_test_csv,
            teacher_id_col=distill_teacher_id_col,
            teacher_cols=list(distill_cfg["teacher_cols"]),
            project_root=root,
            test_track_ids=test_track_ids,
        )
        distill_cfg["teacher_test_csv"] = (
            str(_resolve_existing_path(distill_teacher_test_csv, project_root=root))
            if distill_teacher_test_csv is not None
            else None
        )
        distill_cfg["teacher_test_rows"] = int(len(distill_teacher_test_map) if distill_teacher_test_map is not None else 0)
        if bool(distill_cfg.get("residual_mode", False)) and distill_teacher_test_map is None:
            raise ValueError("distill.residual_mode=true requires distill.teacher_test_csv with full test coverage")
    if distill_cfg is not None:
        print(
            "[redesign][distill] "
            f"enabled teacher={distill_cfg['teacher_oof_csv']} "
            f"T={distill_cfg['temperature']} "
            f"alpha_true={distill_cfg['alpha_true']:.3f} "
            f"alpha_soft={distill_cfg['alpha_soft']:.3f} "
            f"selective={bool(distill_cfg.get('selective_conf_enabled', False))} "
            f"thr={float(distill_cfg.get('selective_conf_threshold', 0.8)):.3f} "
            f"mode={str(distill_cfg.get('selective_conf_mode', 'max'))} "
            f"pseudo_target={bool(distill_cfg.get('pseudo_target_enabled', False))} "
            f"alpha_pseudo={float(distill_cfg.get('alpha_pseudo', 0.0)):.3f} "
            f"pseudo_hi={float(distill_cfg.get('pseudo_threshold_high', 0.95)):.3f} "
            f"pseudo_lo={float(distill_cfg.get('pseudo_threshold_low', 0.05)):.3f} "
            f"cons={float(distill_cfg.get('consistency_lambda', 0.0)):.3f}",
            flush=True,
        )

    if (debug_overfit_active or partial_cv_active) and len(covered_idx) > 0:
        eval_idx = covered_idx
    else:
        eval_idx = np.arange(len(train_df), dtype=np.int64)

    teacher_eval_prob: np.ndarray | None = None
    if distill_teacher_map is not None:
        teacher_all = np.stack(
            [np.asarray(distill_teacher_map[int(tid)], dtype=np.float32) for tid in train_track_ids],
            axis=0,
        )
        teacher_eval_prob = np.clip(teacher_all[eval_idx], 1e-6, 1.0 - 1.0e-6)
    y_eval = y[eval_idx]

    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_registry: dict[str, dict[str, Any]] = {}
    deep_save_embeddings = bool((cfg.get("train", {}) or {}).get("save_embeddings", False))

    for arch in cfg["model"]["architectures"]:
        for seed in cfg["model"]["seeds"]:
            name = f"deep_{arch}_seed{seed}"
            model_path = artifacts_dir / f"{name}_oof.npy"
            test_path = artifacts_dir / f"{name}_test.npy"
            emb_oof_path = artifacts_dir / f"{name}_oof_emb.npy"
            emb_test_path = artifacts_dir / f"{name}_test_emb.npy"

            if model_path.exists() and test_path.exists():
                oof_cached = np.load(model_path)
                test_cached = np.load(test_path)
                if oof_cached.shape == (len(train_df), len(CLASSES)) and test_cached.shape == (len(test_df), len(CLASSES)):
                    if deep_save_embeddings and (not emb_oof_path.exists() or not emb_test_path.exists()):
                        # Need a real forward pass to materialize embeddings even if probs are cached.
                        pass
                    else:
                        fold_scores = []
                        for tr_idx, va_idx in folds:
                            _ = tr_idx
                            fold_scores.append(float(macro_map(y[va_idx], oof_cached[va_idx])))
                        if (debug_overfit_active or partial_cv_active) and len(covered_idx) > 0:
                            macro_value = float(macro_map(y[covered_idx], oof_cached[covered_idx]))
                            per_value = per_class_ap(y[covered_idx], oof_cached[covered_idx])
                        else:
                            macro_value = float(macro_map(y, oof_cached))
                            per_value = per_class_ap(y, oof_cached)
                        teacher_diag: dict[str, Any] = {}
                        if teacher_eval_prob is not None:
                            teacher_diag = _teacher_blend_diagnostics(
                                y_true=y_eval,
                                teacher_prob=teacher_eval_prob,
                                model_prob=oof_cached[eval_idx],
                            )
                        model_registry[name] = {
                            "type": "deep",
                            "architecture": arch,
                            "seed": int(seed),
                            "oof_path": str(model_path),
                            "test_path": str(test_path),
                            "oof_emb_path": str(emb_oof_path) if emb_oof_path.exists() else None,
                            "test_emb_path": str(emb_test_path) if emb_test_path.exists() else None,
                            "macro_map": macro_value,
                            "per_class_ap": per_value,
                            "fold_scores": fold_scores,
                            "fold_train_scores": [],
                            "fold_val_pred_diag": [],
                            "weak_classes": [],
                            "resumed_from_cache": True,
                            "teacher_diag": teacher_diag,
                        }
                        continue

            oof = np.zeros((len(train_df), len(CLASSES)), dtype=np.float32)
            test_accum = np.zeros((len(test_df), len(CLASSES)), dtype=np.float32)
            oof_emb: np.ndarray | None = None
            test_emb_accum: np.ndarray | None = None
            fold_scores: list[float] = []
            fold_train_scores: list[float] = []
            fold_val_pred_diag: list[dict[str, float]] = []
            fold_per_class: list[dict[str, float]] = []
            weak_classes: list[list[str]] = []

            for fold_id, (tr_idx, va_idx) in enumerate(folds):
                cw = _class_weights(y[tr_idx])
                fold = train_one_fold(
                    train_track_ids=train_track_ids[tr_idx],
                    val_track_ids=train_track_ids[va_idx],
                    train_tab=train_x[tr_idx],
                    val_tab=train_x[va_idx],
                    test_track_ids=test_track_ids,
                    test_tab=test_x,
                    train_y=y[tr_idx],
                    val_y=y[va_idx],
                    cache_train=train_cache,
                    cache_test=test_cache,
                    seq_cfg=seq_cfg,
                    arch_name=arch,
                    seed=int(seed) + fold_id * 101,
                    cfg=cfg,
                    class_weights=cw,
                    device=device,
                    train_time_groups=train_time_groups[tr_idx],
                    distill_cfg=distill_cfg,
                    distill_teacher_map=distill_teacher_map,
                    distill_teacher_test_map=distill_teacher_test_map,
                )
                oof[va_idx] = fold.oof_pred
                test_accum += fold.test_pred / len(folds)
                if deep_save_embeddings and fold.oof_emb is not None and fold.test_emb is not None:
                    if oof_emb is None:
                        emb_dim = int(fold.oof_emb.shape[1])
                        oof_emb = np.zeros((len(train_df), emb_dim), dtype=np.float32)
                        test_emb_accum = np.zeros((len(test_df), emb_dim), dtype=np.float32)
                    oof_emb[va_idx] = fold.oof_emb
                    test_emb_accum += fold.test_emb / len(folds)
                fold_scores.append(float(macro_map(y[va_idx], fold.oof_pred)))
                if fold.train_score is not None:
                    fold_train_scores.append(float(fold.train_score))
                if fold.val_pred_diag is not None:
                    fold_val_pred_diag.append(dict(fold.val_pred_diag))
                fold_per_class.append(fold.per_class)
                weak_classes.append(fold.weak_classes)

            np.save(model_path, oof)
            np.save(test_path, test_accum)
            if oof_emb is not None and test_emb_accum is not None:
                np.save(emb_oof_path, oof_emb)
                np.save(emb_test_path, test_emb_accum)
            if (debug_overfit_active or partial_cv_active) and len(covered_idx) > 0:
                macro_value = float(macro_map(y[covered_idx], oof[covered_idx]))
                per_value = per_class_ap(y[covered_idx], oof[covered_idx])
            else:
                macro_value = float(macro_map(y, oof))
                per_value = per_class_ap(y, oof)
            teacher_diag: dict[str, Any] = {}
            if teacher_eval_prob is not None:
                teacher_diag = _teacher_blend_diagnostics(
                    y_true=y_eval,
                    teacher_prob=teacher_eval_prob,
                    model_prob=oof[eval_idx],
                )

            model_registry[name] = {
                "type": "deep",
                "architecture": arch,
                "seed": int(seed),
                "oof_path": str(model_path),
                "test_path": str(test_path),
                "oof_emb_path": str(emb_oof_path) if emb_oof_path.exists() else None,
                "test_emb_path": str(emb_test_path) if emb_test_path.exists() else None,
                "macro_map": macro_value,
                "per_class_ap": per_value,
                "fold_scores": [float(s) for s in fold_scores],
                "fold_train_scores": [float(s) for s in fold_train_scores],
                "fold_val_pred_diag": fold_val_pred_diag,
                "weak_classes": weak_classes,
                "resumed_from_cache": False,
                "teacher_diag": teacher_diag,
            }

    tabular_enabled = bool((cfg.get("tabular", {}) or {}).get("enabled", True))
    if tabular_enabled:
        # Pure tabular CatBoost member
        teacher_train_prob_tab: np.ndarray | None = None
        teacher_test_prob_tab: np.ndarray | None = None
        tab_mode = str((cfg.get("tabular", {}) or {}).get("mode", "ovr_logloss")).lower().strip()
        if tab_mode in {"residual_delta_logit", "delta_logit_residual", "delta_logit"}:
            tab_teacher_oof = str((cfg.get("tabular", {}) or {}).get("teacher_oof_csv", "")).strip()
            tab_teacher_test = str((cfg.get("tabular", {}) or {}).get("teacher_test_csv", "")).strip()
            if not tab_teacher_oof or not tab_teacher_test:
                raise ValueError(
                    "tabular.mode=residual_delta_logit requires tabular.teacher_oof_csv and tabular.teacher_test_csv"
                )
            teacher_train_prob_tab = _load_aligned_teacher_probs(
                csv_path=tab_teacher_oof,
                project_root=root,
                track_ids=train_track_ids,
                id_col=str((cfg.get("tabular", {}) or {}).get("teacher_id_col", "track_id")),
            )
            teacher_test_prob_tab = _load_aligned_teacher_probs(
                csv_path=tab_teacher_test,
                project_root=root,
                track_ids=test_track_ids,
                id_col=str((cfg.get("tabular", {}) or {}).get("teacher_id_col", "track_id")),
            )

        cat_name = "catboost_ovr"
        cat_oof, cat_test, cat_report = _train_catboost_ovr(
            x_train=train_x,
            y_train=y,
            x_test=test_x,
            folds=folds,
            seed=int(cfg["seed"]),
            cfg=cfg,
            teacher_train_prob=teacher_train_prob_tab,
            teacher_test_prob=teacher_test_prob_tab,
        )
        cat_oof_path = artifacts_dir / f"{cat_name}_oof.npy"
        cat_test_path = artifacts_dir / f"{cat_name}_test.npy"
        np.save(cat_oof_path, cat_oof)
        np.save(cat_test_path, cat_test)
        if (debug_overfit_active or partial_cv_active) and len(covered_idx) > 0:
            cat_macro = float(macro_map(y[covered_idx], cat_oof[covered_idx]))
            cat_per = per_class_ap(y[covered_idx], cat_oof[covered_idx])
        else:
            cat_macro = float(cat_report["macro_map"])
            cat_per = cat_report["per_class_ap"]

        model_registry[cat_name] = {
            "type": "tabular",
            "oof_path": str(cat_oof_path),
            "test_path": str(cat_test_path),
            "macro_map": cat_macro,
            "per_class_ap": cat_per,
            "fallback_events": cat_report.get("fallback_events", []),
            "mode": str(cat_report.get("mode", "ovr_logloss")),
            "residual_clip": cat_report.get("residual_clip", None),
            "residual_target_clip": cat_report.get("residual_target_clip", None),
            "residual_label_smoothing": cat_report.get("residual_label_smoothing", None),
        }

    summary = {
        "project_root": str(root),
        "output_dir": str(out_dir),
        "cv_direction": cv_direction,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "features_total": int(len(feature_cols)),
        "features_kept": int(len(keep_cols)),
        "features_dropped": sorted(list(set(feature_cols) - set(keep_cols))),
        "kept_feature_columns": keep_cols,
        "adversarial": adv_report,
        "debug_overfit": debug_overfit_meta,
        "requested_device": requested_device,
        "device": str(device),
        "sequence_norm_mode": str(cfg["sequence"].get("norm_mode", "global_robust")),
        "sequence_clip_abs": float(cfg["sequence"].get("clip_abs", 30.0)),
        "sequence_seq_in_channels": int(seq_in_channels),
        "sequence_keep_raw_channels": list(seq_cfg.keep_raw_channels) if seq_cfg.keep_raw_channels is not None else None,
        "sequence_robust_channels": list(seq_cfg.robust_channels) if seq_cfg.robust_channels is not None else None,
        "sequence_delta_channels": list(seq_cfg.delta_channels) if seq_cfg.delta_channels is not None else [],
        "sequence_channel_dropout_p": float(seq_cfg.channel_dropout_p),
        "sequence_time_dropout_p": float(seq_cfg.time_dropout_p),
        "sequence_channel_dropout_candidates": list(seq_cfg.channel_dropout_candidates) if seq_cfg.channel_dropout_candidates is not None else [],
        "sequence_group_dropout_channels": [list(g) for g in (seq_cfg.group_dropout_channels or ())],
        "sequence_group_dropout_probs": [float(p) for p in (seq_cfg.group_dropout_probs or ())],
        "sequence_log_dt": bool(seq_cfg.log_dt),
        "sequence_dt_eps": float(seq_cfg.dt_eps),
        "sequence_delta_clip_abs": float(seq_cfg.delta_clip_abs) if seq_cfg.delta_clip_abs is not None else None,
        "sequence_global_median": [float(v) for v in seq_median.tolist()],
        "sequence_global_iqr": [float(v) for v in seq_iqr.tolist()],
        "distill": distill_cfg if distill_cfg is not None else {"enabled": False},
        "models": model_registry,
        "track_ids_test_path": str((out_dir / "test_track_ids.npy").resolve()),
        "track_ids_train_path": str((out_dir / "train_track_ids.npy").resolve()),
    }

    np.save(out_dir / "test_track_ids.npy", test_track_ids)
    np.save(out_dir / "train_track_ids.npy", train_track_ids)
    np.save(out_dir / "oof_targets.npy", y)

    dump_json(out_dir / "oof_summary.json", summary)
    return summary
