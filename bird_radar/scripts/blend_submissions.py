from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=str, required=True, help="Comma-separated submission CSV paths")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument(
        "--weights",
        type=str,
        default="",
        help="Comma-separated weights matching --inputs. Optional; defaults to uniform.",
    )
    p.add_argument(
        "--modes",
        type=str,
        default="mean,logit_mean,rank_mean",
        help="Comma-separated: mean,logit_mean,rank_mean",
    )
    return p.parse_args()


def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def rank_normalize_col(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks / max(1, len(x) - 1)


def rank_normalize_matrix(p: np.ndarray) -> np.ndarray:
    out = np.zeros_like(p, dtype=np.float64)
    for c in range(p.shape[1]):
        out[:, c] = rank_normalize_col(p[:, c])
    return out


def parse_weights(raw: str, n: int) -> np.ndarray:
    if not raw.strip():
        return np.ones(n, dtype=np.float64) / n
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) != n:
        raise ValueError(f"--weights length {len(vals)} does not match inputs {n}")
    w = np.asarray(vals, dtype=np.float64)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 0.0:
        raise ValueError("sum of weights must be > 0")
    return w / s


def load_submissions(paths: list[Path]) -> tuple[np.ndarray, list[str], list[np.ndarray]]:
    dfs = [pd.read_csv(p) for p in paths]
    for i, df in enumerate(dfs):
        if "track_id" not in df.columns:
            raise ValueError(f"{paths[i]} missing track_id")
    cols0 = [c for c in dfs[0].columns if c != "track_id"]
    ids0 = dfs[0]["track_id"].to_numpy()
    mats: list[np.ndarray] = []
    for i, df in enumerate(dfs):
        cols = [c for c in df.columns if c != "track_id"]
        if cols != cols0:
            raise ValueError(f"class columns mismatch in {paths[i]}")
        ids = df["track_id"].to_numpy()
        if len(ids) != len(ids0) or not np.array_equal(ids, ids0):
            raise ValueError(f"track_id mismatch in {paths[i]}")
        mat = np.clip(df[cols0].to_numpy(dtype=np.float64), 0.0, 1.0)
        mats.append(mat)
    return ids0, cols0, mats


def blend_mean(mats: list[np.ndarray], w: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mats[0], dtype=np.float64)
    for i, m in enumerate(mats):
        out += w[i] * m
    return np.clip(out, 0.0, 1.0)


def blend_logit_mean(mats: list[np.ndarray], w: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mats[0], dtype=np.float64)
    for i, m in enumerate(mats):
        out += w[i] * to_logit(m)
    return np.clip(sigmoid(out), 0.0, 1.0)


def blend_rank_mean(mats: list[np.ndarray], w: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mats[0], dtype=np.float64)
    for i, m in enumerate(mats):
        out += w[i] * rank_normalize_matrix(m)
    return np.clip(out, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    input_paths = [Path(x.strip()).resolve() for x in args.inputs.split(",") if x.strip()]
    if len(input_paths) < 2:
        raise ValueError("need at least 2 input files")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    allowed = {"mean", "logit_mean", "rank_mean"}
    bad = [m for m in modes if m not in allowed]
    if bad:
        raise ValueError(f"unsupported mode(s): {bad}")

    track_ids, class_cols, mats = load_submissions(input_paths)
    w = parse_weights(args.weights, len(mats))

    outputs: dict[str, str] = {}
    summaries: dict[str, dict[str, float]] = {}

    for mode in modes:
        if mode == "mean":
            pred = blend_mean(mats, w)
            fname = "sub_blend_mean.csv"
        elif mode == "logit_mean":
            pred = blend_logit_mean(mats, w)
            fname = "sub_blend_logit_mean.csv"
        else:
            pred = blend_rank_mean(mats, w)
            fname = "sub_blend_rank_mean.csv"

        sub = pd.DataFrame(pred, columns=class_cols)
        sub.insert(0, "track_id", track_ids)
        out_path = out_dir / fname
        sub.to_csv(out_path, index=False)
        outputs[mode] = str(out_path)
        summaries[mode] = {c: float(np.mean(sub[c])) for c in class_cols}

    report = {
        "inputs": [str(p) for p in input_paths],
        "weights": [float(x) for x in w.tolist()],
        "modes": modes,
        "outputs": outputs,
        "mean_probabilities": summaries,
    }
    (out_dir / "blend_report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    for mode in modes:
        print(outputs[mode], flush=True)


if __name__ == "__main__":
    main()
