#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASSES

EPS = 1e-6


def _align_probs_from_csv(csv_path: str, ids: np.ndarray) -> np.ndarray:
    df = pd.read_csv(csv_path, usecols=["track_id", *CLASSES]).set_index("track_id")
    return df.loc[ids, CLASSES].to_numpy(dtype=np.float32)


def _align_oof_from_npy(npy_path: str, ids_path: str, target_ids: np.ndarray) -> np.ndarray:
    arr = np.load(npy_path).astype(np.float32)
    ids = np.load(ids_path).astype(np.int64)
    pos = {int(t): i for i, t in enumerate(ids.tolist())}
    missing = [int(t) for t in target_ids if int(t) not in pos]
    if missing:
        raise ValueError(f"{npy_path} missing {len(missing)} train ids; first={missing[:10]}")
    return np.stack([arr[pos[int(t)]] for t in target_ids], axis=0).astype(np.float32)


def _margin(p: np.ndarray) -> np.ndarray:
    ps = np.sort(p, axis=1)
    return (ps[:, -1] - ps[:, -2]).astype(np.float32)


def _entropy(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, EPS, 1.0 - EPS)
    return (-np.sum(pc * np.log(pc), axis=1)).astype(np.float32)


def _safe_logit(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, EPS, 1.0 - EPS)
    return np.log(pc / (1.0 - pc)).astype(np.float32)


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    t = max(float(temperature), 1e-6)
    z = x / t
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    s = e / np.clip(np.sum(e, axis=1, keepdims=True), EPS, None)
    return s.astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build soft mixture bank from multiple specialists.")
    p.add_argument("--teacher-oof-csv", required=True)
    p.add_argument("--teacher-test-csv", required=True)
    p.add_argument("--train-track-ids-npy", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--spec1-oof-npy", required=True)
    p.add_argument("--spec1-track-ids-npy", required=True)
    p.add_argument("--spec1-test-csv", required=True)
    p.add_argument("--spec1-name", default="spec1")

    p.add_argument("--spec2-oof-npy", required=True)
    p.add_argument("--spec2-track-ids-npy", required=True)
    p.add_argument("--spec2-test-csv", required=True)
    p.add_argument("--spec2-name", default="spec2")

    p.add_argument("--spec3-oof-npy", default="")
    p.add_argument("--spec3-track-ids-npy", default="")
    p.add_argument("--spec3-test-csv", default="")
    p.add_argument("--spec3-name", default="spec3")

    p.add_argument("--coef-proxy", type=float, default=1.0)
    p.add_argument("--coef-margin", type=float, default=0.5)
    p.add_argument("--coef-entropy", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir = out_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    train_ids = np.load(args.train_track_ids_npy).astype(np.int64)
    test_df = pd.read_csv(args.teacher_test_csv, usecols=["track_id"])
    test_ids = test_df["track_id"].to_numpy(dtype=np.int64)

    teacher_oof = _align_probs_from_csv(args.teacher_oof_csv, train_ids)
    teacher_test = _align_probs_from_csv(args.teacher_test_csv, test_ids)

    spec_names = [str(args.spec1_name), str(args.spec2_name)]
    spec_oof = [
        _align_oof_from_npy(args.spec1_oof_npy, args.spec1_track_ids_npy, train_ids),
        _align_oof_from_npy(args.spec2_oof_npy, args.spec2_track_ids_npy, train_ids),
    ]
    spec_test = [
        _align_probs_from_csv(args.spec1_test_csv, test_ids),
        _align_probs_from_csv(args.spec2_test_csv, test_ids),
    ]
    if str(args.spec3_oof_npy).strip():
        if not str(args.spec3_track_ids_npy).strip() or not str(args.spec3_test_csv).strip():
            raise ValueError("spec3 requires --spec3-track-ids-npy and --spec3-test-csv")
        spec_names.append(str(args.spec3_name))
        spec_oof.append(_align_oof_from_npy(args.spec3_oof_npy, args.spec3_track_ids_npy, train_ids))
        spec_test.append(_align_probs_from_csv(args.spec3_test_csv, test_ids))

    e_oof = _entropy(teacher_oof)
    e_test = _entropy(teacher_test)
    m_oof = _margin(teacher_oof)
    m_test = _margin(teacher_test)
    l_oof = _safe_logit(teacher_oof)
    l_test = _safe_logit(teacher_test)

    a = float(args.coef_proxy)
    b = float(args.coef_margin)
    c = float(args.coef_entropy)
    t = float(args.temperature)

    score_oof_list: list[np.ndarray] = []
    score_test_list: list[np.ndarray] = []
    for so, st in zip(spec_oof, spec_test):
        proxy_oof = np.mean(np.abs(_safe_logit(so) - l_oof), axis=1)
        proxy_test = np.mean(np.abs(_safe_logit(st) - l_test), axis=1)
        d_margin_oof = _margin(so) - m_oof
        d_margin_test = _margin(st) - m_test
        d_entropy_oof = e_oof - _entropy(so)
        d_entropy_test = e_test - _entropy(st)
        score_oof = a * proxy_oof + b * d_margin_oof + c * d_entropy_oof
        score_test = a * proxy_test + b * d_margin_test + c * d_entropy_test
        score_oof_list.append(score_oof.astype(np.float32))
        score_test_list.append(score_test.astype(np.float32))

    score_oof = np.stack(score_oof_list, axis=1)  # [N,E]
    score_test = np.stack(score_test_list, axis=1)
    w_oof = _softmax(score_oof, temperature=t)    # [N,E]
    w_test = _softmax(score_test, temperature=t)

    bank_oof = np.zeros_like(teacher_oof, dtype=np.float32)
    bank_test = np.zeros_like(teacher_test, dtype=np.float32)
    for i in range(len(spec_names)):
        bank_oof += w_oof[:, [i]] * spec_oof[i]
        bank_test += w_test[:, [i]] * spec_test[i]
    bank_oof = np.clip(bank_oof, 0.0, 1.0).astype(np.float32)
    bank_test = np.clip(bank_test, 0.0, 1.0).astype(np.float32)

    np.save(art_dir / "spec_bank_oof.npy", bank_oof)
    np.save(out_dir / "train_track_ids.npy", train_ids)

    oof_df = pd.DataFrame({"track_id": train_ids})
    sub_df = pd.DataFrame({"track_id": test_ids})
    for j, cls in enumerate(CLASSES):
        oof_df[cls] = bank_oof[:, j]
        sub_df[cls] = bank_test[:, j]
    oof_df.to_csv(out_dir / "spec_bank_oof.csv", index=False)
    sub_df.to_csv(out_dir / "submission_spec_bank.csv", index=False)

    report = {
        "experts": spec_names,
        "coefs": {"proxy": a, "margin": b, "entropy": c},
        "temperature": t,
        "weights_oof_mean": {spec_names[i]: float(np.mean(w_oof[:, i])) for i in range(len(spec_names))},
        "weights_oof_p90": {spec_names[i]: float(np.quantile(w_oof[:, i], 0.90)) for i in range(len(spec_names))},
        "weights_test_mean": {spec_names[i]: float(np.mean(w_test[:, i])) for i in range(len(spec_names))},
        "weights_test_p90": {spec_names[i]: float(np.quantile(w_test[:, i], 0.90)) for i in range(len(spec_names))},
        "paths": {
            "oof_npy": str((art_dir / "spec_bank_oof.npy").resolve()),
            "oof_csv": str((out_dir / "spec_bank_oof.csv").resolve()),
            "sub_csv": str((out_dir / "submission_spec_bank.csv").resolve()),
            "train_ids": str((out_dir / "train_track_ids.npy").resolve()),
        },
    }
    (out_dir / "bank_report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print("=== SOFT BANK BUILT ===", flush=True)
    print(f"output_dir: {out_dir}", flush=True)
    print(f"experts={spec_names} temperature={t:.4f}", flush=True)
    print(str((out_dir / "bank_report.json").resolve()), flush=True)


if __name__ == "__main__":
    main()
