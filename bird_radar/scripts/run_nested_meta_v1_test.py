#!/usr/bin/env python3
"""Retrain nested meta on full train, predict test. Uses same logic as run_nested_meta_v1.py"""
from __future__ import annotations
import argparse, ast, json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from shapely import wkb
from sklearn.metrics import average_precision_score
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from config import CLASSES

CLASS_ORDER = ["Birds of Prey","Clutter","Cormorants","Ducks","Geese","Gulls","Pigeons","Songbirds","Waders"]

def parse_ewkb(hex_str):
    geom = wkb.loads(str(hex_str), hex=True)
    coords = np.asarray(geom.coords, dtype=np.float64)
    x, y = coords[:,0], coords[:,1]
    z = coords[:,2] if coords.shape[1] >= 3 else np.zeros_like(x)
    return x, y, z

def _sp(a, q): return float(np.percentile(a, q)) if len(a) else 0.0

def extract_features(row):
    try:
        x, y, z = parse_ewkb(row["trajectory"])
    except:
        return np.zeros(52, dtype=np.float32)
    n = len(x)
    if n < 3: return np.zeros(52, dtype=np.float32)
    dx, dy, dz = np.diff(x), np.diff(y), np.diff(z)
    step2 = np.sqrt(dx**2 + dy**2)
    step3 = np.sqrt(dx**2 + dy**2 + dz**2)
    disp = float(np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2))
    angles = []
    for i in range(len(dx)-1):
        v1, v2 = np.array([dx[i],dy[i]]), np.array([dx[i+1],dy[i+1]])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-9 and n2 > 1e-9:
            angles.append(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2),-1,1))))
    angles = np.array(angles) if angles else np.zeros(1)
    f = [
        float(n), float(np.sum(step2)), float(np.sum(step3)), disp,
        float(np.sum(step2))/(disp+1e-9),
        float(np.mean(z)), float(np.std(z)), _sp(z,10), _sp(z,50), _sp(z,90),
        float(np.min(z)), float(np.max(z)), float(np.max(z)-np.min(z)),
        float(np.mean(step2)), float(np.std(step2)), _sp(step2,10), _sp(step2,90),
        float(np.mean(np.abs(dz))), float(np.std(dz)),
        float(np.mean(angles)), float(np.std(angles)), _sp(angles,90),
        float((angles > 90).mean()),
        float(row.get("airspeed", 0) or 0),
        float(row.get("min_z", 0) or 0),
        float(row.get("max_z", 0) or 0),
        {"Small bird": 1.0, "Large bird": 2.0, "Flock": 3.0}.get(str(row.get("radar_bird_size", "")), 1.0),
    ]
    f += [0.0] * (52 - len(f))
    return np.array(f[:52], dtype=np.float32)

def build_meta(bp, w, c, idx_w, idx_c, idx_g, idx_s):
    return np.concatenate([bp, w[:,None], c[:,None]], axis=1).astype(np.float32)

def norm(p):
    p = np.clip(p, 0, None)
    return p / (p.sum(axis=1, keepdims=True) + 1e-9)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-path", default="train.csv")
    p.add_argument("--test-path", default="test.csv")
    p.add_argument("--teacher-oof-csv", default="bird_radar/artifacts/oof_csv_for_per_class/oof_teacher_fr_complete.csv")
    p.add_argument("--tcn-oof-npy", default="bird_radar/artifacts/tcn_trajectory_forward_2stream_distill_full_seed777_cpu/artifacts/tcn_oof.npy")
    p.add_argument("--tcn-train-ids", default="bird_radar/artifacts/tcn_trajectory_forward_2stream_distill_full_seed777_cpu/train_track_ids.npy")
    p.add_argument("--tcn-test-npy", default="bird_radar/artifacts/tcn_trajectory_forward_2stream_distill_full_seed777_cpu/artifacts/tcn_test.npy")
    p.add_argument("--teacher-test-csv", default="bird_radar/artifacts/temporal_model0_reg_drop10_none_reverse_complete/sub_teacher_fr.csv")
    p.add_argument("--output-dir", default="bird_radar/artifacts/nested_meta_v1_with_test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--iterations", type=int, default=600)
    p.add_argument("--thread-count", type=int, default=-1)
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_path)
    test_df  = pd.read_csv(args.test_path)
    labels   = train_df["bird_group"].values
    train_ids = train_df["track_id"].values
    test_ids  = test_df["track_id"].values

    # Base probs train
    t_oof = pd.read_csv(args.teacher_oof_csv).set_index("track_id")[CLASS_ORDER]
    tcn_ids = np.load(args.tcn_train_ids, allow_pickle=True)
    tcn_oof = np.load(args.tcn_oof_npy)
    tcn_df = pd.DataFrame(tcn_oof, index=tcn_ids, columns=CLASS_ORDER)

    t_tr = norm(t_oof.reindex(train_ids).fillna(1/9).values)
    tcn_tr = norm(tcn_df.reindex(train_ids).fillna(1/9).values)
    base_train = norm(0.75*t_tr + 0.25*tcn_tr)

    # Base probs test
    t_test = pd.read_csv(args.teacher_test_csv).set_index("track_id")[CLASS_ORDER]
    # TCN test
    tcn_test_path = Path(args.tcn_test_npy)
    if tcn_test_path.exists():
        tcn_test_raw = np.load(str(tcn_test_path))
        tcn_test_ids_path = Path(args.tcn_train_ids).parent / "test_track_ids.npy"
        tcn_test_ids = np.load(str(tcn_test_ids_path), allow_pickle=True)
        tcn_test_df = pd.DataFrame(tcn_test_raw, index=tcn_test_ids, columns=CLASS_ORDER)
    else:
        print("TCN test not found, using teacher only for test")
        tcn_test_df = t_test.copy()

    t_te  = norm(t_test.reindex(test_ids).fillna(1/9).values)
    tcn_te = norm(tcn_test_df.reindex(test_ids).fillna(1/9).values)
    base_test = norm(0.75*t_te + 0.25*tcn_te)

    idx_w = CLASS_ORDER.index("Waders")
    idx_c = CLASS_ORDER.index("Cormorants")
    idx_g = CLASS_ORDER.index("Geese")
    idx_s = CLASS_ORDER.index("Songbirds")

    print("Extracting train features...", flush=True)
    X_train = np.vstack([extract_features(row) for _, row in train_df.iterrows()])
    print("Extracting test features...", flush=True)
    X_test  = np.vstack([extract_features(row) for _, row in test_df.iterrows()])
    print(f"X_train={X_train.shape}  X_test={X_test.shape}", flush=True)

    def train_detector(X, y_bin, tag):
        m = CatBoostClassifier(iterations=args.iterations, depth=6, learning_rate=0.05,
                               od_wait=50, eval_metric="AUC", random_seed=args.seed,
                               verbose=0, allow_writing_files=False, thread_count=args.thread_count)
        m.fit(X, y_bin)
        print(f"  {tag} trained", flush=True)
        return m

    # Train full detectors on all train data
    print("Training Wader detector on full train...", flush=True)
    y_w = (labels == "Waders").astype(int)
    y_c = (labels == "Cormorants").astype(int)
    det_w = train_detector(X_train, y_w, "Wader")
    det_c = train_detector(X_train, y_c, "Cormorant")

    w_test = det_w.predict_proba(X_test)[:,1].astype(np.float32)
    c_test = det_c.predict_proba(X_test)[:,1].astype(np.float32)
    w_train = det_w.predict_proba(X_train)[:,1].astype(np.float32)
    c_train = det_c.predict_proba(X_train)[:,1].astype(np.float32)

    # Train meta on full train, predict test
    print("Training meta model on full train...", flush=True)
    meta_train = build_meta(base_train, w_train, c_train, idx_w, idx_c, idx_g, idx_s)
    meta_test  = build_meta(base_test,  w_test,  c_test,  idx_w, idx_c, idx_g, idx_s)

    meta_test_probs = np.zeros((len(test_ids), 9), dtype=np.float32)
    for ci, cls in enumerate(CLASS_ORDER):
        y_bin = (labels == cls).astype(int)
        if y_bin.sum() == 0: continue
        m = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05,
                               od_wait=30, random_seed=args.seed,
                               verbose=0, allow_writing_files=False, thread_count=args.thread_count)
        m.fit(meta_train, y_bin)
        meta_test_probs[:, ci] = m.predict_proba(meta_test)[:,1]
        print(f"  {cls} done", flush=True)

    meta_test_probs = norm(meta_test_probs)

    # Save submission
    sub = pd.DataFrame(meta_test_probs, columns=CLASS_ORDER)
    sub.insert(0, "track_id", test_ids)
    sub_path = out_dir / "sub_nested_meta_test.csv"
    sub.to_csv(sub_path, index=False)
    np.save(out_dir / "meta_test_probs.npy", meta_test_probs)
    print(f"\nSaved: {sub_path}", flush=True)

    # Verify OOF score using existing meta_oof_probs
    oof_path = Path("bird_radar/artifacts/advanced_features_v1_altcv_patch/meta_oof_probs.npy")
    if oof_path.exists():
        spec_ids = np.load("bird_radar/artifacts/spec_ds_v1/train_track_ids.npy", allow_pickle=True)
        id2idx = {tid: i for i, tid in enumerate(spec_ids)}
        oof_raw = np.load(str(oof_path))
        oof = norm(np.array([oof_raw[id2idx[tid]] if tid in id2idx else np.zeros(9) for tid in train_ids]))
        covered_mask = np.array([tid in set(t_oof.index) for tid in train_ids])
        y_cov = labels[covered_mask]
        p_cov = oof[covered_mask]
        ap = np.mean([average_precision_score((y_cov==c).astype(int), p_cov[:,i])
                      for i,c in enumerate(CLASS_ORDER) if (y_cov==c).sum()>0])
        print(f"OOF meta macro AP (covered): {ap:.6f}", flush=True)

if __name__ == "__main__":
    main()
