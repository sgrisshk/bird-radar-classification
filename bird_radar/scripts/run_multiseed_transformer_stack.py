from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
import sys
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.redesign.utils import load_yaml


def _parse_seeds(raw: str) -> list[int]:
    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("no seeds parsed")
    return out


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    return out


def _run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _build_seed_cfg(
    base_cfg: dict[str, Any],
    seed: int,
    out_dir: Path,
    hard_mining_enabled: bool,
    hard_mining_start_epoch: int,
    hard_mining_fraction: float,
    hard_mining_mix: float,
    hard_mining_boost: float,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = int(seed)
    cfg.setdefault("model", {})
    cfg["model"]["seeds"] = [int(seed)]
    cfg.setdefault("paths", {})
    cfg["paths"]["output_dir"] = str(out_dir.resolve())
    cfg.setdefault("train", {})
    cfg["train"]["hard_mining_enabled"] = bool(hard_mining_enabled)
    cfg["train"]["hard_mining_start_epoch"] = int(hard_mining_start_epoch)
    cfg["train"]["hard_mining_fraction"] = float(hard_mining_fraction)
    cfg["train"]["hard_mining_mix"] = float(hard_mining_mix)
    cfg["train"]["hard_mining_boost"] = float(hard_mining_boost)
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", required=True)
    p.add_argument("--seeds", default="42,777,1337", help="comma-separated seeds")
    p.add_argument("--python-bin", default=".venv/bin/python")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--extra-run-dir", action="append", default=[], help="existing run dir(s) to include in stack")
    p.add_argument("--enable-hard-mining", action="store_true", default=False)
    p.add_argument("--hard-mining-start-epoch", type=int, default=1)
    p.add_argument("--hard-mining-fraction", type=float, default=0.2)
    p.add_argument("--hard-mining-mix", type=float, default=0.4)
    p.add_argument("--hard-mining-boost", type=float, default=3.0)

    p.add_argument("--train-csv", default="train.csv")
    p.add_argument("--sample-submission", default="sample_submission.csv")
    p.add_argument(
        "--teacher-oof-csv",
        default="bird_radar/artifacts/oof_csv_for_per_class/oof_base_forward_complete.csv",
    )
    p.add_argument(
        "--teacher-test-csv",
        default="bird_radar/artifacts/temporal_model0_reg_drop10_none_complete/sub_temporal_best.csv",
    )
    p.add_argument("--stack-output-dir", default="bird_radar/artifacts/stack_teacher_3xtr_ridge")
    p.add_argument("--stack-out-name", default="submission_stacked_ridge.csv")
    p.add_argument(
        "--stack-learner",
        default="ridge",
        choices=["ridge", "logreg", "convex_anchor", "teacher_passthrough"],
    )
    p.add_argument("--stack-split-mode", default="forward", choices=["forward", "stratified_group"])
    p.add_argument("--stack-ridge-alpha", type=float, default=1.0)
    p.add_argument("--stack-n-splits", type=int, default=5)
    p.add_argument(
        "--stack-uncovered-fill",
        default="teacher_fallback",
        choices=["teacher_fallback", "meta_backcast", "covered_mean", "zero"],
    )
    p.add_argument(
        "--stack-select-metric",
        default="stack_gain_vs_teacher_fallback",
        choices=[
            "stack_gain_vs_teacher_fallback",
            "stack_gain_vs_teacher_full_filled",
            "stack_gain_vs_teacher_covered",
        ],
    )
    p.add_argument("--stack-min-teacher", type=float, default=0.95)
    p.add_argument("--stack-grid-step", type=float, default=0.01)
    p.add_argument(
        "--stack-min-teacher-scan",
        default="",
        help="comma-separated min_teacher values for convex_anchor (e.g. 0.95,0.96,0.97,0.98)",
    )

    # Optional overrides for transformer files inside each run/artifacts.
    p.add_argument("--oof-npy-name", default=None)
    p.add_argument("--test-npy-name", default=None)
    p.add_argument("--run-arrays-are-logits", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)

    base_cfg_path = Path(args.base_config).resolve()
    base_cfg = load_yaml(str(base_cfg_path))

    base_out = Path(base_cfg["paths"]["output_dir"]).resolve()
    cfg_gen_dir = Path(args.stack_output_dir).resolve() / "generated_configs"
    cfg_gen_dir.mkdir(parents=True, exist_ok=True)

    import os

    env = os.environ.copy()
    # Keep existing environment and force a writable MPL cache.
    env.setdefault("MPLCONFIGDIR", "/tmp/.mpl")

    run_dirs: list[Path] = [Path(p).resolve() for p in args.extra_run_dir]
    for s in seeds:
        run_out = base_out.parent / f"{base_out.name}_seed{s}"
        run_dirs.append(run_out)
        cfg_path = cfg_gen_dir / f"{base_out.name}_seed{s}.yaml"
        cfg_s = _build_seed_cfg(
            base_cfg=base_cfg,
            seed=s,
            out_dir=run_out,
            hard_mining_enabled=bool(args.enable_hard_mining),
            hard_mining_start_epoch=int(args.hard_mining_start_epoch),
            hard_mining_fraction=float(args.hard_mining_fraction),
            hard_mining_mix=float(args.hard_mining_mix),
            hard_mining_boost=float(args.hard_mining_boost),
        )
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_s, f, sort_keys=False, allow_unicode=False)

        summary_path = run_out / "oof_summary.json"
        if args.skip_existing and summary_path.exists():
            print(f"[skip] existing run found: {run_out}", flush=True)
            continue

        _run_cmd(
            [
                args.python_bin,
                "bird_radar/run_redesign.py",
                "--config",
                str(cfg_path),
                "--mode",
                "train",
            ],
            env=env,
        )

    base_stack_cmd = [
        args.python_bin,
        "bird_radar/scripts/stack_teacher_transformers_ridge.py",
        "--train-csv",
        str(Path(args.train_csv).resolve()),
        "--sample-submission",
        str(Path(args.sample_submission).resolve()),
        "--teacher-oof-csv",
        str(Path(args.teacher_oof_csv).resolve()),
        "--teacher-test-csv",
        str(Path(args.teacher_test_csv).resolve()),
        "--learner",
        str(args.stack_learner),
        "--split-mode",
        str(args.stack_split_mode),
        "--ridge-alpha",
        str(float(args.stack_ridge_alpha)),
        "--n-splits",
        str(int(args.stack_n_splits)),
        "--uncovered-fill",
        str(args.stack_uncovered_fill),
    ]
    for rd in run_dirs:
        base_stack_cmd.extend(["--run-dir", str(rd.resolve())])
    if args.oof_npy_name:
        base_stack_cmd.extend(["--oof-npy-name", args.oof_npy_name])
    if args.test_npy_name:
        base_stack_cmd.extend(["--test-npy-name", args.test_npy_name])
    if args.run_arrays_are_logits:
        base_stack_cmd.append("--run-arrays-are-logits")

    if str(args.stack_learner) == "convex_anchor":
        scan_vals = _parse_float_list(str(args.stack_min_teacher_scan))
        if not scan_vals:
            scan_vals = [float(args.stack_min_teacher)]

        best: dict[str, Any] | None = None
        base_out_dir = Path(args.stack_output_dir).resolve()
        for mt in scan_vals:
            suffix = f"mt{str(mt).replace('.', 'p')}_gs{str(args.stack_grid_step).replace('.', 'p')}"
            out_dir = base_out_dir if len(scan_vals) == 1 else (base_out_dir / suffix)
            out_name = args.stack_out_name if len(scan_vals) == 1 else f"{Path(args.stack_out_name).stem}_{suffix}{Path(args.stack_out_name).suffix}"

            stack_cmd = list(base_stack_cmd)
            stack_cmd.extend(
                [
                    "--min-teacher",
                    str(float(mt)),
                    "--grid-step",
                    str(float(args.stack_grid_step)),
                    "--output-dir",
                    str(out_dir),
                    "--out-name",
                    out_name,
                ]
            )
            _run_cmd(stack_cmd, env=env)

            report_path = out_dir / "stack_report.json"
            if not report_path.exists():
                print(f"[warn] stack report missing: {report_path}", flush=True)
                continue
            with report_path.open("r", encoding="utf-8") as f:
                rep = json.load(f)
            metric_name = str(args.stack_select_metric)
            gain = float(rep.get(metric_name, -1e9))
            row = {
                "min_teacher": float(mt),
                "selected_metric": metric_name,
                "selected_gain": gain,
                "gain_fallback": float(rep.get("stack_gain_vs_teacher_fallback", 0.0)),
                "gain_full_filled": float(rep.get("stack_gain_vs_teacher_full_filled", 0.0)),
                "gain_covered": float(rep.get("stack_gain_vs_teacher_covered", 0.0)),
                "report_path": str(report_path),
                "submission_path": str(rep.get("submission_path", "")),
            }
            print(
                f"[scan] min_teacher={mt:.4f} {metric_name}={gain:.6f} "
                f"(fallback={row['gain_fallback']:.6f}, full={row['gain_full_filled']:.6f}) "
                f"submission={row['submission_path']}",
                flush=True,
            )
            if best is None or row["selected_gain"] > float(best["selected_gain"]):
                best = row

        if best is not None and len(scan_vals) > 1:
            best_json_path = base_out_dir / "best_convex_scan.json"
            best_json_path.parent.mkdir(parents=True, exist_ok=True)
            with best_json_path.open("w", encoding="utf-8") as f:
                json.dump(best, f, ensure_ascii=True, indent=2)
            print(
                f"[best] min_teacher={best['min_teacher']:.4f} "
                f"{best['selected_metric']}={best['selected_gain']:.6f} "
                f"(fallback={best['gain_fallback']:.6f}, full={best['gain_full_filled']:.6f})",
                flush=True,
            )
            print(f"[best] submission={best['submission_path']}", flush=True)
            print(f"[best] report={best['report_path']}", flush=True)
            print(f"[best] summary={best_json_path}", flush=True)
    else:
        stack_cmd = list(base_stack_cmd)
        stack_cmd.extend(
            [
                "--output-dir",
                str(Path(args.stack_output_dir).resolve()),
                "--out-name",
                args.stack_out_name,
            ]
        )
        _run_cmd(stack_cmd, env=env)

    print("=== MULTISEED TRAIN+STACK COMPLETE ===", flush=True)
    print(f"stack_dir={Path(args.stack_output_dir).resolve()}", flush=True)


if __name__ == "__main__":
    main()
