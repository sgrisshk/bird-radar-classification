from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="/Users/sgrisshk/Desktop/AI-task")
    p.add_argument("--cache-dir", type=str, default="/Users/sgrisshk/Desktop/AI-task/bird_radar/artifacts/cache")
    p.add_argument(
        "--output-root",
        type=str,
        default="/Users/sgrisshk/Desktop/AI-task/bird_radar/artifacts/multi_cutoff_temporal",
    )
    p.add_argument("--cutoffs", type=str, default="2023-11-01,2024-01-01,2024-02-15,2024-03-15")
    p.add_argument("--configs", type=str, default="base,reg")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-estimators", type=int, default=8000)
    p.add_argument("--drop-k-grid", type=str, default="0")
    p.add_argument("--blacklist-mode", type=str, default="none")
    p.add_argument("--extra-blacklist", type=str, default="")
    p.add_argument("--use-time-weights", action="store_true", default=True)
    p.add_argument("--no-time-weights", dest="use_time_weights", action="store_false")
    p.add_argument("--blend-top-k", type=int, default=4)
    p.add_argument("--blend-modes", type=str, default="mean,logit_mean")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--force-rerun", dest="skip_existing", action="store_false")
    return p.parse_args()


def _slug(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", s).strip("_")


def _run(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=cwd, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    py = sys.executable
    project_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root).resolve()
    runs_dir = output_root / "runs"
    blends_dir = output_root / "blends"
    output_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    blends_dir.mkdir(parents=True, exist_ok=True)

    cutoffs = [x.strip() for x in args.cutoffs.split(",") if x.strip()]
    configs = [x.strip() for x in args.configs.split(",") if x.strip()]
    if not cutoffs:
        raise ValueError("no cutoffs provided")
    if not configs:
        raise ValueError("no configs provided")

    run_results: list[dict[str, Any]] = []

    for cutoff in cutoffs:
        for cfg_name in configs:
            run_name = f"cutoff_{_slug(cutoff)}_cfg_{_slug(cfg_name)}"
            out_dir = runs_dir / run_name
            report_path = out_dir / "report.json"
            sub_path = out_dir / "sub_temporal_best.csv"

            if args.skip_existing and report_path.exists() and sub_path.exists():
                report = _load_json(report_path)
                run_results.append(
                    {
                        "run_name": run_name,
                        "cutoff": cutoff,
                        "config": cfg_name,
                        "output_dir": str(out_dir),
                        "report_path": str(report_path),
                        "submission_path": str(sub_path),
                        "temporal_macro_map_best": float(report.get("temporal_macro_map_best", 0.0)),
                        "adversarial_auc_mean": float(report.get("adversarial_auc_mean", 0.0)),
                    }
                )
                continue

            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                py,
                str(project_root / "train_temporal_lgbm.py"),
                "--data-dir",
                str(Path(args.data_dir).resolve()),
                "--cache-dir",
                str(Path(args.cache_dir).resolve()),
                "--output-dir",
                str(out_dir),
                "--temporal-cutoff-date",
                cutoff,
                "--force-config",
                cfg_name,
                "--seed",
                str(args.seed),
                "--n-estimators",
                str(args.n_estimators),
                "--drop-k-grid",
                args.drop_k_grid,
                "--blacklist-mode",
                args.blacklist_mode,
            ]
            if args.extra_blacklist.strip():
                cmd += ["--extra-blacklist", args.extra_blacklist]
            if args.use_time_weights:
                cmd += ["--use-time-weights"]
            else:
                cmd += ["--no-time-weights"]

            _run(cmd, cwd=project_root.parent)

            report = _load_json(report_path)
            run_results.append(
                {
                    "run_name": run_name,
                    "cutoff": cutoff,
                    "config": cfg_name,
                    "output_dir": str(out_dir),
                    "report_path": str(report_path),
                    "submission_path": str(sub_path),
                    "temporal_macro_map_best": float(report.get("temporal_macro_map_best", 0.0)),
                    "adversarial_auc_mean": float(report.get("adversarial_auc_mean", 0.0)),
                }
            )

    run_results.sort(key=lambda x: x["temporal_macro_map_best"], reverse=True)
    top_k = run_results[: max(1, min(args.blend_top_k, len(run_results)))]

    blend_outputs: dict[str, Any] = {}
    if len(top_k) >= 2:
        inputs_csv = ",".join([r["submission_path"] for r in top_k])
        blend_cmd = [
            py,
            str(project_root / "scripts" / "blend_submissions.py"),
            "--inputs",
            inputs_csv,
            "--output-dir",
            str(blends_dir),
            "--modes",
            args.blend_modes,
        ]
        _run(blend_cmd, cwd=project_root)
        blend_report_path = blends_dir / "blend_report.json"
        if blend_report_path.exists():
            blend_outputs = _load_json(blend_report_path)

    summary = {
        "project_root": str(project_root),
        "data_dir": str(Path(args.data_dir).resolve()),
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "output_root": str(output_root),
        "cutoffs": cutoffs,
        "configs": configs,
        "n_runs": int(len(run_results)),
        "runs_sorted": run_results,
        "top_k_for_blend": top_k,
        "blend_outputs": blend_outputs,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print(str(output_root / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
