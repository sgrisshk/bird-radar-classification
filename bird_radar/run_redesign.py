from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.redesign import export_submissions, load_yaml, run_blend, run_oof_training
from src.redesign.utils import config_hash, dump_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="bird_radar/redesign_config.yaml")
    p.add_argument("--mode", type=str, default="all", choices=["train", "blend", "submit", "all"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        fallback = Path(__file__).resolve().parent / "configs" / "redesign" / config_path.name
        if fallback.exists():
            print(f"[run_redesign] config not found at {config_path}, using {fallback}", flush=True)
            config_path = fallback
    cfg = load_yaml(str(config_path))
    set_seed(int(cfg["seed"]))

    out_dir = Path(cfg["paths"]["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_report: dict[str, object] = {
        "config_path": str(config_path.resolve()),
        "config_hash": config_hash(cfg),
        "mode": args.mode,
    }

    oof_summary = None
    blend_report = None
    sub_report = None

    if args.mode in {"train", "all"}:
        oof_summary = run_oof_training(cfg)
        pipeline_report["oof_summary_path"] = str((out_dir / "oof_summary.json").resolve())

    if args.mode in {"blend", "all"}:
        if oof_summary is None:
            with (out_dir / "oof_summary.json").open("r", encoding="utf-8") as f:
                oof_summary = json.load(f)
        blend_report = run_blend(cfg, oof_summary)
        pipeline_report["blend_report_path"] = str((out_dir / "blends" / "blend_report.json").resolve())

    if args.mode in {"submit", "all"}:
        if blend_report is None:
            with (out_dir / "blends" / "blend_report.json").open("r", encoding="utf-8") as f:
                blend_report = json.load(f)
        sub_report = export_submissions(cfg, blend_report)
        pipeline_report["submission_report_path"] = str((out_dir / "submissions" / "submission_report.json").resolve())

    dump_json(out_dir / "pipeline_report.json", pipeline_report)

    print("=== REDESIGN PIPELINE COMPLETE ===", flush=True)
    print(f"output_dir: {out_dir}", flush=True)
    if oof_summary is not None:
        print(f"oof_summary: {out_dir / 'oof_summary.json'}", flush=True)
    if blend_report is not None:
        print(f"blend_report: {out_dir / 'blends' / 'blend_report.json'}", flush=True)
    if sub_report is not None:
        print(f"submission_report: {out_dir / 'submissions' / 'submission_report.json'}", flush=True)


if __name__ == "__main__":
    main()
