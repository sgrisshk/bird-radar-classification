from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.overnight import OvernightConfig, OvernightRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-submissions", type=int, default=2)
    parser.add_argument("--mode", type=str, default="all", choices=["lgbm", "seq", "all"])
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--min-improvement", type=float, default=0.002)
    parser.add_argument("--patience-experiments", type=int, default=6)
    parser.add_argument("--force-last", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = OvernightConfig(
        hours=float(args.hours),
        device=str(args.device),
        max_submissions=int(args.max_submissions),
        mode=str(args.mode),
        resume=bool(args.resume),
        dry_run=bool(args.dry_run),
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        min_improvement=float(args.min_improvement),
        patience_experiments=int(args.patience_experiments),
        force_last=bool(args.force_last),
    )
    runner = OvernightRunner(project_root=Path(__file__).resolve().parent, cfg=cfg)
    return int(runner.run())


if __name__ == "__main__":
    sys.exit(main())
