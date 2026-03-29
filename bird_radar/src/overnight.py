from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.blend import blend_predictions, optimize_two_model_weight, save_blend_artifact
from src.calibrate import calibrate_and_score, save_calibration
from src.experiment_registry import Experiment, generate_experiment_queue
from src.infer import write_submission_csv
from src.scoreboard import CLASSES, Scoreboard, macro_map, per_class_ap
from src.state import OvernightState
from src.train_lgbm import run_lgbm_experiment
from src.train_seq import run_seq_experiment
from src.validate_submission import run_validation


@dataclass
class OvernightConfig:
    hours: float
    device: str
    max_submissions: int
    mode: str
    resume: bool
    dry_run: bool
    output_dir: str | None
    data_dir: str | None
    min_improvement: float
    patience_experiments: int
    force_last: bool


class OvernightRunner:
    def __init__(self, project_root: str | Path, cfg: OvernightConfig) -> None:
        self.project_root = Path(project_root).resolve()
        self.cfg = cfg
        self.yaml_cfg = self._load_yaml_config(self.project_root / "config.yaml")
        cfg_data_dir = cfg.data_dir if getattr(cfg, "data_dir", None) else None
        self.data_dir = Path(cfg_data_dir or self.yaml_cfg.get("data_dir", self.project_root.parent / "ai-cup-2026-performance")).resolve()
        self.cache_dir = (self.project_root / str(self.yaml_cfg.get("cache_dir", "artifacts/cache"))).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_output = self.project_root / "outputs" / "overnight_runs" / run_ts
        self.output_dir = Path(cfg.output_dir).resolve() if cfg.output_dir else default_output.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = self.output_dir.name
        self.state_path = self.output_dir / "state.json"
        self.scoreboard = Scoreboard(self.output_dir / "scoreboard.csv")
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir = self.output_dir / "experiments"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.blends_dir = self.output_dir / "blends"
        self.blends_dir.mkdir(parents=True, exist_ok=True)
        self.submissions_dir = self.output_dir / "submissions"
        self.submissions_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.output_dir / "summary.json"
        self.queue = generate_experiment_queue(mode=cfg.mode)
        self.state = self._load_or_init_state()
        self._unlock_budget_skipped_for_resume()
        self.train_df = pd.read_csv(self.data_dir / "train.csv", usecols=["track_id", "bird_group", "observation_id"])
        self.test_df = pd.read_csv(self.data_dir / "test.csv", usecols=["track_id"])
        self.y_true = self._one_hot(self.train_df["bird_group"])
        self.start_monotonic = time.monotonic()
        self.deadline_monotonic = self.start_monotonic + float(cfg.hours) * 3600.0
        self.no_improve_counter = 0
        self.best_blended_score = float(self.state.best_scores.get("blend", {}).get("macro_map", -1.0))

    def _load_yaml_config(self, path: Path) -> dict[str, Any]:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return dict(data)
        return {}

    def _one_hot(self, labels: pd.Series) -> np.ndarray:
        label_to_idx = {c: i for i, c in enumerate(CLASSES)}
        y = np.zeros((len(labels), len(CLASSES)), dtype=np.uint8)
        for i, label in enumerate(labels.astype(str).tolist()):
            if label not in label_to_idx:
                raise ValueError(f"Unknown class label in train.csv: {label}")
            y[i, label_to_idx[label]] = 1
        return y

    def _load_or_init_state(self) -> OvernightState:
        if self.cfg.resume and self.state_path.exists():
            state = OvernightState.load(self.state_path)
            if not state.experiment_order:
                state.experiment_order = [e.name for e in self.queue]
            return state
        state = OvernightState.create(self.run_id, str(self.output_dir), [e.name for e in self.queue])
        state.save(self.state_path)
        return state

    def _unlock_budget_skipped_for_resume(self) -> None:
        # Experiments skipped only because of time budget are retriable on the next run.
        to_delete = [
            name
            for name, payload in self.state.failed_experiments.items()
            if str(payload.get("status", "")) == "skipped_budget"
        ]
        if to_delete:
            for name in to_delete:
                self.state.failed_experiments.pop(name, None)
            self.state.save(self.state_path)

    def _now_iso(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _remaining_sec(self) -> float:
        return max(0.0, self.deadline_monotonic - time.monotonic())

    def _estimate_experiment_sec(self, exp: Experiment) -> float:
        defaults = {"lgbm": 600.0, "seq": 5400.0}
        return self.state.estimate_duration_sec(exp.model_type, defaults.get(exp.model_type, 900.0))

    def _should_skip_for_time(self, exp: Experiment) -> bool:
        remaining = self._remaining_sec()
        estimate = self._estimate_experiment_sec(exp)
        if self.cfg.force_last:
            return False
        buffer_sec = 60.0
        return (estimate + buffer_sec) > remaining

    def _record_scoreboard_row(
        self,
        experiment_name: str,
        model_type: str,
        seed: int | str,
        params: dict[str, Any],
        duration_sec: float,
        status: str,
        artifact_dir: str | Path,
        macro_val: float | None = None,
        per_class: dict[str, float] | None = None,
    ) -> None:
        self.scoreboard.append(
            {
                "timestamp": self._now_iso(),
                "experiment_name": experiment_name,
                "model_type": model_type,
                "seed": seed,
                "params_json": Scoreboard.json_dumps(params),
                "macro_map": "" if macro_val is None else float(macro_val),
                "per_class_ap_json": "" if per_class is None else Scoreboard.json_dumps(per_class),
                "duration_sec": float(duration_sec),
                "status": status,
                "artifact_dir": str(artifact_dir),
            }
        )

    def _load_pred(self, path: str | Path) -> np.ndarray:
        return np.load(Path(path)).astype(np.float32)

    def _update_best(self, model_type: str, exp_name: str, score: float, artifact_dir: str, meta: dict[str, Any]) -> bool:
        current = self.state.best_scores.get(model_type)
        current_score = float(current.get("macro_map", -1.0)) if current else -1.0
        improved = score >= (current_score + float(self.cfg.min_improvement))
        if improved:
            self.state.best_scores[model_type] = {
                "macro_map": float(score),
                "experiment_name": exp_name,
                "artifact_dir": str(artifact_dir),
                "updated_at": self._now_iso(),
                "meta": meta,
            }
            self.no_improve_counter = 0
            return True
        self.no_improve_counter += 1
        return False

    def _candidate_paths_from_state(self, model_type: str) -> dict[str, Any] | None:
        info = self.state.best_scores.get(model_type)
        if not info:
            return None
        artifact_dir = Path(info["artifact_dir"])
        if model_type == "lgbm":
            oof_path = artifact_dir / "oof_calibrated.npy"
            test_path = artifact_dir / "test_calibrated.npy"
        elif model_type == "seq":
            oof_path = artifact_dir / "oof_sequence_mean_calibrated.npy"
            test_path = artifact_dir / "test_sequence_mean_calibrated.npy"
        elif model_type == "blend":
            meta = info.get("meta", {})
            oof_path = Path(meta.get("oof_path", artifact_dir / "oof_blend.npy"))
            test_path = Path(meta.get("test_path", artifact_dir / "test_blend.npy"))
        else:
            return None
        if not oof_path.exists() or not test_path.exists():
            return None
        return {"artifact_dir": str(artifact_dir), "oof_path": str(oof_path), "test_path": str(test_path), "info": info}

    def _attempt_blend_and_submission(self, trigger_experiment: Experiment) -> None:
        best_lgbm = self._candidate_paths_from_state("lgbm")
        best_seq = self._candidate_paths_from_state("seq")
        if best_lgbm is None or best_seq is None:
            return

        p_lgbm_oof = self._load_pred(best_lgbm["oof_path"])
        p_seq_oof = self._load_pred(best_seq["oof_path"])
        p_lgbm_test = self._load_pred(best_lgbm["test_path"])
        p_seq_test = self._load_pred(best_seq["test_path"])

        opt = optimize_two_model_weight(self.y_true, p_lgbm_oof, p_seq_oof, steps=401)
        oof_blend = blend_predictions([p_lgbm_oof, p_seq_oof], np.array([opt["w0"], opt["w1"]], dtype=np.float32))
        test_blend = blend_predictions([p_lgbm_test, p_seq_test], np.array([opt["w0"], opt["w1"]], dtype=np.float32))

        calib = calibrate_and_score(self.y_true, oof_blend)
        oof_final = calib["selected_probs"]
        if calib["selected"] == "calibrated":
            logits_input_test = test_blend
            from src.calibrate import apply_temperature_scaling
            test_final = apply_temperature_scaling(logits_input_test, calib["temperatures"])
        else:
            test_final = test_blend

        macro_final = float(macro_map(self.y_true, oof_final))
        per_class_final = per_class_ap(self.y_true, oof_final)

        blend_name = f"blend_after_{trigger_experiment.name}"
        blend_dir = self.blends_dir / blend_name
        meta = {
            "timestamp": self._now_iso(),
            "blend_name": blend_name,
            "sources": {
                "lgbm": best_lgbm["info"],
                "seq": best_seq["info"],
            },
            "weights": {"lgbm": float(opt["w0"]), "seq": float(opt["w1"])},
            "oof_macro_map_raw_blend": float(opt["macro_map"]),
            "oof_macro_map_final": float(macro_final),
            "calibration": {
                "selected": calib["selected"],
                "raw_macro_map": float(calib["raw_macro_map"]),
                "calibrated_macro_map": float(calib["calibrated_macro_map"]),
                "selected_macro_map": float(calib["selected_macro_map"]),
                "temperatures": [float(x) for x in np.asarray(calib["temperatures"]).ravel()],
            },
            "per_class_ap": per_class_final,
        }
        saved = save_blend_artifact(blend_dir, oof_final, test_final, meta)
        meta["oof_path"] = saved["oof_path"]
        meta["test_path"] = saved["test_path"]
        with (blend_dir / "blend_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=True)
        save_calibration(blend_dir, calib)

        self._record_scoreboard_row(
            experiment_name=blend_name,
            model_type="blend",
            seed="na",
            params={"weights": meta["weights"], "sources": {"lgbm": best_lgbm["info"]["experiment_name"], "seq": best_seq["info"]["experiment_name"]}},
            duration_sec=0.0,
            status="ok",
            artifact_dir=blend_dir,
            macro_val=macro_final,
            per_class=per_class_final,
        )
        self.state.blend_history.append(
            {
                "timestamp": self._now_iso(),
                "name": blend_name,
                "macro_map": macro_final,
                "artifact_dir": str(blend_dir),
                "weights": meta["weights"],
            }
        )

        prev_best = float(self.state.best_scores.get("blend", {}).get("macro_map", -1.0))
        improved = macro_final >= (prev_best + float(self.cfg.min_improvement))
        if improved:
            self.state.best_scores["blend"] = {
                "macro_map": macro_final,
                "experiment_name": blend_name,
                "artifact_dir": str(blend_dir),
                "updated_at": self._now_iso(),
                "meta": meta,
            }
            self.best_blended_score = macro_final
            self.no_improve_counter = 0

            if self.state.submissions_exported < int(self.cfg.max_submissions):
                sub_idx = self.state.submissions_exported + 1
                sub_dir = self.submissions_dir / f"submission_{sub_idx:02d}_{blend_name}"
                sub_dir.mkdir(parents=True, exist_ok=True)
                sample_path = self.yaml_cfg.get("submission_sample")
                submission_csv = write_submission_csv(
                    test_csv=self.data_dir / "test.csv",
                    probs=test_final,
                    out_csv=sub_dir / "submission.csv",
                    sample_submission_csv=sample_path,
                )
                oof_csv = sub_dir / "oof_predictions.csv"
                oof_df = pd.DataFrame({"track_id": self.train_df["track_id"].to_numpy()})
                for i, c in enumerate(CLASSES):
                    oof_df[c] = oof_final[:, i]
                oof_df.to_csv(oof_csv, index=False)
                vres = run_validation(
                    test_csv=self.data_dir / "test.csv",
                    submission_csv=submission_csv,
                    report_path=sub_dir / "validation_report.txt",
                    train_csv=self.data_dir / "train.csv",
                    oof_predictions_csv=oof_csv,
                )
                with (sub_dir / "submission_meta.json").open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "blend_name": blend_name,
                            "macro_map_oof": macro_final,
                            "weights": meta["weights"],
                            "validation": vres,
                        },
                        f,
                        indent=2,
                        ensure_ascii=True,
                    )
                self.state.submissions_exported += 1
                self.state.last_successful_step = f"submission:{blend_name}"

    def _attempt_lgbm_submission(self, exp: Experiment, run_result: dict[str, Any], macro_val: float) -> None:
        if self.state.submissions_exported >= int(self.cfg.max_submissions):
            return
        spread = None
        try:
            spread = float(run_result.get("scores", {}).get("fold_macro_map_spread"))
        except Exception:
            spread = None
        if spread is not None and spread > 0.02:
            return

        sub_idx = self.state.submissions_exported + 1
        sub_dir = self.submissions_dir / f"submission_{sub_idx:02d}_{exp.name}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        sample_path = self.yaml_cfg.get("submission_sample")
        test_pred = self._load_pred(run_result["test_path"])
        submission_csv = write_submission_csv(
            test_csv=self.data_dir / "test.csv",
            probs=test_pred,
            out_csv=sub_dir / "submission.csv",
            sample_submission_csv=sample_path,
        )
        oof_pred = self._load_pred(run_result["oof_path"])
        oof_csv = sub_dir / "oof_predictions.csv"
        oof_df = pd.DataFrame({"track_id": self.train_df["track_id"].to_numpy()})
        for i, c in enumerate(CLASSES):
            oof_df[c] = oof_pred[:, i]
        oof_df.to_csv(oof_csv, index=False)
        vres = run_validation(
            test_csv=self.data_dir / "test.csv",
            submission_csv=submission_csv,
            report_path=sub_dir / "validation_report.txt",
            train_csv=self.data_dir / "train.csv",
            oof_predictions_csv=oof_csv,
        )
        with (sub_dir / "submission_meta.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_experiment": exp.name,
                    "model_type": "lgbm",
                    "macro_map_oof": float(macro_val),
                    "fold_macro_map_spread": spread,
                    "validation": vres,
                },
                f,
                indent=2,
                ensure_ascii=True,
            )
        self.state.submissions_exported += 1
        self.state.last_successful_step = f"submission:{exp.name}"

    def _cleanup_non_top_artifacts(self, top_k: int = 5) -> None:
        try:
            top = self.scoreboard.top_k(k=top_k)
        except Exception:
            return
        protected: set[str] = set()
        if not top.empty and "artifact_dir" in top.columns:
            protected.update(str(p) for p in top["artifact_dir"].dropna().astype(str).tolist())
        for key in ["lgbm", "seq", "blend"]:
            info = self.state.best_scores.get(key)
            if info and "artifact_dir" in info:
                protected.add(str(info["artifact_dir"]))
        for exp_name, info in self.state.completed_experiments.items():
            art = str(info.get("artifact_dir", ""))
            if not art or art in protected:
                continue
            p = Path(art)
            if p.exists() and self.experiments_dir in p.parents:
                try:
                    shutil.rmtree(p)
                except Exception:
                    pass
        for p in self.blends_dir.iterdir() if self.blends_dir.exists() else []:
            if p.is_dir() and str(p) not in protected:
                best_blend_dir = str(self.state.best_scores.get("blend", {}).get("artifact_dir", ""))
                if str(p) != best_blend_dir:
                    try:
                        shutil.rmtree(p)
                    except Exception:
                        pass

    def _run_one(self, exp: Experiment) -> dict[str, Any]:
        artifact_dir = self.experiments_dir / exp.name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if exp.model_type == "lgbm":
            return run_lgbm_experiment(
                project_root=self.project_root,
                data_dir=self.data_dir,
                cache_dir=self.cache_dir,
                artifact_dir=artifact_dir,
                seed=exp.seed,
                hyperparams=exp.hyperparams,
            )
        if exp.model_type == "seq":
            return run_seq_experiment(
                project_root=self.project_root,
                data_dir=self.data_dir,
                cache_dir=self.cache_dir,
                artifact_dir=artifact_dir,
                seed=exp.seed,
                device=self.cfg.device,
                hyperparams=exp.hyperparams,
                max_len=int(exp.max_len or 512),
                batch_size=int(exp.batch_size or 16),
            )
        raise ValueError(f"Unsupported model_type: {exp.model_type}")

    def _evaluate_experiment_outputs(self, exp: Experiment, run_result: dict[str, Any]) -> tuple[float, dict[str, float]]:
        oof_path = Path(run_result["oof_path"])
        if not oof_path.exists():
            raise FileNotFoundError(f"OOF predictions not found: {oof_path}")
        pred = self._load_pred(oof_path)
        if pred.shape != self.y_true.shape:
            raise ValueError(f"OOF shape mismatch for {exp.name}: {pred.shape} vs {self.y_true.shape}")
        m = float(macro_map(self.y_true, pred))
        pc = per_class_ap(self.y_true, pred)
        return m, pc

    def _write_summary(self) -> None:
        summary = {
            "run_id": self.run_id,
            "output_dir": str(self.output_dir),
            "best_scores": self.state.best_scores,
            "completed_count": len(self.state.completed_experiments),
            "failed_count": len(self.state.failed_experiments),
            "submissions_exported": int(self.state.submissions_exported),
            "remaining_sec_at_end": float(self._remaining_sec()),
            "time_stats": self.state.time_stats,
        }
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)

    def run(self) -> int:
        if self.cfg.dry_run:
            print(f"Run output: {self.output_dir}")
            print(f"Time budget (hours): {self.cfg.hours}")
            for i, exp in enumerate(self.queue, start=1):
                print(f"{i:02d}. {exp.name} [{exp.model_type}] seed={exp.seed} hp={exp.hyperparams} max_len={exp.max_len} batch={exp.batch_size}")
            return 0

        for exp in self.queue:
            if self.state.is_done(exp.name):
                continue
            if self.no_improve_counter >= int(self.cfg.patience_experiments):
                break
            if self._remaining_sec() <= 0:
                break
            if self._should_skip_for_time(exp):
                self.state.record_failed(
                    exp.name,
                    {
                        "status": "skipped_budget",
                        "timestamp": self._now_iso(),
                        "remaining_sec": self._remaining_sec(),
                        "estimated_sec": self._estimate_experiment_sec(exp),
                    },
                )
                self.state.save(self.state_path)
                break

            self.state.last_successful_step = f"starting:{exp.name}"
            self.state.save(self.state_path)
            started = time.time()
            try:
                result = self._run_one(exp)
                duration = float(result.get("duration_sec", time.time() - started))
                self.state.update_time_stats(exp.model_type, duration)

                if result.get("status") != "ok":
                    self._record_scoreboard_row(
                        experiment_name=exp.name,
                        model_type=exp.model_type,
                        seed=exp.seed,
                        params={**exp.hyperparams, "max_len": exp.max_len, "batch_size": exp.batch_size},
                        duration_sec=duration,
                        status="failed",
                        artifact_dir=result.get("artifact_dir", ""),
                    )
                    self.state.record_failed(exp.name, {**result, "timestamp": self._now_iso(), "experiment": exp.to_dict()})
                    self.state.save(self.state_path)
                    continue

                macro_val, per_class_vals = self._evaluate_experiment_outputs(exp, result)
                self._record_scoreboard_row(
                    experiment_name=exp.name,
                    model_type=exp.model_type,
                    seed=exp.seed,
                    params={**exp.hyperparams, "max_len": exp.max_len, "batch_size": exp.batch_size},
                    duration_sec=duration,
                    status="ok",
                    artifact_dir=result["artifact_dir"],
                    macro_val=macro_val,
                    per_class=per_class_vals,
                )
                self.state.record_completed(
                    exp.name,
                    {
                        **result,
                        "timestamp": self._now_iso(),
                        "experiment": exp.to_dict(),
                        "macro_map": macro_val,
                        "per_class_ap": per_class_vals,
                    },
                )
                improved = self._update_best(exp.model_type, exp.name, macro_val, str(result["artifact_dir"]), {"result": result})
                if exp.model_type == "lgbm" and improved:
                    self._attempt_lgbm_submission(exp, result, macro_val)
                if exp.model_type == "seq":
                    self._attempt_blend_and_submission(exp)
                self.state.save(self.state_path)
                self._cleanup_non_top_artifacts(top_k=5)
            except Exception as e:
                duration = time.time() - started
                tb = traceback.format_exc()
                fail_dir = self.experiments_dir / exp.name
                fail_dir.mkdir(parents=True, exist_ok=True)
                (fail_dir / "failure_traceback.txt").write_text(tb, encoding="utf-8")
                self._record_scoreboard_row(
                    experiment_name=exp.name,
                    model_type=exp.model_type,
                    seed=exp.seed,
                    params={**exp.hyperparams, "max_len": exp.max_len, "batch_size": exp.batch_size},
                    duration_sec=duration,
                    status="failed",
                    artifact_dir=fail_dir,
                )
                self.state.update_time_stats(exp.model_type, duration)
                self.state.record_failed(
                    exp.name,
                    {
                        "status": "failed_exception",
                        "timestamp": self._now_iso(),
                        "error": f"{type(e).__name__}: {e}",
                        "traceback_path": str(fail_dir / "failure_traceback.txt"),
                        "experiment": exp.to_dict(),
                        "artifact_dir": str(fail_dir),
                    },
                )
                self.state.save(self.state_path)
                continue

        self._write_summary()
        best_by_type = self.scoreboard.best_by_type()
        print("=== OVERNIGHT SUMMARY ===")
        print(f"output_dir: {self.output_dir}")
        print(f"completed: {len(self.state.completed_experiments)}")
        print(f"failed: {len(self.state.failed_experiments)}")
        print(f"submissions_exported: {self.state.submissions_exported}")
        for mt in ["lgbm", "seq", "blend"]:
            info = self.state.best_scores.get(mt)
            if info:
                print(f"best_{mt}: {info.get('macro_map')} ({info.get('experiment_name')})")
        print(f"scoreboard_csv: {self.scoreboard.csv_path}")
        print(f"summary_json: {self.summary_path}")
        return 0
