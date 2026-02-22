import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


class OutputManager:
    def __init__(self, outputs_root: Path, exp_name: str = "camvid_deeplabv3p") -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = outputs_root / f"{exp_name}_{ts}"
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.vis_dir = self.run_dir / "visualizations"
        self.log_dir = self.run_dir / "logs"

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_csv = self.log_dir / "metrics.csv"
        self.per_class_metrics_csv = self.log_dir / "per_class_metrics.csv"
        self.config_json = self.run_dir / "config.json"

    def save_config(self, cfg: Any) -> None:
        self.config_json.write_text(
            json.dumps(asdict(cfg), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def init_metrics(self) -> None:
        if self.metrics_csv.exists():
            pass
        else:
            with self.metrics_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "epoch",
                        "train_loss",
                        "val_loss",
                        "val_miou",
                        "val_bf1",
                        "time_sec",
                        "train_auto_contrast_enable",
                        "train_low_light_preprocess_enable",
                        "low_light_gamma",
                        "low_light_brightness_gain",
                    ]
                )

        if not self.per_class_metrics_csv.exists():
            with self.per_class_metrics_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "class_id", "class_name", "iou", "precision", "recall"])

    def append_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_miou: float,
        val_bf1: float,
        dt: float,
        train_auto_contrast_enable: bool,
        train_low_light_preprocess_enable: bool,
        low_light_gamma: float,
        low_light_brightness_gain: float,
    ) -> None:
        with self.metrics_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_miou:.6f}",
                    f"{val_bf1:.6f}",
                    f"{dt:.2f}",
                    str(train_auto_contrast_enable),
                    str(train_low_light_preprocess_enable),
                    f"{low_light_gamma:.4f}",
                    f"{low_light_brightness_gain:.4f}",
                ]
            )

    def append_per_class_metrics(
        self,
        epoch: int,
        class_id: int,
        class_name: str,
        iou: float,
        precision: float,
        recall: float,
    ) -> None:
        with self.per_class_metrics_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    epoch,
                    class_id,
                    class_name,
                    f"{iou:.6f}",
                    f"{precision:.6f}",
                    f"{recall:.6f}",
                ]
            )
