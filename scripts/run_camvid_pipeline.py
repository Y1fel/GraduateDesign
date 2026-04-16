import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import MobileTrainConfig, TrainConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full CamVid pipeline: best teacher training, student baseline, and distillation."
    )
    parser.add_argument(
        "--teacher-preset",
        type=str,
        default="full_preprocess_classweights",
        choices=[
            "baseline",
            "large",
            "large_v3",
            "full_preprocess",
            "full_preprocess_classweights",
        ],
        help="Teacher preset to use on CamVid. Default is the current best teacher training recipe.",
    )
    parser.add_argument("--camvid-root", type=Path, default=None, help="Override CamVid dataset root.")
    parser.add_argument("--outputs-root", type=Path, default=None, help="Override outputs root.")
    parser.add_argument("--teacher-epochs", type=int, default=None, help="Override teacher epochs.")
    parser.add_argument("--student-epochs", type=int, default=None, help="Override student epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for all stages.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override num_workers for all stages.")
    parser.add_argument("--teacher-lr", type=float, default=None, help="Override teacher initial lr.")
    parser.add_argument("--student-lr", type=float, default=None, help="Override student initial lr.")
    parser.add_argument(
        "--distill-type",
        type=str,
        default="cwd",
        help="Distillation type for the student stage, e.g. cwd, kl, feature_cwd_residual.",
    )
    parser.add_argument("--distill-temperature", type=float, default=4.0, help="Distillation temperature.")
    parser.add_argument("--distill-loss-weight", type=float, default=0.4, help="Main distillation loss weight.")
    parser.add_argument("--distill-aux-weight", type=float, default=0.4, help="Aux distillation loss weight.")
    parser.add_argument("--distill-start-epoch", type=int, default=10, help="First epoch to enable distillation.")
    parser.add_argument("--distill-ramp-epochs", type=int, default=10, help="Distillation ramp-up epochs.")
    parser.add_argument(
        "--student-pretrained",
        action="store_true",
        help="Use ImageNet-pretrained MobileNetV2 for the student. Default is scratch.",
    )
    parser.add_argument(
        "--student-no-class-weights",
        action="store_true",
        help="Disable class weights for student baseline and distillation stages.",
    )
    parser.add_argument("--teacher-ckpt", type=Path, default=None, help="Reuse an existing teacher checkpoint.")
    parser.add_argument("--skip-teacher", action="store_true", help="Skip teacher training.")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip student baseline training.")
    parser.add_argument("--skip-distill", action="store_true", help="Skip student distillation training.")
    parser.add_argument("--summary-path", type=Path, default=None, help="Optional JSON summary output path.")
    return parser


def _apply_common_overrides(cfg, args: argparse.Namespace) -> None:
    if args.camvid_root is not None:
        cfg.camvid_root = Path(args.camvid_root)
        cfg.data_root = Path(args.camvid_root)
    if args.outputs_root is not None:
        cfg.outputs_root = Path(args.outputs_root)
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        cfg.num_workers = int(args.num_workers)


def build_teacher_cfg(args: argparse.Namespace) -> TrainConfig:
    import train_camvid

    cfg = TrainConfig()
    train_camvid.apply_camvid_preset(cfg, args.teacher_preset)
    cfg.exp_tag = f"pipeline_teacher_{args.teacher_preset}"
    _apply_common_overrides(cfg, args)
    if args.teacher_epochs is not None:
        cfg.epochs = int(args.teacher_epochs)
    if args.teacher_lr is not None:
        cfg.lr_0 = float(args.teacher_lr)
    return cfg


def build_student_cfg(
    args: argparse.Namespace,
    *,
    use_distillation: bool,
    teacher_ckpt: Path | None = None,
) -> MobileTrainConfig:
    cfg = MobileTrainConfig()
    cfg.dataset_name = "camvid"
    cfg.backbone_pretrained = bool(args.student_pretrained)
    cfg.output_stride = 16
    cfg.decoder_upsample_mode = "bilinear"
    cfg.use_aux_loss = False
    cfg.loss_mode = "ce"
    cfg.use_class_weights = not bool(args.student_no_class_weights)
    cfg.use_rare_class_sampler = False

    cfg.train_multi_scale_min = 0.75
    cfg.train_multi_scale_max = 1.5
    cfg.hflip_prob = 0.5
    cfg.color_jitter_prob = 0.5
    cfg.color_jitter_brightness = 0.2
    cfg.color_jitter_contrast = 0.2
    cfg.color_jitter_saturation = 0.2
    cfg.gaussian_blur_prob = 0.2

    cfg.use_distillation = bool(use_distillation)
    if use_distillation:
        if teacher_ckpt is None:
            raise ValueError("teacher_ckpt is required when use_distillation=True")
        cfg.distill_teacher_ckpt = Path(teacher_ckpt)
        cfg.distill_teacher_arch = "resnet"
        cfg.distill_teacher_backbone_name = "rsnet-100"
        cfg.distill_teacher_backbone_pretrained = False
        cfg.distill_teacher_output_stride = 8
        cfg.distill_type = str(args.distill_type)
        cfg.distill_temperature = float(args.distill_temperature)
        cfg.distill_loss_weight = float(args.distill_loss_weight)
        cfg.distill_aux_weight = float(args.distill_aux_weight)
        cfg.distill_start_epoch = int(args.distill_start_epoch)
        cfg.distill_ramp_epochs = int(args.distill_ramp_epochs)
        cfg.exp_tag = f"pipeline_student_{str(args.distill_type).lower()}"
    else:
        cfg.exp_tag = "pipeline_student_baseline"

    _apply_common_overrides(cfg, args)
    if args.student_epochs is not None:
        cfg.epochs = int(args.student_epochs)
    if args.student_lr is not None:
        cfg.lr_0 = float(args.student_lr)
    return cfg


def build_summary_path(args: argparse.Namespace, outputs_root: Path) -> Path:
    if args.summary_path is not None:
        return Path(args.summary_path)
    ts = time.strftime("%Y%m%d_%H%M%S")
    return outputs_root / f"camvid_pipeline_summary_{ts}.json"


def main() -> None:
    args = build_parser().parse_args()

    import train as teacher_train
    import train_mobile as student_train

    teacher_ckpt: Path | None = None
    teacher_run_dir: Path | None = None
    baseline_run_dir: Path | None = None
    distill_run_dir: Path | None = None

    if args.skip_teacher:
        if args.teacher_ckpt is None:
            raise ValueError("--skip-teacher requires --teacher-ckpt")
        teacher_ckpt = Path(args.teacher_ckpt)
        if not teacher_ckpt.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt}")
        print(f"[INFO] Reusing teacher checkpoint: {teacher_ckpt}")
        outputs_root = Path(args.outputs_root) if args.outputs_root is not None else PROJECT_ROOT / "outputs"
    else:
        teacher_cfg = build_teacher_cfg(args)
        outputs_root = Path(teacher_cfg.outputs_root)
        print("[PIPELINE] Stage 1/3: train CamVid teacher")
        teacher_run_dir = teacher_train.run_training(teacher_cfg)
        teacher_ckpt = teacher_run_dir / "checkpoints" / "best.pth"
        if not teacher_ckpt.exists():
            raise FileNotFoundError(f"Teacher best checkpoint not found after training: {teacher_ckpt}")

    if not args.skip_baseline:
        baseline_cfg = build_student_cfg(args, use_distillation=False)
        print("[PIPELINE] Stage 2/3: train CamVid student baseline")
        baseline_run_dir = student_train.run_training(baseline_cfg)

    if not args.skip_distill:
        if teacher_ckpt is None:
            raise RuntimeError("Teacher checkpoint is required for distillation stage")
        distill_cfg = build_student_cfg(args, use_distillation=True, teacher_ckpt=teacher_ckpt)
        print("[PIPELINE] Stage 3/3: train CamVid student with distillation")
        distill_run_dir = student_train.run_training(distill_cfg)

    summary = {
        "teacher_preset": args.teacher_preset,
        "teacher_ckpt": str(teacher_ckpt) if teacher_ckpt is not None else None,
        "teacher_run_dir": str(teacher_run_dir) if teacher_run_dir is not None else None,
        "student_baseline_run_dir": str(baseline_run_dir) if baseline_run_dir is not None else None,
        "student_distill_run_dir": str(distill_run_dir) if distill_run_dir is not None else None,
        "distill_type": str(args.distill_type),
        "camvid_root": str(args.camvid_root) if args.camvid_root is not None else None,
    }
    summary_path = build_summary_path(args, outputs_root)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[DONE] CamVid pipeline finished.")
    print(f"[INFO] summary={summary_path}")
    if teacher_ckpt is not None:
        print(f"[INFO] teacher_ckpt={teacher_ckpt}")
    if baseline_run_dir is not None:
        print(f"[INFO] student_baseline_run_dir={baseline_run_dir}")
    if distill_run_dir is not None:
        print(f"[INFO] student_distill_run_dir={distill_run_dir}")


if __name__ == "__main__":
    main()
