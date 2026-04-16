import argparse
from pathlib import Path

from config.config import TrainConfig
if __package__:
    from . import train as teacher_train
else:
    import train as teacher_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train teacher models on CamVid with fixed paper presets.")
    parser.add_argument(
        "--preset",
        type=str,
        default="large",
        choices=[
            "baseline",
            "large",
            "large_v3",
            "full_preprocess",
            "full_preprocess_classweights",
        ],
        help="CamVid experiment preset.",
    )
    parser.add_argument("--camvid-root", type=Path, default=None, help="Override CamVid dataset root.")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--lr", type=float, default=None, help="Override initial learning rate.")
    return parser


def apply_camvid_preset(cfg: TrainConfig, preset: str) -> None:
    preset = str(preset).lower()

    cfg.dataset_name = "camvid"
    cfg.backbone_pretrained = True
    cfg.output_stride = 8
    cfg.decoder_upsample_mode = "bilinear"
    cfg.use_aux_loss = False
    cfg.loss_mode = "ce"
    cfg.use_class_weights = False
    cfg.use_rare_class_sampler = False
    cfg.segmentation_head = "hybrid"
    cfg.hybrid_variant = "large"
    cfg.hybrid_use_strip = False

    cfg.train_multi_scale_min = 0.75
    cfg.train_multi_scale_max = 1.5
    cfg.hflip_prob = 0.5
    cfg.color_jitter_prob = 0.0
    cfg.color_jitter_brightness = 0.0
    cfg.color_jitter_contrast = 0.0
    cfg.color_jitter_saturation = 0.0
    cfg.gaussian_blur_prob = 0.0

    if preset == "baseline":
        cfg.segmentation_head = "aspp"
        return

    if preset == "large":
        return

    if preset == "large_v3":
        cfg.hybrid_variant = "large_v3"
        return

    if preset == "full_preprocess":
        cfg.color_jitter_prob = 0.5
        cfg.color_jitter_brightness = 0.2
        cfg.color_jitter_contrast = 0.2
        cfg.color_jitter_saturation = 0.2
        cfg.gaussian_blur_prob = 0.2
        return

    if preset == "full_preprocess_classweights":
        cfg.color_jitter_prob = 0.5
        cfg.color_jitter_brightness = 0.2
        cfg.color_jitter_contrast = 0.2
        cfg.color_jitter_saturation = 0.2
        cfg.gaussian_blur_prob = 0.2
        cfg.use_class_weights = True
        return

    raise ValueError(f"Unsupported preset: {preset}")


def main() -> None:
    args = build_parser().parse_args()

    cfg = TrainConfig()
    apply_camvid_preset(cfg, args.preset)
    cfg.exp_tag = f"camvid_{args.preset}"

    if args.camvid_root is not None:
        cfg.camvid_root = Path(args.camvid_root)
        cfg.data_root = Path(args.camvid_root)
    if args.epochs is not None:
        cfg.epochs = int(args.epochs)
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        cfg.num_workers = int(args.num_workers)
    if args.lr is not None:
        cfg.lr_0 = float(args.lr)

    teacher_train.run_training(cfg)


if __name__ == "__main__":
    main()
