import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import TrainConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train comparison segmentation models on Cityscapes or CamVid."
    )
    parser.add_argument("--model", type=str, required=True, choices=["unet", "pspnet", "fcn"])
    parser.add_argument("--dataset", type=str, required=True, choices=["cityscapes", "camvid"])
    parser.add_argument("--data-root", type=Path, default=None, help="Override dataset root.")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Override validation batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--lr", type=float, default=None, help="Override initial learning rate.")
    parser.add_argument("--backbone", type=str, default=None, help="Optional override for ResNet-based models.")
    parser.add_argument("--output-stride", type=int, default=None, choices=[8, 16], help="Optional override for ResNet-based models.")
    parser.add_argument("--full-preprocess", action="store_true", help="Enable color jitter and gaussian blur.")
    parser.add_argument("--class-weights", action="store_true", help="Enable class weights.")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained encoder.")
    parser.add_argument("--use-aux", action="store_true", help="Force enable auxiliary loss.")
    parser.add_argument("--no-aux", action="store_true", help="Force disable auxiliary loss.")
    parser.add_argument("--exp-tag", type=str, default="", help="Optional extra experiment tag.")
    return parser


def _apply_dataset_preset(cfg: TrainConfig, dataset: str) -> None:
    cfg.dataset_name = str(dataset).lower()


def _resolve_aux_flag(args: argparse.Namespace) -> bool:
    if args.use_aux and args.no_aux:
        raise ValueError("--use-aux and --no-aux cannot be used together")
    if args.use_aux:
        return True
    if args.no_aux:
        return False
    return args.model in {"pspnet", "fcn"}


def build_cfg(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    cfg.model_name = str(args.model).lower()
    cfg.loss_mode = "ce"
    cfg.use_class_weights = bool(args.class_weights)
    cfg.use_aux_loss = _resolve_aux_flag(args)
    _apply_dataset_preset(cfg, args.dataset)

    if args.full_preprocess:
        cfg.color_jitter_prob = 0.5
        cfg.color_jitter_brightness = 0.2
        cfg.color_jitter_contrast = 0.2
        cfg.color_jitter_saturation = 0.2
        cfg.gaussian_blur_prob = 0.2

    if args.backbone is not None:
        cfg.backbone_name = str(args.backbone)
    cfg.backbone_pretrained = not bool(args.no_pretrained)
    if args.output_stride is not None:
        cfg.output_stride = int(args.output_stride)

    if args.data_root is not None:
        cfg.data_root = Path(args.data_root)
        if cfg.dataset_name == "cityscapes":
            cfg.cityscapes_root = Path(args.data_root)
        else:
            cfg.camvid_root = Path(args.data_root)

    if args.epochs is not None:
        cfg.epochs = int(args.epochs)
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.eval_batch_size is not None:
        cfg.eval_batch_size = int(args.eval_batch_size)
    if args.num_workers is not None:
        cfg.num_workers = int(args.num_workers)
    if args.lr is not None:
        cfg.lr_0 = float(args.lr)

    tag_parts = ["compare"]
    if args.full_preprocess:
        tag_parts.append("fullpre")
    if args.class_weights:
        tag_parts.append("cw")
    if cfg.use_aux_loss:
        tag_parts.append("aux")
    if args.exp_tag:
        tag_parts.append(str(args.exp_tag))
    cfg.exp_tag = "_".join(tag_parts)
    return cfg


def main() -> None:
    args = build_parser().parse_args()
    cfg = build_cfg(args)
    if __package__:
        from . import train as teacher_train
    else:
        import train as teacher_train
    teacher_train.run_training(cfg)


if __name__ == "__main__":
    main()
