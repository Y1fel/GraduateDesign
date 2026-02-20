import math
import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from src.commom.output_manager import OutputManager
from src.commom.repro import set_seed
from src.datasets.CamVid import CamVidFolderDataset
from src.eval.mIoU import compute_segmentation_metrics
from src.models.deeplabv3_plus import DeepLabV3Plus
from src.utils.Id2Mask import load_class_dict_csv
from src.utils.Id2Mask import color_mask_to_id
from src.viz.visualizer import save_predictions_triplet
from src.losses.composite import CrossEntropyDiceLoss


@dataclass
class TrainConfig:
    """训练脚本的集中式超参数配置。

    这个 dataclass 的作用是把数据路径、模型结构、损失函数、增强策略、
    可视化与日志输出等超参数收敛到一个对象里，方便：
    1) 统一管理默认值；
    2) 在实验输出目录保存完整配置；
    3) 通过命令行参数做少量覆盖，而不破坏主配置结构。
    """
    data_root: Path = PROJECT_ROOT / "data" / "archive" / "CamVid"

    num_classes: int = 32
    ignore_index: int = 255

    epochs: int = 100
    batch_size: int = 8
    num_workers: int = 4
    lr_0: float = 5e-4
    weight_decay: float = 1e-4

    ce_weight: float = 1
    dice_weight: float = 0.0
    label_smoothing: float = 0.0

    output_stride: int = 8
    backbone_pretrained: bool = True
    head_norm: str = "bn"
    use_mid_level_fusion: bool = True

    resize_h: int = 720
    resize_w: int = 960
    crop_h: int = 512
    crop_w: int = 768
    train_multi_scale_min: float = 1.0
    train_multi_scale_max: float = 1.0
    hflip_prob: float = 0.5

    photo_aug_prob: float = 0.35
    brightness_jitter: float = 0.10
    contrast_jitter: float = 0.08
    saturation_jitter: float = 0.06
    gamma_min: float = 0.95
    gamma_max: float = 1.05
    photo_op_prob: float = 0.50
    blur_prob: float = 0.03
    blur_radius_min: float = 0.1
    blur_radius_max: float = 0.6
    jpeg_prob: float = 0.03
    jpeg_quality_min: int = 90
    jpeg_quality_max: int = 98
    photo_aug_warmup_epochs: int = 20

    train_auto_contrast_enable: bool = True
    train_auto_contrast_cutoff: float = 1.0
    eval_auto_contrast_enable: bool = False
    eval_auto_contrast_cutoff: float = 1.0

    train_low_light_preprocess_enable: bool = True
    eval_low_light_preprocess_enable: bool = False
    low_light_gamma: float = 0.85
    low_light_brightness_gain: float = 1.10

    avoid_overstrong_tone_ops: bool = True
    photo_aug_prob_cap_when_tone_stack: float = 0.20
    photo_op_prob_cap_when_tone_stack: float = 0.35
    jitter_scale_when_tone_stack: float = 0.7

    ce_variant: str = "ce"  # ce | focal | ohem
    use_class_balanced_ce: bool = False
    loss_preset: str = "baseline"  # baseline | cbce | focal | ohem
    cb_beta: float = 0.999
    focal_gamma: float = 2.0
    ohem_min_kept: int = 100000
    ohem_thresh: float = 0.7
    boundary_weight: float = 1.5
    boundary_weight_warmup_epochs: int = 0

    disable_blur_jpeg_when_boundary_weight_high: bool = True
    boundary_weight_threshold: float = 1.0
    reduced_blur_prob_when_boundary_high: float = 0.0
    reduced_jpeg_prob_when_boundary_high: float = 0.0

    save_vis_every: int = 50
    save_vis_max_items: int = 8

    outputs_root: Path = PROJECT_ROOT / "outputs"
    seed: int = 42
    use_amp: bool = True




def assert_camvid_key_old_ids(id2name: list[str] | dict[int, str]) -> None:
    """校验 CamVid 类别字典是否符合训练脚本对 old_id 的顺序假设。

    训练/评估流程依赖 `class_dict.csv` 中旧标签 id 到类别名称的一致映射。
    这里选取若干关键类别（Sky/Road/Car）做快速 sanity check。

    Args:
        id2name: 旧标签 id 到类别名的映射，可为 list 或 dict。

    Raises:
        RuntimeError: 当关键 old_id 对应名称与预期不一致时抛出，
            防止后续训练在错误标签空间上进行。
    """
    if isinstance(id2name, dict):
        get_name = lambda idx: str(id2name.get(idx, "<MISSING>"))
    else:
        get_name = lambda idx: str(id2name[idx]) if 0 <= idx < len(id2name) else "<MISSING>"

    expected = {21: "Sky", 17: "Road", 5: "Car"}
    print("[CLASS-DICT] key old_id check:")
    mismatch = []
    for old_id, expected_name in expected.items():
        actual_name = get_name(old_id)
        print(f"  - old_id {old_id:>2}: expected={expected_name}, actual={actual_name}")
        if actual_name != expected_name:
            mismatch.append((old_id, expected_name, actual_name))

    if mismatch:
        details = "; ".join(
            f"old_id {old_id} expected {expected_name} but got {actual_name}"
            for old_id, expected_name, actual_name in mismatch
        )
        raise RuntimeError(f"class_dict.csv 顺序假设不一致: {details}")

def parse_args() -> argparse.Namespace:
    """解析训练脚本命令行参数。

    当前只暴露与实验对比直接相关的参数（loss preset、边界权重、ablation 变体），
    其余参数使用 `TrainConfig` 默认值，避免命令行参数过多导致实验不可复现。

    Returns:
        argparse.Namespace: 命令行参数对象。
    """
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ on CamVid")
    parser.add_argument(
        "--loss_preset",
        type=str,
        default=None,
        choices=["baseline", "cbce", "focal", "ohem"],
        help="Loss preset: baseline(plain CE), cbce(class-balanced CE), focal, or ohem.",
    )
    parser.add_argument("--boundary_weight", type=float, default=None, help="Override boundary_weight.")
    parser.add_argument(
        "--ablation_variant",
        type=str,
        default=None,
        choices=["A", "B", "C", "D05", "D10"],
        help="A=baseline, B=disable blur/jpeg, C=disable mid-level fusion, D05/D10 set boundary_weight.",
    )
    return parser.parse_args()


def apply_ablation_variant(cfg: TrainConfig, variant: str) -> None:
    """根据消融实验代号修改配置。

    该函数用于论文/报告中的可控变量实验：
    - A: baseline，不改动；
    - B: 关闭 blur/jpeg；
    - C: 关闭 mid-level fusion；
    - D05/D10: 指定 boundary weight。

    Args:
        cfg: 待修改的训练配置对象。
        variant: 消融实验代号。

    Raises:
        ValueError: 输入未知的 variant 时抛出。
    """
    v = variant.upper()
    if v == "A":
        return
    if v == "B":
        cfg.blur_prob = 0.0
        cfg.jpeg_prob = 0.0
        return
    if v == "C":
        cfg.use_mid_level_fusion = False
        return
    if v == "D05":
        cfg.boundary_weight = 0.5
        return
    if v == "D10":
        cfg.boundary_weight = 1.0
        return
    raise ValueError(f"Unsupported ablation variant: {variant}")


def resolve_boundary_weight_for_epoch(target_weight: float, epoch: int, warmup_epochs: int) -> float:
    """为当前 epoch 计算边界损失权重（支持 warmup）。

    当 `warmup_epochs > 0` 时，权重在前 warmup 轮线性从 0 增长到 `target_weight`，
    目的是避免训练初期边界约束过强影响主分割收敛。
    """
    if warmup_epochs <= 0:
        return float(target_weight)
    progress = min(max(epoch, 0), warmup_epochs) / float(warmup_epochs)
    return float(target_weight) * progress


def resolve_effective_tone_aug(cfg: TrainConfig) -> dict[str, float]:
    """计算最终生效的色调类增强强度。

    当低照度预处理、自动对比度、以及 photometric 增强同时启用时，
    容易出现“叠加强化”导致样本分布偏移过大。函数会按配置自动削弱：
    - photometric 触发概率；
    - photo op 子操作概率；
    - brightness/contrast/saturation 抖动幅度。

    Returns:
        dict[str, float]: 训练集实际采用的增强参数。
    """
    effective = {
        "photo_aug_prob": float(cfg.photo_aug_prob),
        "photo_op_prob": float(cfg.photo_op_prob),
        "brightness_jitter": float(cfg.brightness_jitter),
        "contrast_jitter": float(cfg.contrast_jitter),
        "saturation_jitter": float(cfg.saturation_jitter),
    }

    if cfg.avoid_overstrong_tone_ops and cfg.train_low_light_preprocess_enable and cfg.train_auto_contrast_enable:
        scale = max(0.0, min(1.0, float(cfg.jitter_scale_when_tone_stack)))
        effective["photo_aug_prob"] = min(effective["photo_aug_prob"], float(cfg.photo_aug_prob_cap_when_tone_stack))
        effective["photo_op_prob"] = min(effective["photo_op_prob"], float(cfg.photo_op_prob_cap_when_tone_stack))
        effective["brightness_jitter"] *= scale
        effective["contrast_jitter"] *= scale
        effective["saturation_jitter"] *= scale
        print(
            "[AUG-TONE] low_light + auto_contrast + photometric stack detected; "
            f"scaled photo_aug_prob={effective['photo_aug_prob']:.3f}, "
            f"photo_op_prob={effective['photo_op_prob']:.3f}, jitter_scale={scale:.2f}"
        )

    return effective


def apply_loss_preset(cfg: TrainConfig, loss_preset: str) -> None:
    """根据预设名称选择 CE 变体和类平衡策略。

    这里把可对比的 loss 组合固化成有限 preset，确保实验对比只改变必要变量。
    """
    cfg.loss_preset = loss_preset

    if loss_preset == "baseline":
        cfg.ce_variant = "ce"
        cfg.use_class_balanced_ce = False
    elif loss_preset == "cbce":
        cfg.ce_variant = "ce"
        cfg.use_class_balanced_ce = True
    elif loss_preset == "focal":
        cfg.ce_variant = "focal"
        cfg.use_class_balanced_ce = False
    elif loss_preset == "ohem":
        cfg.ce_variant = "ohem"
        cfg.use_class_balanced_ce = False
    else:
        raise ValueError(f"Unsupported loss_preset: {loss_preset}")


def compute_class_pixel_distribution(
    masks_dir: Path,
    color2id,
    num_classes: int,
    ignore_index: int,
) -> np.ndarray:
    """统计训练标签中每个类别的像素数量。

    流程：逐个读取 mask（RGB）→ 颜色映射到 old_id → 过滤 ignore_index → bincount 累加。
    该统计用于：
    - 打印长尾分布；
    - 计算 class-balanced CE 权重。
    """
    mask_paths = sorted([p for p in masks_dir.iterdir() if p.is_file()])
    if not mask_paths:
        raise RuntimeError(f"No masks found for pixel statistics in {masks_dir}")

    counts = np.zeros((num_classes,), dtype=np.int64)
    for p in mask_paths:
        mask_rgb = np.array(torchvision_safe_open_rgb(p), dtype=np.uint8)
        mask_old = color_mask_to_id(mask_rgb, color2id, ignore_index)
        valid = mask_old != ignore_index
        if np.any(valid):
            binc = np.bincount(mask_old[valid], minlength=num_classes)
            counts += binc[:num_classes]
    return counts


def torchvision_safe_open_rgb(path: Path):
    """以 RGB 模式安全打开图像文件。

    单独封装读取逻辑，便于后续替换读取后端或在此处加入异常处理策略。
    """
    from PIL import Image

    return Image.open(path).convert("RGB")


def class_balanced_weights_from_counts(counts: np.ndarray, beta: float, eps: float = 1e-8) -> np.ndarray:
    """根据 Effective Number of Samples 公式计算类别权重。

    对于样本少（像素少）的类别，分配更高权重；无样本类别权重置 0。
    最后对非零权重做归一化，使其均值接近 1，避免整体 loss 尺度突变。
    """
    counts = counts.astype(np.float64)
    beta = float(beta)
    eff_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(eff_num, eps)
    weights[counts <= 0] = 0.0

    nz = weights > 0
    if np.any(nz):
        weights[nz] = weights[nz] * (nz.sum() / np.sum(weights[nz]))
    return weights.astype(np.float32)


def print_small_object_metrics(metric_name: str, values: Sequence[float], names: Sequence[str], indices: Sequence[int]) -> None:
    """打印指定小目标类别的评估指标，便于观察长尾类别表现。"""
    msg = []
    for cls_name, idx in zip(names, indices):
        v = values[idx]
        if np.isnan(v):
            msg.append(f"{cls_name}=nan")
        else:
            msg.append(f"{cls_name}={v:.4f}")
    print(f"[VAL-small] {metric_name}: " + " | ".join(msg))


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch: int,
    total_iters: int,
    base_lr: float,
    use_amp: bool,
    power: float = 0.9,
) -> dict[str, float]:
    """执行单个 epoch 的训练。

    核心步骤：
    1) polynomial lr 调度；
    2) 前向计算 + CE/Dice 组合损失；
    3) 反向传播与参数更新；
    4) 汇总 batch 级损失为 epoch 均值。

    Returns:
        dict[str, float]: `total/ce/dice` 三项平均损失。
    """
    model.train()
    total_loss, total_ce, total_dice, n = 0.0, 0.0, 0.0, 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for it, (imgs, masks, _names) in enumerate(loader):
        global_step = (epoch - 1) * len(loader) + it
        lr = base_lr * (1 - global_step / total_iters) ** power
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(imgs)
            loss_parts = criterion.forward_components(logits, masks)
            loss = loss_parts["total"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_ce += loss_parts["ce"].item() * bs
        total_dice += loss_parts["dice"].item() * bs
        n += bs

    denom = max(n, 1)
    return {
        "total": total_loss / denom,
        "ce": total_ce / denom,
        "dice": total_dice / denom,
    }


@torch.inference_mode()
def evaluate_loss(model, loader, criterion, device, use_amp: bool) -> dict[str, float]:
    """在验证集上计算平均损失（不更新梯度）。"""
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0
    n = 0

    for imgs, masks, _names in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(imgs)
            loss_parts = criterion.forward_components(logits, masks)

        bs = imgs.size(0)
        total_loss += loss_parts["total"].item() * bs
        total_ce += loss_parts["ce"].item() * bs
        total_dice += loss_parts["dice"].item() * bs
        n += bs

    denom = max(n, 1)
    return {
        "total": total_loss / denom,
        "ce": total_ce / denom,
        "dice": total_dice / denom,
    }


@torch.inference_mode()
def save_vis_using_best_ckpt(
    model,
    val_loader,
    device,
    out_dir: Path,
    id2color,
    ignore_index: int,
    epoch: int,
    max_items: int,
    best_ckpt_path: Path,
) -> None:
    """临时加载 best checkpoint 做可视化，再恢复当前模型权重。

    这样可以保证保存的可视化结果始终对应“当前最优模型”，
    同时不打断训练过程中的参数状态。
    """
    cur_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(state, strict=True)

    model.eval()
    save_predictions_triplet(
        model=model,
        loader=val_loader,
        device=device,
        out_dir=out_dir,
        id2color=id2color,
        ignore_index=ignore_index,
        epoch=epoch,
        max_items=max_items,
    )

    model.load_state_dict(cur_state, strict=True)


def main() -> None:
    """训练脚本入口函数。

    主要流程：
    - 读取配置与命令行覆盖；
    - 构建数据集/模型/损失/优化器；
    - 逐 epoch 训练并验证；
    - 记录指标、保存 best/periodic checkpoint、输出可视化结果。
    """
    args = parse_args()
    cfg = TrainConfig()
    apply_loss_preset(cfg, args.loss_preset or cfg.loss_preset)
    if args.boundary_weight is not None:
        cfg.boundary_weight = float(args.boundary_weight)
    if args.ablation_variant is not None:
        apply_ablation_variant(cfg, args.ablation_variant)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    print(f"[INFO] AMP enabled = {amp_enabled}")

    # 32类颜色表/映射用于 RGB->old_id 解码与可视化
    csv_path = cfg.data_root / "class_dict.csv"
    color2id, id2color, id2name = load_class_dict_csv(csv_path)
    assert_camvid_key_old_ids(id2name)

    id2color_vis = id2color
    class_names = [id2name[i] for i in range(cfg.num_classes)]

    # outputs
    out = OutputManager(cfg.outputs_root, exp_name="camvid_deeplabv3plus")
    out.save_config(cfg)
    out.init_metrics()
    print(f"[INFO] run_dir = {out.run_dir}")
    print(
        "[PREPROCESS][train] "
        f"low_light={cfg.train_low_light_preprocess_enable} gamma={cfg.low_light_gamma:.2f} "
        f"brightness_gain={cfg.low_light_brightness_gain:.2f} "
        f"auto_contrast={cfg.train_auto_contrast_enable} cutoff={cfg.train_auto_contrast_cutoff:.2f}"
    )
    print(
        "[PREPROCESS][val] "
        f"low_light={cfg.eval_low_light_preprocess_enable} gamma={cfg.low_light_gamma:.2f} "
        f"brightness_gain={cfg.low_light_brightness_gain:.2f} "
        f"auto_contrast={cfg.eval_auto_contrast_enable} cutoff={cfg.eval_auto_contrast_cutoff:.2f}"
    )

    train_class_pixel_counts = compute_class_pixel_distribution(
        masks_dir=cfg.data_root / "train_labels",
        color2id=color2id,
        num_classes=cfg.num_classes,
        ignore_index=cfg.ignore_index,
    )
    pixel_ratio = train_class_pixel_counts / np.maximum(train_class_pixel_counts.sum(), 1)
    print("[DATA] train pixel ratios:")
    for i, name in enumerate(class_names):
        print(f"  - {name:<10} count={int(train_class_pixel_counts[i]):>10d} ratio={pixel_ratio[i] * 100:6.3f}%")

    tail_check_indices = [i for i in [5, 20, 21] if i < cfg.num_classes]
    for idx in tail_check_indices:
        print(f"[TAIL-CHECK] {class_names[idx]} ratio={pixel_ratio[idx] * 100:.3f}%")

    print(f"[LOSS-PRESET] selected={cfg.loss_preset}")
    if cfg.loss_preset == "baseline":
        print("[LOSS-PRESET] baseline enforces ce_variant=ce and class_weights=None.")
    elif cfg.loss_preset == "cbce":
        print("[LOSS-PRESET] cbce enabled: use only after validating long-tail gains.")

    class_weights_t = None
    if cfg.use_class_balanced_ce:
        cb_w = class_balanced_weights_from_counts(train_class_pixel_counts, beta=cfg.cb_beta)
        class_weights_t = torch.tensor(cb_w, dtype=torch.float32, device=device)
        print("[LOSS] class-balanced CE weights:", np.array2string(cb_w, precision=4, separator=", "))

    class_weights_state = "enabled" if class_weights_t is not None else "disabled(None)"
    print(
        "[LOSS] effective setup: "
        f"preset={cfg.loss_preset} | ce_variant={cfg.ce_variant} | "
        f"class_balanced_ce={cfg.use_class_balanced_ce} | class_weights={class_weights_state} | "
        f"ce_weight={cfg.ce_weight} | dice_weight={cfg.dice_weight} | boundary_weight={cfg.boundary_weight}"
    )

    effective_blur_prob = cfg.blur_prob
    effective_jpeg_prob = cfg.jpeg_prob
    if (
        cfg.disable_blur_jpeg_when_boundary_weight_high
        and cfg.boundary_weight >= cfg.boundary_weight_threshold
    ):
        effective_blur_prob = cfg.reduced_blur_prob_when_boundary_high
        effective_jpeg_prob = cfg.reduced_jpeg_prob_when_boundary_high

    print(
        "[AUG-LINK] "
        f"boundary_weight={cfg.boundary_weight:.3f} threshold={cfg.boundary_weight_threshold:.3f} "
        f"blur_prob={effective_blur_prob:.4f} jpeg_prob={effective_jpeg_prob:.4f}"
    )

    tone_aug = resolve_effective_tone_aug(cfg)

    # datasets
    train_preprocess = {
        "hflip_prob": cfg.hflip_prob,
        "photo_aug_prob": tone_aug["photo_aug_prob"],
        "brightness_jitter": tone_aug["brightness_jitter"],
        "contrast_jitter": tone_aug["contrast_jitter"],
        "saturation_jitter": tone_aug["saturation_jitter"],
        "gamma_range": (cfg.gamma_min, cfg.gamma_max),
        "photo_op_prob": tone_aug["photo_op_prob"],
        "blur_prob": effective_blur_prob,
        "blur_radius_range": (cfg.blur_radius_min, cfg.blur_radius_max),
        "jpeg_prob": effective_jpeg_prob,
        "jpeg_quality_range": (cfg.jpeg_quality_min, cfg.jpeg_quality_max),
        "multi_scale_range": (cfg.train_multi_scale_min, cfg.train_multi_scale_max),
        "random_crop_size": None,
        "auto_contrast": cfg.train_auto_contrast_enable,
        "auto_contrast_cutoff": cfg.train_auto_contrast_cutoff,
        "low_light_preprocess_enable": cfg.train_low_light_preprocess_enable,
        "low_light_gamma": cfg.low_light_gamma,
        "low_light_brightness_gain": cfg.low_light_brightness_gain,
    }
    eval_preprocess = {
        "auto_contrast": cfg.eval_auto_contrast_enable,
        "auto_contrast_cutoff": cfg.eval_auto_contrast_cutoff,
        "low_light_preprocess_enable": cfg.eval_low_light_preprocess_enable,
        "low_light_gamma": cfg.low_light_gamma,
        "low_light_brightness_gain": cfg.low_light_brightness_gain,
    }

    print(
        "[PREPROCESS-CONFIG][train] "
        f"hflip={train_preprocess['hflip_prob']}, multiscale={train_preprocess['multi_scale_range']}, "
        f"photo_aug_prob={train_preprocess['photo_aug_prob']:.3f}, blur_prob={train_preprocess['blur_prob']:.3f}, "
        f"jpeg_prob={train_preprocess['jpeg_prob']:.3f}, low_light={train_preprocess['low_light_preprocess_enable']}, "
        f"auto_contrast={train_preprocess['auto_contrast']}"
    )
    print(
        "[PREPROCESS-CONFIG][val] "
        f"minimal preprocess (resize+normalize) with low_light={eval_preprocess['low_light_preprocess_enable']}, "
        f"auto_contrast={eval_preprocess['auto_contrast']}"
    )

    train_ds = CamVidFolderDataset(
        root=cfg.data_root,
        split="train",
        color2id=color2id,
        resize_w=cfg.resize_w,
        resize_h=cfg.resize_h,
        ignore_index=cfg.ignore_index,
        training=True,
        label_lut=None,
        train_preprocess=train_preprocess,
        eval_preprocess=eval_preprocess,
    )
    val_ds = CamVidFolderDataset(
        root=cfg.data_root,
        split="val",
        color2id=color2id,
        resize_w=cfg.resize_w,
        resize_h=cfg.resize_h,
        ignore_index=cfg.ignore_index,
        training=False,
        label_lut=None,
        train_preprocess=train_preprocess,
        eval_preprocess=eval_preprocess,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # model
    model = DeepLabV3Plus(
        num_classes=cfg.num_classes,
        backbone_pretrained=cfg.backbone_pretrained,
        output_stride=cfg.output_stride,
        head_norm=cfg.head_norm,
        use_mid_level_fusion=cfg.use_mid_level_fusion,
    ).to(device)

    target_boundary_weight = float(cfg.boundary_weight)
    init_boundary_weight = resolve_boundary_weight_for_epoch(
        target_weight=target_boundary_weight,
        epoch=1,
        warmup_epochs=cfg.boundary_weight_warmup_epochs,
    )

    criterion = CrossEntropyDiceLoss(
        num_classes=cfg.num_classes,
        ignore_index=cfg.ignore_index,
        ce_weight=cfg.ce_weight,
        dice_weight=cfg.dice_weight,
        label_smoothing=cfg.label_smoothing,
        dice_include_background=True,
        ce_variant=cfg.ce_variant,
        class_weights=class_weights_t,
        focal_gamma=cfg.focal_gamma,
        ohem_min_kept=cfg.ohem_min_kept,
        ohem_thresh=cfg.ohem_thresh,
        boundary_weight=init_boundary_weight,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr_0,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
        nesterov=True,
    )

    best_miou = -1.0
    best_val_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        total_iters = cfg.epochs * len(train_loader)
        t0 = time.time()

        current_boundary_weight = resolve_boundary_weight_for_epoch(
            target_weight=target_boundary_weight,
            epoch=epoch,
            warmup_epochs=cfg.boundary_weight_warmup_epochs,
        )
        criterion.boundary_weight = current_boundary_weight

        if hasattr(train_ds, "reset_aug_stats"):
            train_ds.reset_aug_stats()

        if hasattr(train_ds, "set_photo_aug_scale"):
            if cfg.photo_aug_warmup_epochs > 0:
                aug_scale = min(1.0, epoch / float(cfg.photo_aug_warmup_epochs))
            else:
                aug_scale = 1.0
            train_ds.set_photo_aug_scale(aug_scale)
            print(
                f"[AUG] photo_aug_prob={train_ds.photo_aug_prob_current:.3f} (scale={aug_scale:.2f}) "
                f"boundary_weight={current_boundary_weight:.3f}"
            )

        train_loss_parts = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch=epoch,
            total_iters=total_iters,
            base_lr=cfg.lr_0,
            use_amp=amp_enabled,
        )
        val_loss_parts = evaluate_loss(model, val_loader, criterion, device, use_amp=amp_enabled)
        train_loss = train_loss_parts["total"]
        val_loss = val_loss_parts["total"]
        val_metrics = compute_segmentation_metrics(model, val_loader, device, cfg.num_classes, cfg.ignore_index)
        val_miou = float(val_metrics["miou"])

        dt = time.time() - t0
        print(
            f"[EPOCH {epoch:03d}/{cfg.epochs}] train_loss={train_loss:.4f} (closer to 0 is better) "
            f"[ce={train_loss_parts['ce']:.4f}, dice={train_loss_parts['dice']:.4f}] "
            f" val_loss={val_loss:.4f} [ce={val_loss_parts['ce']:.4f}, dice={val_loss_parts['dice']:.4f}] "
            f" val_mIoU={val_miou:.4f} "
            f" val_BF1={val_metrics['boundary_fscore']:.4f} val_TrimapIoU={val_metrics['trimap_iou']:.4f}  time={dt:.1f}s"
        )
        small_indices = [i for i in [5, 20, 21] if i < cfg.num_classes]
        print_small_object_metrics("IoU", val_metrics["iou_per_class"], class_names, small_indices)
        print_small_object_metrics("Recall", val_metrics["recall_per_class"], class_names, small_indices)
        if cfg.use_class_balanced_ce:
            print_small_object_metrics("Precision", val_metrics["precision_per_class"], class_names, small_indices)
        if hasattr(train_ds, "consume_aug_stats"):
            aug_stats = train_ds.consume_aug_stats()
            print(
                "[AUG-STATS] "
                f"photometric={aug_stats['photometric_applied']}/{aug_stats['samples_seen']} "
                f"blur={aug_stats['blur_applied']} jpeg={aug_stats['jpeg_applied']}"
            )
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[MEM] peak_allocated = {peak:.2f} GB")

        out.append_metrics(epoch, train_loss, val_loss, val_miou, dt)

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_miou": best_miou,
            "best_val_loss": best_val_loss,
        }

        if epoch % 10 == 0:
            torch.save(ckpt, out.ckpt_dir / f"epoch_{epoch:03d}.pth")

        if (not math.isnan(val_loss)) and (val_loss < best_val_loss):
            best_val_loss = val_loss

        if (not math.isnan(val_miou)) and (val_miou > best_miou):
            best_miou = val_miou
            ckpt["best_miou"] = best_miou
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, out.ckpt_dir / "best.pth")
            print(f"[INFO] New best mIoU = {best_miou:.4f} -> saved best.pth (current val_loss={val_loss:.4f})")

        if epoch % cfg.save_vis_every == 0:
            print(f"[INFO] Saving visualizations (best.pth) at epoch {epoch} ...")
            save_vis_using_best_ckpt(
                model=model,
                val_loader=val_loader,
                device=device,
                out_dir=out.vis_dir,
                id2color=id2color_vis,
                ignore_index=cfg.ignore_index,
                epoch=epoch,
                max_items=cfg.save_vis_max_items,
                best_ckpt_path=out.ckpt_dir / "best.pth",
            )

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"... lr={cur_lr:.6f}")

    print("[DONE] Training finished.")


if __name__ == "__main__":
    main()
