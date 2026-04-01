import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader, WeightedRandomSampler

from config.config import MobileTrainConfig
from src.commom.output_manager import OutputManager
from src.commom.repro import set_seed
from src.datasets.cityscapes import CityscapesDataset
from src.datasets.cityscapes_labels import (
    CITYSCAPES_19_CLASS_NAMES,
    CITYSCAPES_19_ID2COLOR,
    CITYSCAPES_34_TO_19,
)
from src.eval.mIoU import compute_segmentation_metrics
from src.losses.combined_loss import CombinedCEFocalLoss, OHEMBoundaryLoss, OHEMCELoss
from src.models.deeplabv3_plus import DeepLabV3Plus
from src.models.deeplabv3_plus_moblie import DeepLabV3PlusMobile
from src.viz.visualizer import save_predictions_triplet


def freeze_bn(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.eval()
            if m.weight is not None:
                m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.requires_grad = False


def _accumulate_pred_hist(pred: torch.Tensor, hist: torch.Tensor, num_classes: int) -> None:
    hist += torch.bincount(pred.view(-1), minlength=num_classes).to(hist.device, dtype=hist.dtype)


def _compute_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().float().norm(2).item()
            total += g * g
    return total ** 0.5


def _build_rare_class_sampler(train_ds: CityscapesDataset, cfg: MobileTrainConfig) -> WeightedRandomSampler:
    rare_ids = set(int(cid) for cid in cfg.rare_class_ids)
    remap = np.asarray(CITYSCAPES_34_TO_19, dtype=np.uint8)
    weights = np.ones(len(train_ds.img_paths), dtype=np.float64)

    for idx, img_path in enumerate(train_ds.img_paths):
        mask_path = train_ds._resolve_mask(img_path)
        mask_34 = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        valid = mask_34 <= 33
        mapped = np.full(mask_34.shape, fill_value=train_ds.ignore_index, dtype=np.uint8)
        mapped[valid] = remap[mask_34[valid]]
        present = set(np.unique(mapped).tolist())
        if any(cid in present for cid in rare_ids):
            weights[idx] *= float(cfg.rare_class_weight_multiplier)

    num_samples = max(1, int(len(train_ds) * float(cfg.sampler_num_samples_factor)))
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=num_samples,
        replacement=True,
    )


def _compute_class_weights(train_ds: CityscapesDataset, cfg: MobileTrainConfig) -> torch.Tensor:
    remap = np.asarray(CITYSCAPES_34_TO_19, dtype=np.uint8)
    counts = np.zeros(cfg.num_classes, dtype=np.float64)

    for img_path in train_ds.img_paths:
        mask_path = train_ds._resolve_mask(img_path)
        mask_34 = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        valid = mask_34 <= 33
        mapped = remap[mask_34[valid]]
        mapped = mapped[mapped < cfg.num_classes]
        counts += np.bincount(mapped, minlength=cfg.num_classes).astype(np.float64)

    counts = np.maximum(counts, 1.0)
    freq = counts / counts.sum()
    if str(cfg.class_weight_strategy).lower() == "median_frequency":
        median_freq = np.median(freq[freq > 0])
        weights = median_freq / freq
    else:
        weights = 1.0 / np.power(freq, float(cfg.class_weight_power))

    weights = weights / weights.mean()
    low_freq_threshold = np.quantile(freq, 0.2)
    rare_mask = freq <= low_freq_threshold
    weights[rare_mask] = np.minimum(weights[rare_mask], float(cfg.class_weight_rare_cap))
    weights = np.clip(weights, float(cfg.class_weight_min), float(cfg.class_weight_max))
    return torch.tensor(weights, dtype=torch.float32)


def _build_criterion(cfg: MobileTrainConfig, class_weights: torch.Tensor | None, device: torch.device) -> nn.Module:
    mode = str(cfg.loss_mode).lower().replace("+", "_")
    if mode not in {"baseline", "ohem", "ohem_boundary"}:
        raise ValueError(f"Unsupported loss_mode={cfg.loss_mode}. Use baseline/ohem/ohem_boundary")

    if mode == "baseline":
        criterion = CombinedCEFocalLoss(
            ce_weight=cfg.ce_weight,
            focal_weight=cfg.focal_weight,
            focal_gamma=cfg.focal_gamma,
            class_weights=class_weights,
            label_smoothing=cfg.label_smoothing,
            ignore_index=cfg.ignore_index,
        )
    elif mode == "ohem":
        criterion = OHEMCELoss(
            ignore_index=cfg.ignore_index,
            ohem_ratio=cfg.ohem_ratio,
            class_weights=class_weights,
        )
    else:
        criterion = OHEMBoundaryLoss(
            ignore_index=cfg.ignore_index,
            ohem_ratio=cfg.ohem_ratio,
            class_weights=class_weights,
            ohem_weight=cfg.ohem_weight,
            boundary_weight=cfg.boundary_weight,
            boundary_kernel_size=cfg.boundary_kernel_size,
        )

    return criterion.to(device)


def _compute_lr_at_iter(global_iter: int, max_iter: int, cfg: MobileTrainConfig) -> float:
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}")

    warmup_iters = max(0, int(cfg.warmup_iters))
    base_lr = float(cfg.lr_0)
    eta_min = float(cfg.lr_eta_min)
    policy = str(cfg.lr_policy).lower()

    if warmup_iters > 0 and global_iter < warmup_iters:
        alpha = float(global_iter + 1) / float(warmup_iters)
        return base_lr * (float(cfg.warmup_ratio) + (1.0 - float(cfg.warmup_ratio)) * alpha)

    progress = (global_iter - warmup_iters) / max(1, (max_iter - warmup_iters))
    progress = min(max(progress, 0.0), 1.0)

    if policy == "poly":
        lr = base_lr * ((1.0 - progress) ** float(cfg.poly_power))
    elif policy == "cosine":
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = eta_min + (base_lr - eta_min) * cosine
    else:
        raise ValueError(f"Unsupported lr_policy={cfg.lr_policy}. Use poly/cosine")

    return max(eta_min, lr)


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = float(lr)


def _detect_teacher_arch(state: dict[str, torch.Tensor], arch_cfg: str) -> str:
    arch_cfg = arch_cfg.lower()
    if arch_cfg in {"resnet", "mobile"}:
        return arch_cfg
    if arch_cfg != "auto":
        raise ValueError(f"Unsupported distill_teacher_arch={arch_cfg}. Use auto/resnet/mobile")
    if any(k.startswith("backbone.features") for k in state.keys()):
        return "mobile"
    return "resnet"


def _detect_teacher_segmentation_head(state: dict[str, torch.Tensor]) -> str:
    if any(k.startswith("ocr_pre.") or k.startswith("ocr_head.") for k in state.keys()):
        return "ocr"
    return "aspp"


def _load_teacher_model(cfg: MobileTrainConfig, device: torch.device) -> nn.Module | None:
    if not bool(cfg.use_distillation):
        return None

    teacher_ckpt = Path(cfg.distill_teacher_ckpt)
    if not teacher_ckpt.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt}")

    raw = torch.load(teacher_ckpt, map_location="cpu")
    state = raw["model_state"] if isinstance(raw, dict) and "model_state" in raw else raw
    if not isinstance(state, dict):
        raise RuntimeError("Teacher checkpoint format invalid: no state dict found")

    teacher_arch = _detect_teacher_arch(state, str(cfg.distill_teacher_arch))
    teacher_head = _detect_teacher_segmentation_head(state)

    if teacher_arch == "resnet":
        teacher = DeepLabV3Plus(
            num_classes=cfg.num_classes,
            backbone_pretrained=cfg.distill_teacher_backbone_pretrained,
            backbone_name=cfg.distill_teacher_backbone_name,
            output_stride=cfg.distill_teacher_output_stride,
            segmentation_head=teacher_head,
            aspp_dropout=cfg.aspp_dropout,
            ocr_mid_channels=cfg.ocr_mid_channels,
            ocr_key_channels=cfg.ocr_key_channels,
            ocr_dropout=cfg.ocr_dropout,
            decoder_dropout=cfg.decoder_dropout,
        )
    else:
        teacher = DeepLabV3PlusMobile(
            num_classes=cfg.num_classes,
            output_stride=cfg.distill_teacher_output_stride,
            aspp_dropout=cfg.aspp_dropout,
            decoder_dropout=cfg.decoder_dropout,
        )

    teacher.load_state_dict(state, strict=True)
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"[INFO] Teacher loaded: {teacher_ckpt} (arch={teacher_arch}, stride={cfg.distill_teacher_output_stride})")
    return teacher


def _distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, cfg: MobileTrainConfig) -> torch.Tensor:
    t = float(cfg.distill_temperature)
    if t <= 0:
        raise ValueError("distill_temperature must be positive")

    if teacher_logits.shape[-2:] != student_logits.shape[-2:]:
        teacher_logits = F.interpolate(
            teacher_logits,
            size=student_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    distill_type = str(cfg.distill_type).lower()
    if distill_type == "kl":
        s_log_prob = F.log_softmax(student_logits / t, dim=1)
        t_prob = F.softmax(teacher_logits / t, dim=1)
        return F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (t * t)

    if distill_type == "cwd":
        n, c, _, _ = student_logits.shape
        s = student_logits.reshape(n, c, -1)
        te = teacher_logits.reshape(n, c, -1)
        s_log_prob = F.log_softmax(s / t, dim=-1)
        t_prob = F.softmax(te / t, dim=-1)
        return F.kl_div(s_log_prob, t_prob, reduction="none").sum(dim=-1).mean() * (t * t)

    raise ValueError(f"Unsupported distill_type={cfg.distill_type}. Use cwd/kl")


def _compute_total_loss(
    student_main: torch.Tensor,
    student_aux: torch.Tensor,
    target: torch.Tensor,
    criterion: nn.Module,
    cfg: MobileTrainConfig,
    teacher_main: torch.Tensor | None = None,
    teacher_aux: torch.Tensor | None = None,
    return_components: bool = False,
) -> tuple[torch.Tensor, dict[str, float] | None]:
    if return_components:
        main_loss, c = criterion(student_main, target, return_components=True)
        components = dict(c)
    else:
        main_loss = criterion(student_main, target)
        components = None

    aux_loss = criterion(student_aux, target)
    total = main_loss + float(cfg.aux_loss_weight) * aux_loss

    if components is not None:
        components["main"] = float(main_loss.detach())
        components["aux"] = float(aux_loss.detach())

    if teacher_main is not None and teacher_aux is not None:
        kd_main = _distill_loss(student_main, teacher_main, cfg)
        kd_aux = _distill_loss(student_aux, teacher_aux, cfg)
        kd_total = kd_main + float(cfg.distill_aux_weight) * kd_aux
        total = total + float(cfg.distill_loss_weight) * kd_total
        if components is not None:
            components["kd_main"] = float(kd_main.detach())
            components["kd_aux"] = float(kd_aux.detach())
            components["kd_total"] = float(kd_total.detach())

    if components is not None:
        components["total"] = float(total.detach())
    return total, components


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    cfg: MobileTrainConfig,
    epoch: int,
    max_iter: int,
    teacher_model: nn.Module | None,
    return_loss_components: bool,
) -> dict:
    model.train()
    if cfg.freeze_bn:
        freeze_bn(model)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    pred_hist = torch.zeros(cfg.num_classes, dtype=torch.int64, device=device)
    total_loss = 0.0
    n = 0
    grad_norm_sum = 0.0
    loss_components_sum = None
    data_time_sum = 0.0
    compute_time_sum = 0.0
    loop_t = time.perf_counter()

    iter_start = (epoch - 1) * len(loader)
    for it, (imgs, masks, _names) in enumerate(loader):
        global_iter = iter_start + it
        _set_optimizer_lr(optimizer, _compute_lr_at_iter(global_iter, max_iter, cfg))

        data_time_sum += time.perf_counter() - loop_t
        iter_t = time.perf_counter()

        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            student_main, student_aux = model(imgs, return_aux=True)
            teacher_main = None
            teacher_aux = None
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_main, teacher_aux = teacher_model(imgs, return_aux=True)

            loss, loss_components = _compute_total_loss(
                student_main=student_main,
                student_aux=student_aux,
                target=masks,
                criterion=criterion,
                cfg=cfg,
                teacher_main=teacher_main,
                teacher_aux=teacher_aux,
                return_components=return_loss_components,
            )

        scaler.scale(loss).backward()
        if use_amp:
            scaler.unscale_(optimizer)
        grad_norm_sum += _compute_grad_norm(model)
        scaler.step(optimizer)
        scaler.update()

        pred = torch.argmax(student_main.detach(), dim=1)
        _accumulate_pred_hist(pred, pred_hist, cfg.num_classes)

        bs = imgs.size(0)
        total_loss += float(loss.detach()) * bs
        n += bs

        if loss_components is not None:
            if loss_components_sum is None:
                loss_components_sum = {k: 0.0 for k in loss_components}
            for k, v in loss_components.items():
                loss_components_sum[k] += float(v) * bs

        compute_time_sum += time.perf_counter() - iter_t
        loop_t = time.perf_counter()

    return {
        "loss": total_loss / max(n, 1),
        "pred_hist": pred_hist.detach().cpu(),
        "avg_grad_norm": grad_norm_sum / max(len(loader), 1),
        "avg_data_time": data_time_sum / max(len(loader), 1),
        "avg_compute_time": compute_time_sum / max(len(loader), 1),
        "last_lr": optimizer.param_groups[0]["lr"],
        "loss_components": (
            {k: v / max(n, 1) for k, v in loss_components_sum.items()}
            if return_loss_components and loss_components_sum is not None
            else None
        ),
    }


@torch.inference_mode()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    cfg: MobileTrainConfig,
    return_loss_components: bool,
) -> tuple[float, dict[str, float] | None]:
    model.eval()
    total_loss = 0.0
    n = 0
    loss_components_sum = None

    for imgs, masks, _names in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            main_logits, aux_logits = model(imgs, return_aux=True)
            loss, loss_components = _compute_total_loss(
                student_main=main_logits,
                student_aux=aux_logits,
                target=masks,
                criterion=criterion,
                cfg=cfg,
                return_components=return_loss_components,
            )

        bs = imgs.size(0)
        total_loss += float(loss.detach()) * bs
        n += bs

        if loss_components is not None:
            if loss_components_sum is None:
                loss_components_sum = {k: 0.0 for k in loss_components}
            for k, v in loss_components.items():
                loss_components_sum[k] += float(v) * bs

    avg_loss = total_loss / max(n, 1)
    if return_loss_components and loss_components_sum is not None:
        return avg_loss, {k: v / max(n, 1) for k, v in loss_components_sum.items()}
    return avg_loss, None


def _append_loss_components(out: OutputManager, epoch: int, split: str, components: dict[str, float] | None) -> None:
    if components is None:
        return
    out.append_loss_components(
        epoch=epoch,
        split=split,
        ohem_ce=float(components.get("ohem_ce", 0.0)),
        boundary=float(components.get("boundary", 0.0)),
        total=float(components.get("total", 0.0)),
    )


def _maybe_save_vis(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    out: OutputManager,
    cfg: MobileTrainConfig,
    epoch: int,
) -> None:
    if epoch % cfg.save_vis_every != 0:
        return
    print(f"[INFO] Saving visualizations at epoch={epoch}")
    save_predictions_triplet(
        model=model,
        loader=val_loader,
        device=device,
        out_dir=out.vis_dir,
        id2color=CITYSCAPES_19_ID2COLOR,
        ignore_index=cfg.ignore_index,
        epoch=epoch,
        max_items=cfg.save_vis_max_items,
    )


def main() -> None:
    cfg = MobileTrainConfig()
    set_seed(cfg.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    print(f"[INFO] device={device}, amp={amp_enabled}")

    out = OutputManager(cfg.outputs_root, exp_name="cityscapes_deeplabv3plus_mobile_distill")
    out.save_config(cfg)
    out.init_metrics()
    print(f"[INFO] run_dir={out.run_dir}")

    train_ds = CityscapesDataset(
        root=cfg.data_root,
        split="train",
        ignore_index=cfg.ignore_index,
        training=True,
        hflip_prob=cfg.hflip_prob,
        multi_scale_range=(cfg.train_multi_scale_min, cfg.train_multi_scale_max),
        random_crop_size=(cfg.crop_w, cfg.crop_h),
        crop_retry=cfg.crop_retry,
        crop_max_class_ratio=cfg.crop_max_class_ratio,
        color_jitter_prob=cfg.color_jitter_prob,
        color_jitter_brightness=cfg.color_jitter_brightness,
        color_jitter_contrast=cfg.color_jitter_contrast,
        color_jitter_saturation=cfg.color_jitter_saturation,
        gaussian_blur_prob=cfg.gaussian_blur_prob,
        gaussian_blur_radius_range=(cfg.gaussian_blur_radius_min, cfg.gaussian_blur_radius_max),
    )
    val_ds = CityscapesDataset(
        root=cfg.data_root,
        split="val",
        ignore_index=cfg.ignore_index,
        training=False,
    )

    sampler = _build_rare_class_sampler(train_ds, cfg) if cfg.use_rare_class_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=bool(cfg.persistent_workers and cfg.num_workers > 0),
        prefetch_factor=(cfg.prefetch_factor if cfg.num_workers > 0 else None),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=bool(cfg.persistent_workers and cfg.num_workers > 0),
        prefetch_factor=(cfg.prefetch_factor if cfg.num_workers > 0 else None),
        drop_last=False,
    )

    student = DeepLabV3PlusMobile(
        num_classes=cfg.num_classes,
        output_stride=cfg.output_stride,
        aspp_dropout=cfg.aspp_dropout,
        decoder_dropout=cfg.decoder_dropout,
    ).to(device)

    teacher = _load_teacher_model(cfg, device)
    if teacher is not None:
        print(
            f"[INFO] Distillation enabled: type={cfg.distill_type}, T={cfg.distill_temperature}, "
            f"loss_w={cfg.distill_loss_weight}, aux_w={cfg.distill_aux_weight}"
        )

    class_weights = _compute_class_weights(train_ds, cfg).to(device) if cfg.use_class_weights else None
    criterion = _build_criterion(cfg, class_weights, device)

    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=cfg.lr_0,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
        nesterov=True,
    )

    max_iter = cfg.epochs * len(train_loader)
    id2name = {idx: name for idx, name in enumerate(CITYSCAPES_19_CLASS_NAMES)}

    best_miou = -1.0
    best_val_loss = float("inf")
    report_components = str(cfg.loss_mode).lower() != "baseline"

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        train_stats = train_one_epoch(
            model=student,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            use_amp=amp_enabled,
            cfg=cfg,
            epoch=epoch,
            max_iter=max_iter,
            teacher_model=teacher,
            return_loss_components=report_components,
        )

        val_loss, val_loss_components = evaluate_loss(
            model=student,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=amp_enabled,
            cfg=cfg,
            return_loss_components=report_components,
        )

        val_metrics = compute_segmentation_metrics(
            model=student,
            loader=val_loader,
            device=device,
            num_classes=cfg.num_classes,
            ignore_index=cfg.ignore_index,
        )

        dt = time.time() - t0
        train_loss = float(train_stats["loss"])
        val_miou = float(val_metrics["miou"])

        pred_hist = train_stats["pred_hist"]
        pred_total = int(pred_hist.sum().item())
        dominant_class = int(torch.argmax(pred_hist).item()) if pred_total > 0 else -1
        dominant_ratio = float(pred_hist.max().item() / pred_total) if pred_total > 0 else 0.0

        print(
            f"[EPOCH {epoch:03d}/{cfg.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_miou={val_miou:.4f} bf1={val_metrics['boundary_fscore']:.4f} "
            f"trimap_iou={val_metrics['trimap_iou']:.4f} lr={train_stats['last_lr']:.8f} "
            f"data_t={train_stats['avg_data_time']:.3f}s iter_t={train_stats['avg_compute_time']:.3f}s "
            f"grad_norm={train_stats['avg_grad_norm']:.4f} time={dt:.1f}s"
        )
        print(f"[TRAIN-PRED-HIST] counts={pred_hist.tolist()}")
        print(f"[TRAIN-PRED-HIST] dominant_class={dominant_class} dominant_ratio={dominant_ratio:.4f}")
        if dominant_ratio >= float(cfg.dominant_class_warn_ratio):
            print(f"[ALERT] dominant prediction ratio is high: class={dominant_class}, ratio={dominant_ratio:.4f}")

        iou_per_class = val_metrics["iou_per_class"]
        precision_per_class = val_metrics["precision_per_class"]
        recall_per_class = val_metrics["recall_per_class"]
        for class_id in range(cfg.num_classes):
            class_name = id2name.get(class_id, f"class_{class_id}")
            out.append_per_class_metrics(
                epoch=epoch,
                class_id=class_id,
                class_name=class_name,
                iou=float(iou_per_class[class_id]),
                precision=float(precision_per_class[class_id]),
                recall=float(recall_per_class[class_id]),
            )

        out.append_metrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=float(val_loss),
            val_miou=val_miou,
            val_bf1=float(val_metrics["boundary_fscore"]),
            lr=float(train_stats["last_lr"]),
            dt=dt,
        )
        _append_loss_components(out, epoch, "train", train_stats["loss_components"])
        _append_loss_components(out, epoch, "val", val_loss_components)

        if (not math.isnan(val_loss)) and val_loss < best_val_loss:
            best_val_loss = float(val_loss)

        ckpt = {
            "epoch": epoch,
            "model_state": student.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_miou": best_miou,
            "best_val_loss": best_val_loss,
        }

        if epoch % 10 == 0:
            torch.save(ckpt, out.ckpt_dir / f"epoch_{epoch:03d}.pth")

        if (not math.isnan(val_miou)) and (val_miou > best_miou):
            best_miou = val_miou
            ckpt["best_miou"] = best_miou
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, out.ckpt_dir / "best.pth")
            print(f"[INFO] New best mIoU: {best_miou:.4f}")

        _maybe_save_vis(student, val_loader, device, out, cfg, epoch)

    print("[DONE] Training completed.")


if __name__ == "__main__":
    main()
