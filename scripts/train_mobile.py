import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader, WeightedRandomSampler

from config.config import MobileTrainConfig
from src.commom.output_manager import OutputManager
from src.commom.repro import set_seed
from src.datasets.factory import apply_dataset_profile, build_dataset, normalize_dataset_name, resolve_dataset_root
from src.eval.mIoU import compute_segmentation_metrics
from src.losses.combined_loss import CrossEntropySegLoss, CombinedCEFocalLoss, OHEMBoundaryLoss, OHEMCELoss
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


def _build_rare_class_sampler(train_ds, cfg: MobileTrainConfig) -> WeightedRandomSampler:
    rare_ids = {int(cid) for cid in cfg.rare_class_ids if 0 <= int(cid) < int(cfg.num_classes)}
    weights = np.ones(len(train_ds.img_paths), dtype=np.float64)

    for idx, img_path in enumerate(train_ds.img_paths):
        mask_path = train_ds._resolve_mask(img_path)
        mapped = train_ds.load_mask_ids(mask_path)
        valid = mapped != train_ds.ignore_index
        present = set(np.unique(mapped[valid]).tolist()) if np.any(valid) else set()
        if any(cid in present for cid in rare_ids):
            weights[idx] *= float(cfg.rare_class_weight_multiplier)

    num_samples = max(1, int(len(train_ds) * float(cfg.sampler_num_samples_factor)))
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=num_samples,
        replacement=True,
    )


def _compute_class_weights(train_ds, cfg: MobileTrainConfig) -> torch.Tensor:
    counts = np.zeros(cfg.num_classes, dtype=np.float64)

    for img_path in train_ds.img_paths:
        mask_path = train_ds._resolve_mask(img_path)
        mapped = train_ds.load_mask_ids(mask_path)
        valid = (mapped != train_ds.ignore_index) & (mapped >= 0) & (mapped < cfg.num_classes)
        mapped = mapped[valid]
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
    if mode not in {"ce", "baseline", "ohem", "ohem_boundary"}:
        raise ValueError(f"Unsupported loss_mode={cfg.loss_mode}. Use ce/baseline/ohem/ohem_boundary")

    if mode == "ce":
        criterion = CrossEntropySegLoss(
            class_weights=class_weights,
            label_smoothing=cfg.label_smoothing,
            ignore_index=cfg.ignore_index,
        )
    elif mode == "baseline":
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

    warmup_iters = max(0, min(int(cfg.warmup_iters), max_iter - 1))
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
    if any(k.startswith("hybrid_neck.") for k in state.keys()):
        return "hybrid"
    return "aspp"


def _detect_teacher_hybrid_variant(state: dict[str, torch.Tensor]) -> str:
    if any(k.startswith("hybrid_neck.mid_kernel_branch.") or k == "hybrid_neck.mid_scale_logit" for k in state.keys()):
        return "large_v3"
    return "large"


def _detect_decoder_upsample_mode(state: dict[str, torch.Tensor]) -> str:
    if any(
        k.startswith("decoder.aspp_upsample.pre.") or k.startswith("decoder.aspp_upsample.post.")
        for k in state.keys()
    ):
        return "learnable"
    return "bilinear"


def _build_experiment_name(cfg: MobileTrainConfig) -> str:
    mode = "distill" if bool(cfg.use_distillation) else "baseline"
    exp_name = f"{normalize_dataset_name(cfg.dataset_name)}_deeplabv3plus_mobile_{mode}"
    if bool(cfg.use_distillation):
        exp_name += f"_{str(cfg.distill_type).lower()}"
    if bool(cfg.use_distillation) and str(cfg.segmentation_head).lower() == "hybrid":
        exp_name += f"_{str(cfg.hybrid_variant).lower()}"
        if bool(cfg.hybrid_use_strip):
            exp_name += "_strip"
    if str(cfg.decoder_upsample_mode).lower() == "bilinear":
        exp_name += "_bilinear"
    if not bool(cfg.use_aux_loss):
        exp_name += "_noaux"
    return exp_name


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
    teacher_hybrid_variant = _detect_teacher_hybrid_variant(state) if teacher_head == "hybrid" else "large"
    teacher_decoder_upsample_mode = _detect_decoder_upsample_mode(state)

    if teacher_arch == "resnet":
        teacher = DeepLabV3Plus(
            num_classes=cfg.num_classes,
            backbone_pretrained=cfg.distill_teacher_backbone_pretrained,
            backbone_name=cfg.distill_teacher_backbone_name,
            output_stride=cfg.distill_teacher_output_stride,
            segmentation_head=teacher_head,
            aspp_dropout=cfg.aspp_dropout,
            hybrid_variant=teacher_hybrid_variant,
            hybrid_use_strip=cfg.hybrid_use_strip,
            hybrid_strip_kernel=cfg.hybrid_strip_kernel,
            hybrid_mid_kernel=cfg.hybrid_mid_kernel,
            hybrid_large_kernel=cfg.hybrid_large_kernel,
            hybrid_gate_reduction=cfg.hybrid_gate_reduction,
            hybrid_residual_channels=cfg.hybrid_residual_channels,
            hybrid_residual_init=cfg.hybrid_residual_init,
            hybrid_dropout=cfg.hybrid_dropout,
            decoder_upsample_mode=teacher_decoder_upsample_mode,
            decoder_dropout=cfg.decoder_dropout,
        )
    else:
        teacher = DeepLabV3PlusMobile(
            num_classes=cfg.num_classes,
            output_stride=cfg.distill_teacher_output_stride,
            backbone_pretrained=False,
            aspp_dropout=cfg.aspp_dropout,
            decoder_upsample_mode=teacher_decoder_upsample_mode,
            decoder_dropout=cfg.decoder_dropout,
        )

    teacher.load_state_dict(state, strict=True)
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(
        f"[INFO] Teacher loaded: {teacher_ckpt} "
        f"(arch={teacher_arch}, stride={cfg.distill_teacher_output_stride}, "
        f"decoder_upsample_mode={teacher_decoder_upsample_mode})"
    )
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

    if cfg.use_aux_loss:
        aux_loss = criterion(student_aux, target)
    else:
        aux_loss = student_main.new_tensor(0.0)
    total = main_loss + float(cfg.aux_loss_weight) * aux_loss

    if components is not None:
        components["main"] = float(main_loss.detach())
        components["aux"] = float(aux_loss.detach())

    if teacher_main is not None and teacher_aux is not None:
        kd_main = _distill_loss(student_main, teacher_main, cfg)
        if cfg.use_aux_loss:
            kd_aux = _distill_loss(student_aux, teacher_aux, cfg)
            kd_total = kd_main + float(cfg.distill_aux_weight) * kd_aux
        else:
            kd_aux = student_main.new_tensor(0.0)
            kd_total = kd_main
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


@torch.inference_mode()
def save_vis_using_best_ckpt(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    out: OutputManager,
    cfg: MobileTrainConfig,
    id2color,
    epoch: int,
    best_ckpt_path: Path,
) -> None:
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
        out_dir=out.vis_dir,
        id2color=id2color,
        ignore_index=cfg.ignore_index,
        epoch=epoch,
        max_items=cfg.save_vis_max_items,
    )
    model.load_state_dict(cur_state, strict=True)


def main() -> None:
    cfg = MobileTrainConfig()
    apply_dataset_profile(cfg)
    cfg.data_root = resolve_dataset_root(cfg)
    set_seed(cfg.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    print(f"[INFO] device={device}, amp={amp_enabled}")
    print(f"[INFO] dataset_name={cfg.dataset_name}, data_root={cfg.data_root}")
    print(
        f"[INFO] decoder_upsample_mode={cfg.decoder_upsample_mode}, "
        f"use_aux_loss={cfg.use_aux_loss}, backbone_pretrained={cfg.backbone_pretrained}"
    )
    train_ds = build_dataset(cfg, split="train", training=True)
    cfg.num_classes = int(train_ds.meta.num_classes)
    val_ds = build_dataset(cfg, split="val", training=False)
    id2name = {idx: name for idx, name in enumerate(train_ds.meta.class_names)}
    id2color_vis = list(train_ds.meta.id2color)

    out = OutputManager(cfg.outputs_root, exp_name=_build_experiment_name(cfg))
    out.save_config(cfg)
    out.init_metrics()
    print(f"[INFO] run_dir={out.run_dir}")

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
        backbone_pretrained=cfg.backbone_pretrained,
        aspp_dropout=cfg.aspp_dropout,
        decoder_upsample_mode=cfg.decoder_upsample_mode,
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

    best_miou = -1.0
    best_epoch = 0
    best_val_loss = float("inf")
    report_components = str(cfg.loss_mode).lower().replace("+", "_") in {"ohem", "ohem_boundary"}

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

        if (not math.isnan(val_miou)) and (val_miou > best_miou):
            best_miou = val_miou
            best_epoch = epoch
            ckpt["best_miou"] = best_miou
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, out.ckpt_dir / "best.pth")
            print(f"[INFO] New best mIoU: {best_miou:.4f} (epoch={epoch:03d})")

    best_ckpt_path = out.ckpt_dir / "best.pth"
    if best_ckpt_path.exists():
        print(
            "[INFO] Saving final visualizations with best.pth "
            f"(best_epoch={best_epoch:03d}, save_epoch={cfg.epochs:03d}) ..."
        )
        save_vis_using_best_ckpt(
            model=student,
            val_loader=val_loader,
            device=device,
            out=out,
            cfg=cfg,
            id2color=id2color_vis,
            epoch=int(cfg.epochs),
            best_ckpt_path=best_ckpt_path,
        )
    else:
        print("[WARN] best.pth not found, skipping final visualization export.")

    print("[DONE] Training completed.")


if __name__ == "__main__":
    main()
