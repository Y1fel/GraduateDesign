import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn

from src.commom.output_manager import OutputManager
from src.commom.repro import set_seed
from src.datasets.factory import apply_dataset_profile, build_dataset, normalize_dataset_name, resolve_dataset_root
from src.eval.mIoU import compute_segmentation_metrics
from src.models.deeplabv3_plus import DeepLabV3Plus
from src.viz.visualizer import save_predictions_triplet
from config.config import TrainConfig
from src.losses.combined_loss import CrossEntropySegLoss, CombinedCEFocalLoss, OHEMBoundaryLoss, OHEMCELoss

#冻结BN层
def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.eval()
            if m.weight is not None:
                m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.requires_grad = False


def accumulate_pred_hist(pred: torch.Tensor, hist: torch.Tensor, num_classes: int) -> None:
    bins = torch.bincount(pred.view(-1), minlength=num_classes)
    hist += bins.to(hist.device, dtype=hist.dtype)


def compute_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().float().norm(2).item()
            total += g * g
    return total ** 0.5


def build_rare_class_sampler(train_ds, cfg: TrainConfig) -> WeightedRandomSampler:
    rare_ids = {int(cid) for cid in cfg.rare_class_ids if 0 <= int(cid) < int(cfg.num_classes)}
    weights = np.ones(len(train_ds.img_paths), dtype=np.float64)

    for idx, img_path in enumerate(train_ds.img_paths):
        mask_path = train_ds._resolve_mask(img_path)
        mapped = train_ds.load_mask_ids(mask_path)
        valid = mapped != train_ds.ignore_index
        present = set(np.unique(mapped[valid]).tolist()) if np.any(valid) else set()
        if any(cid in present for cid in rare_ids):
            weights[idx] *= float(cfg.rare_class_weight_multiplier)

    num_samples = int(len(train_ds) * float(cfg.sampler_num_samples_factor))
    num_samples = max(1, num_samples)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=num_samples,
        replacement=True,
    )


def compute_class_weights(train_ds, cfg: TrainConfig, strategy_override: str | None = None) -> torch.Tensor:
    counts = np.zeros(cfg.num_classes, dtype=np.float64)

    for img_path in train_ds.img_paths:
        mask_path = train_ds._resolve_mask(img_path)
        mapped = train_ds.load_mask_ids(mask_path)
        valid = (mapped != train_ds.ignore_index) & (mapped >= 0) & (mapped < cfg.num_classes)
        mapped = mapped[valid]
        binc = np.bincount(mapped, minlength=cfg.num_classes)
        counts += binc.astype(np.float64)

    counts = np.maximum(counts, 1.0)
    freq = counts / counts.sum()
    strategy = str(strategy_override or cfg.class_weight_strategy).lower()
    if strategy == "median_frequency":
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



def boost_low_iou_class_weights(
    criterion: nn.Module,
    iou_per_class: list[float],
    cfg: TrainConfig,
) -> None:
    if not hasattr(criterion, "class_weights") or criterion.class_weights is None:
        return
    weights = criterion.class_weights.detach().clone()
    boosted = False
    for idx, iou in enumerate(iou_per_class):
        if (not math.isnan(float(iou))) and float(iou) < float(cfg.class_weight_boost_low_iou_threshold):
            weights[idx] = min(
                weights[idx] * float(cfg.class_weight_boost_factor),
                float(cfg.class_weight_rare_cap),
            )
            boosted = True
    if boosted:
        if hasattr(criterion, "update_class_weights"):
            criterion.update_class_weights(weights.to(criterion.class_weights.device))
        else:
            criterion.class_weights.copy_(weights.to(criterion.class_weights.device))
        print(f"[INFO] boosted class_weights={criterion.class_weights.detach().cpu().tolist()}")


def compute_lr_at_iter(global_iter: int, max_iter: int, cfg: TrainConfig) -> float:
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}")

    warmup_iters = max(0, min(int(cfg.warmup_iters), max_iter - 1))
    warmup_ratio = float(cfg.warmup_ratio)
    base_lr = float(cfg.lr_0)
    eta_min = float(cfg.lr_eta_min)
    policy = str(cfg.lr_policy).lower()

    if warmup_iters > 0 and global_iter < warmup_iters:
        alpha = float(global_iter + 1) / float(warmup_iters)
        warmup_scale = warmup_ratio + (1.0 - warmup_ratio) * alpha
        return base_lr * warmup_scale

    progress_iter = global_iter - warmup_iters
    progress_total = max(1, max_iter - warmup_iters)
    progress = min(max(progress_iter / progress_total, 0.0), 1.0)

    if policy == "poly":
        lr = base_lr * ((1.0 - progress) ** float(cfg.poly_power))
        return max(eta_min, lr)
    if policy == "cosine":
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = eta_min + (base_lr - eta_min) * cosine
        return max(eta_min, lr)

    raise ValueError(f"Unsupported lr_policy: {cfg.lr_policy}. Use 'poly' or 'cosine'.")


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = float(lr)

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    use_amp: bool,
    num_classes: int,
    freeze_bn_enabled: bool,
    cfg: TrainConfig,
    global_iter_start: int,
    max_iter: int,
    return_loss_components: bool = False,
) -> dict:
    model.train()
    if freeze_bn_enabled:
        freeze_bn(model)
    total_loss, n = 0.0, 0
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    pred_hist = torch.zeros(num_classes, dtype=torch.int64, device=device)
    grad_norm_sum = 0.0
    grad_steps = 0
    data_time_sum = 0.0
    iter_compute_time_sum = 0.0
    loop_t = time.perf_counter()
    loss_components_sum = None

    for _it, (imgs, masks, _names) in enumerate(loader):
        global_iter = global_iter_start + _it
        lr_now = compute_lr_at_iter(global_iter=global_iter, max_iter=max_iter, cfg=cfg)
        set_optimizer_lr(optimizer, lr_now)

        data_time_sum += time.perf_counter() - loop_t
        iter_t = time.perf_counter()
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits, aux_logits = model(imgs, return_aux=True)

            if return_loss_components:
                main_loss, loss_components = criterion(logits, masks, return_components=True)
                loss_components = dict(loss_components)
            else:
                main_loss = criterion(logits, masks)
                loss_components = None

            if cfg.use_aux_loss:
                aux_loss = torch.nn.functional.cross_entropy(
                    aux_logits,
                    masks,
                    ignore_index=cfg.ignore_index,
                )
            else:
                aux_loss = logits.new_tensor(0.0)

            loss = main_loss + float(cfg.aux_loss_weight) * aux_loss

            if loss_components is not None:
                loss_components["main"] = float(main_loss.detach())
                loss_components["aux"] = float(aux_loss.detach())
                loss_components["total"] = float(loss.detach())

        scaler.scale(loss).backward()
        if use_amp:
            scaler.unscale_(optimizer)
        grad_norm_sum += compute_grad_norm(model)
        grad_steps += 1
        scaler.step(optimizer)
        scaler.update()

        pred = torch.argmax(logits.detach(), dim=1)
        accumulate_pred_hist(pred, pred_hist, num_classes=num_classes)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        if loss_components is not None:
            if loss_components_sum is None:
                loss_components_sum = {k: 0.0 for k in loss_components}
            for k in loss_components:
                loss_components_sum[k] += float(loss_components[k]) * bs
        n += bs
        iter_compute_time_sum += time.perf_counter() - iter_t
        loop_t = time.perf_counter()

    return {
        "loss": total_loss / max(n, 1),
        "pred_hist": pred_hist.detach().cpu(),
        "avg_grad_norm": grad_norm_sum / max(grad_steps, 1),
        "avg_data_time": data_time_sum / max(grad_steps, 1),
        "avg_compute_time": iter_compute_time_sum / max(grad_steps, 1),
        "last_lr": optimizer.param_groups[0]["lr"],
        "iters": grad_steps,
        "loss_components": (
            {k: v / max(n, 1) for k, v in loss_components_sum.items()}
            if return_loss_components and loss_components_sum is not None
            else None
        ),
    }


@torch.inference_mode()
def evaluate_loss(
    model,
    loader,
    criterion,
    device,
    use_amp: bool,
    cfg: TrainConfig,
    return_loss_components: bool = False,
):
    model.eval()
    total_loss = 0.0
    n = 0
    loss_components_sum = None

    for imgs, masks, _names in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits, aux_logits = model(imgs, return_aux=True)
            if return_loss_components:
                main_loss, loss_components = criterion(logits, masks, return_components=True)
                loss_components = dict(loss_components)
            else:
                main_loss = criterion(logits, masks)
                loss_components = None
            if cfg.use_aux_loss:
                aux_loss = torch.nn.functional.cross_entropy(
                    aux_logits,
                    masks,
                    ignore_index=cfg.ignore_index,
                )
            else:
                aux_loss = logits.new_tensor(0.0)
            loss = main_loss + float(cfg.aux_loss_weight) * aux_loss

            if loss_components is not None:
                loss_components["main"] = float(main_loss.detach())
                loss_components["aux"] = float(aux_loss.detach())
                loss_components["total"] = float(loss.detach())

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        if loss_components is not None:
            if loss_components_sum is None:
                loss_components_sum = {k: 0.0 for k in loss_components}
            for k in loss_components:
                loss_components_sum[k] += float(loss_components[k]) * bs
        n += bs

    avg_loss = total_loss / max(n, 1)
    if return_loss_components:
        if loss_components_sum is None:
            return avg_loss, None
        return avg_loss, {k: v / max(n, 1) for k, v in loss_components_sum.items()}
    return avg_loss


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


def build_train_dataset(cfg: TrainConfig):
    return build_dataset(cfg, split="train", training=True)


def build_val_dataset(cfg: TrainConfig):
    return build_dataset(cfg, split="val", training=False)


def build_train_loader(train_ds, cfg: TrainConfig, device: torch.device) -> DataLoader:
    train_sampler = None
    train_shuffle = True
    if cfg.use_rare_class_sampler:
        print(
            "[INFO] Building rare-class-aware sampler "
            f"(classes={cfg.rare_class_ids}, multiplier={cfg.rare_class_weight_multiplier:.2f})"
        )
        train_sampler = build_rare_class_sampler(train_ds, cfg)
        train_shuffle = False

    return DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=bool(cfg.persistent_workers and cfg.num_workers > 0),
        prefetch_factor=(cfg.prefetch_factor if cfg.num_workers > 0 else None),
        drop_last=True,
    )


def build_eval_loader(val_ds, cfg: TrainConfig, device: torch.device) -> DataLoader:
    return DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=bool(cfg.persistent_workers and cfg.num_workers > 0),
        prefetch_factor=(cfg.prefetch_factor if cfg.num_workers > 0 else None),
        drop_last=False,
    )


def build_experiment_name(cfg: TrainConfig) -> str:
    dataset_name = normalize_dataset_name(cfg.dataset_name)
    exp_name = f"{dataset_name}_deeplabv3plus_{str(cfg.segmentation_head).lower()}"
    if str(cfg.segmentation_head).lower() == "hybrid":
        exp_name += f"_{str(cfg.hybrid_variant).lower()}"
        if bool(cfg.hybrid_use_strip):
            exp_name += "_strip"
    if str(cfg.decoder_upsample_mode).lower() == "bilinear":
        exp_name += "_bilinear"
    if not bool(cfg.use_aux_loss):
        exp_name += "_noaux"
    return exp_name


def build_teacher_model(cfg: TrainConfig) -> DeepLabV3Plus:
    return DeepLabV3Plus(
        num_classes=cfg.num_classes,
        backbone_pretrained=cfg.backbone_pretrained,
        backbone_name=cfg.backbone_name,
        output_stride=cfg.output_stride,
        segmentation_head=cfg.segmentation_head,
        aspp_dropout=cfg.aspp_dropout,
        hybrid_variant=cfg.hybrid_variant,
        hybrid_use_strip=cfg.hybrid_use_strip,
        hybrid_strip_kernel=cfg.hybrid_strip_kernel,
        hybrid_mid_kernel=cfg.hybrid_mid_kernel,
        hybrid_large_kernel=cfg.hybrid_large_kernel,
        hybrid_gate_reduction=cfg.hybrid_gate_reduction,
        hybrid_residual_channels=cfg.hybrid_residual_channels,
        hybrid_residual_init=cfg.hybrid_residual_init,
        hybrid_dropout=cfg.hybrid_dropout,
        ocr_mid_channels=cfg.ocr_mid_channels,
        ocr_key_channels=cfg.ocr_key_channels,
        ocr_dropout=cfg.ocr_dropout,
        decoder_upsample_mode=cfg.decoder_upsample_mode,
        decoder_dropout=cfg.decoder_dropout,
    )


def build_criterion(
    train_ds,
    cfg: TrainConfig,
    device: torch.device,
    loss_mode: str,
) -> nn.Module:
    class_weight_strategy = "power_inverse" if loss_mode == "baseline" else None
    class_weights = (
        compute_class_weights(train_ds, cfg, strategy_override=class_weight_strategy).to(device)
        if cfg.use_class_weights
        else None
    )

    if loss_mode == "ce":
        criterion = CrossEntropySegLoss(
            class_weights=class_weights,
            label_smoothing=cfg.label_smoothing,
            ignore_index=cfg.ignore_index,
        )
    elif loss_mode == "baseline":
        criterion = CombinedCEFocalLoss(
            ce_weight=cfg.ce_weight,
            focal_weight=cfg.focal_weight,
            focal_gamma=cfg.focal_gamma,
            class_weights=class_weights,
            label_smoothing=cfg.label_smoothing,
            ignore_index=cfg.ignore_index,
        )
    elif loss_mode == "ohem":
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

    criterion = criterion.to(device)
    if cfg.use_class_weights and hasattr(criterion, "class_weights") and criterion.class_weights is not None:
        print(f"[INFO] class_weights={criterion.class_weights.detach().cpu().tolist()}")
    return criterion


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr_0,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
        nesterov=True,
    )
    print(
        f"[INFO] Optimizer = SGD (lr_0={cfg.lr_0:.2e}, momentum=0.9, "
        f"nesterov=True, weight_decay={cfg.weight_decay:.2e})"
    )
    return optimizer


def main() -> None:
    cfg = TrainConfig()
    apply_dataset_profile(cfg)
    cfg.data_root = resolve_dataset_root(cfg)
    set_seed(cfg.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    print(f"[INFO] AMP enabled = {amp_enabled}")
    print(f"[INFO] dataset_name = {cfg.dataset_name}")
    print(f"[INFO] data_root = {cfg.data_root}")
    print(f"[INFO] segmentation_head = {cfg.segmentation_head}")
    print(f"[INFO] decoder_upsample_mode = {cfg.decoder_upsample_mode}")
    print(f"[INFO] use_aux_loss = {cfg.use_aux_loss}")

    train_ds = build_train_dataset(cfg)
    cfg.num_classes = int(train_ds.meta.num_classes)
    val_ds = build_val_dataset(cfg)
    id2name = {idx: name for idx, name in enumerate(train_ds.meta.class_names)}
    id2color_vis = list(train_ds.meta.id2color)

    exp_name = build_experiment_name(cfg)
    out = OutputManager(cfg.outputs_root, exp_name=exp_name)
    out.save_config(cfg)
    out.init_metrics()
    print(f"[INFO] run_dir = {out.run_dir}")

    train_loader = build_train_loader(train_ds, cfg, device)
    val_loader = build_eval_loader(val_ds, cfg, device)

    model = build_teacher_model(cfg).to(device)

    loss_mode = str(cfg.loss_mode).lower()
    if loss_mode == "ohem+boundary":
        loss_mode = "ohem_boundary"
    if loss_mode not in {"ce", "baseline", "ohem", "ohem_boundary"}:
        raise ValueError(f"Unsupported loss_mode: {cfg.loss_mode}. Use 'ce', 'baseline', 'ohem' or 'ohem_boundary'.")
    print(f"[INFO] loss_mode={cfg.loss_mode}")
    policy = str(cfg.lr_policy).lower()
    if policy not in {"poly", "cosine"}:
        raise ValueError(f"Unsupported lr_policy: {cfg.lr_policy}. Use 'poly' or 'cosine'.")
    print(f"[INFO] norm_strategy=freeze_bn={cfg.freeze_bn}")
    criterion = build_criterion(train_ds, cfg, device, loss_mode)
    optimizer = build_optimizer(model, cfg)
    max_iter = int(cfg.epochs) * len(train_loader)
    warmup_iters = max(0, min(int(cfg.warmup_iters), max_iter - 1))
    print(
        f"[INFO] training_split=train epochs={cfg.epochs} train_size={len(train_ds)} max_iter={max_iter}"
    )
    print(
        "[INFO] LR scheduler = "
        f"{policy} (max_iter={max_iter}, warmup_iters={warmup_iters}, "
        f"warmup_ratio={cfg.warmup_ratio:.3f}, poly_power={cfg.poly_power:.3f}, eta_min={cfg.lr_eta_min:.2e})"
    )

    best_miou = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, int(cfg.epochs) + 1):
        if device.type == "cuda":
            torch.cuda.empty_cache()
        t0 = time.time()

        global_iter_start = (epoch - 1) * len(train_loader)
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            use_amp=amp_enabled,
            num_classes=cfg.num_classes,
            freeze_bn_enabled=cfg.freeze_bn,
            cfg=cfg,
            global_iter_start=global_iter_start,
            max_iter=max_iter,
            return_loss_components=(loss_mode in {"ohem", "ohem_boundary"}),
        )
        train_loss = float(train_stats["loss"])
        if loss_mode in {"ce", "baseline"}:
            val_loss = evaluate_loss(
                model,
                val_loader,
                criterion,
                device,
                use_amp=amp_enabled,
                cfg=cfg,
            )
            val_loss_components = None
        else:
            val_loss, val_loss_components = evaluate_loss(
                model,
                val_loader,
                criterion,
                device,
                use_amp=amp_enabled,
                cfg=cfg,
                return_loss_components=True,
            )

        val_metrics = compute_segmentation_metrics(
            model,
            val_loader,
            device,
            cfg.num_classes,
            cfg.ignore_index,
        )

        val_miou = float(val_metrics["miou"])
        recall_per_class = val_metrics["recall_per_class"]
        effective_mask = [not math.isnan(float(v)) for v in recall_per_class]
        effective_ious = [
            float(val_metrics["iou_per_class"][i])
            for i, ok in enumerate(effective_mask)
            if ok and not math.isnan(float(val_metrics["iou_per_class"][i]))
        ]
        val_miou_effective = float(sum(effective_ious) / len(effective_ious)) if effective_ious else float("nan")

        iou_per_class = val_metrics["iou_per_class"]
        precision_per_class = val_metrics["precision_per_class"]
        dt = time.time() - t0
        pred_hist = train_stats["pred_hist"]
        pred_total = int(pred_hist.sum().item())
        dominant_ratio = float(pred_hist.max().item() / pred_total) if pred_total > 0 else 0.0
        dominant_class = int(torch.argmax(pred_hist).item()) if pred_total > 0 else -1

        print(
            f"[EPOCH {epoch:03d}/{cfg.epochs}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_mIoU={val_miou:.4f} val_mIoU(effective)={val_miou_effective:.4f} "
            f"val_BF1={val_metrics['boundary_fscore']:.4f} val_TrimapIoU={val_metrics['trimap_iou']:.4f} "
            f"data_t={train_stats['avg_data_time']:.3f}s iter_t={train_stats['avg_compute_time']:.3f}s "
            f"grad_norm(avg)={train_stats['avg_grad_norm']:.4f} lr={float(train_stats['last_lr']):.8f} time={dt:.1f}s"
        )
        print(f"[TRAIN-PRED-HIST] counts={pred_hist.tolist()}")
        print(
            f"[TRAIN-PRED-HIST] dominant_class={dominant_class} dominant_ratio={dominant_ratio:.4f} "
            f"warn_threshold={cfg.dominant_class_warn_ratio:.2f}"
        )
        if dominant_ratio >= cfg.dominant_class_warn_ratio:
            print(
                "[ALERT] Predicted class distribution is highly imbalanced: "
                f"class={dominant_class}, ratio={dominant_ratio:.4f}."
            )
        print("[PER-CLASS] class_id class_name iou precision recall")
        per_class_rows = []
        for class_id in range(cfg.num_classes):
            class_name = id2name.get(class_id, id2name.get(str(class_id), f"class_{class_id}"))
            iou_val = float(iou_per_class[class_id])
            precision_val = float(precision_per_class[class_id])
            recall_val = float(recall_per_class[class_id])
            print(
                f"[PER-CLASS] {class_id:02d} {class_name:<18} "
                f"iou={iou_val:.4f} precision={precision_val:.4f} recall={recall_val:.4f}"
            )
            per_class_rows.append((class_id, class_name, iou_val, precision_val, recall_val))

        if (
            loss_mode != "baseline"
            and (epoch % max(1, int(cfg.report_loss_every)) == 0)
            and train_stats["loss_components"] is not None
            and val_loss_components is not None
        ):
            trc = train_stats["loss_components"]
            vac = val_loss_components
            tr_boundary = float(trc.get("boundary", 0.0))
            va_boundary = float(vac.get("boundary", 0.0))
            print(
                "[LOSS-COMP] "
                f"epoch={epoch:03d} train(ohem={trc['ohem_ce']:.4f}, boundary={tr_boundary:.4f}, total={trc['total']:.4f}) "
                f"val(ohem={vac['ohem_ce']:.4f}, boundary={va_boundary:.4f}, total={vac['total']:.4f})"
            )
            out.append_loss_components(epoch, "train", trc["ohem_ce"], tr_boundary, trc["total"])
            out.append_loss_components(epoch, "val", vac["ohem_ce"], va_boundary, vac["total"])

        valid_rows = [row for row in per_class_rows if not math.isnan(row[2])]
        if valid_rows:
            bottom_k = min(5, len(valid_rows))
            bottom_rows = sorted(valid_rows, key=lambda row: row[2])[:bottom_k]
            print(f"[PER-CLASS][BOTTOM-{bottom_k}] Lowest IoU classes:")
            for class_id, class_name, iou_val, precision_val, recall_val in bottom_rows:
                print(
                    f"[PER-CLASS][BOTTOM] {class_id:02d} {class_name:<18} "
                    f"iou={iou_val:.4f} precision={precision_val:.4f} recall={recall_val:.4f}"
                )

        if cfg.use_class_weights and (epoch % max(1, int(cfg.class_weight_boost_low_iou_every)) == 0):
            boost_low_iou_class_weights(criterion, list(iou_per_class), cfg)

        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[MEM] peak_allocated = {peak:.2f} GB")

        out.append_metrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_miou=val_miou_effective,
            val_bf1=float(val_metrics["boundary_fscore"]),
            lr=float(train_stats["last_lr"]),
            dt=dt,
        )
        for class_id, class_name, iou_val, precision_val, recall_val in per_class_rows:
            out.append_per_class_metrics(
                epoch=epoch,
                class_id=class_id,
                class_name=class_name,
                iou=iou_val,
                precision=precision_val,
                recall=recall_val,
            )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_miou": best_miou,
            "best_val_loss": best_val_loss,
        }

        if (not math.isnan(val_loss)) and (val_loss < best_val_loss):
            best_val_loss = val_loss

        if (not math.isnan(val_miou_effective)) and (val_miou_effective > best_miou):
            best_miou = val_miou_effective
            best_epoch = epoch
            ckpt["best_miou"] = best_miou
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, out.ckpt_dir / "best.pth")
            print(
                f"[INFO] New best mIoU = {best_miou:.4f} -> saved best.pth "
                f"(epoch={epoch:03d}, current val_loss={val_loss:.4f})"
            )

        print(f"... lr={float(train_stats['last_lr']):.8f}")

    best_ckpt_path = out.ckpt_dir / "best.pth"
    if best_ckpt_path.exists():
        print(
            "[INFO] Saving final visualizations with best.pth "
            f"(best_epoch={best_epoch:03d}, save_epoch={cfg.epochs:03d}) ..."
        )
        save_vis_using_best_ckpt(
            model=model,
            val_loader=val_loader,
            device=device,
            out_dir=out.vis_dir,
            id2color=id2color_vis,
            ignore_index=cfg.ignore_index,
            epoch=int(cfg.epochs),
            max_items=cfg.save_vis_max_items,
            best_ckpt_path=best_ckpt_path,
        )
    else:
        print("[WARN] best.pth not found, skipping final visualization export.")

    print("[DONE] Training finished.")


if __name__ == "__main__":
    main()
