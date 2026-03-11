from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    # 数据根目录，期望包含 Cityscapes 的 train/val 图像与标注结构。
    data_root: Path = PROJECT_ROOT / "data"
    # 训练类别数（Cityscapes 常用 19 类）。
    num_classes: int = 19
    # 忽略标签的像素值（不会参与 loss 与指标计算）。
    ignore_index: int = 255

    # 总训练 epoch 数。
    epochs: int = 50
    # 单卡 batch size。
    batch_size: int = 8
    # DataLoader 工作进程数。
    num_workers: int = 12
    # 是否持久化 worker，减少每个 epoch 重启开销。
    persistent_workers: bool = True
    # 每个 worker 预取 batch 数。
    prefetch_factor: int = 4

    # 初始学习率（iter 级调度的基准值）。
    lr_0: float = 0.002
    # 权重衰减（L2 正则）。
    weight_decay: float = 1e-4
    # 学习率下限（poly/cosine 调度最终不低于该值）。
    lr_eta_min: float = 5e-5
    # 学习率策略："poly" 或 "cosine"。
    lr_policy: str = "poly"
    # poly 调度幂指数，lr = lr0 * (1-progress)^poly_power。
    poly_power: float = 0.9
    # warmup 迭代数（iter 级），前期线性从 warmup_r atio*lr0 增长到 lr0。
    warmup_iters: int = 3000
    # warmup 起始比例。
    warmup_ratio: float = 0.1

    # BN 策略：小 batch 不稳定时可冻结 BN（仅在验证稳定性后启用）。
    freeze_bn: bool = False
    # 训练期预测分布告警阈值：若某一类占比超过该阈值，打印告警。
    dominant_class_warn_ratio: float = 0.9

    # CE 的 label smoothing 系数。
    label_smoothing: float = 0.00
    # 是否启用类别权重。
    use_class_weights: bool = True
    # 类别权重策略：
    # - "median_frequency": 中值频率平衡（推荐长尾场景）
    # - 其它值回退到 inverse-frequency^power 方案。
    class_weight_strategy: str = "median_frequency"
    # inverse-frequency 方案的幂指数（仅非 median_frequency 时生效）。
    class_weight_power: float = 0.6
    # 类别权重下限，避免过小权重导致类被忽略。
    class_weight_min: float = 0.5
    # 类别权重上限，避免过大权重导致梯度爆炸。
    class_weight_max: float = 2.5
    # 对极少类的额外硬上限（在 rare 类掩码上再裁剪一次）。
    class_weight_rare_cap: float = 3.0
    # 每隔 N 个 epoch 检查一次低 IoU 类别，并尝试提升其类别权重。
    class_weight_boost_low_iou_every: int = 5
    # 触发低 IoU 提升的阈值（如 <0.2）。
    class_weight_boost_low_iou_threshold: float = 0.2
    # 触发后权重乘法因子（如 1.15 代表 +15%）。
    class_weight_boost_factor: float = 1.05

    # 主干网络输出步长（常见 8/16）。
    output_stride: int = 8
    # 是否加载 backbone ImageNet 预训练权重。
    backbone_pretrained: bool = True
    # 主干网络类型：支持 "rsnet-50"、"rs-net-100"。
    backbone_name: str = "rsnet-100"
    # ASPP 模块 dropout。
    aspp_dropout: float = 0.05
    # Decoder 模块 dropout。
    decoder_dropout: float = 0.15
    # 辅助监督头（Aux Head）损失权重，总损失 = main + aux_loss_weight * aux。
    aux_loss_weight: float = 0.3

    # 随机裁剪尺寸（训练时）。
    crop_h: int = 768
    crop_w: int = 768
    # 随机裁剪重试次数（用于控制类别分布约束）。
    crop_retry: int = 10
    # 单类在 crop 中最大占比，超过时重采样（防止纯背景块）。
    crop_max_class_ratio: float = 0.75
    # 多尺度训练最小/最大缩放比例。
    train_multi_scale_min: float = 0.5
    train_multi_scale_max: float = 2
    # 水平翻转概率。
    hflip_prob: float = 0.5

    # 颜色抖动概率（仅训练集图像）。
    color_jitter_prob: float = 0.0
    # 亮度抖动幅度，最终系数范围约为 [1-v, 1+v]。
    color_jitter_brightness: float = 0.0
    # 对比度抖动幅度。
    color_jitter_contrast: float = 0.0
    # 饱和度抖动幅度。
    color_jitter_saturation: float = 0.0
    # 高斯模糊概率（仅训练集图像）。
    gaussian_blur_prob: float = 0.0
    # 高斯模糊半径最小/最大值。
    gaussian_blur_radius_min: float = 0.0
    gaussian_blur_radius_max: float = 0.0

    # 每 N epoch 产出一次预测可视化。
    save_vis_every: int = 20
    # 单次可视化保存样本上限。
    save_vis_max_items: int = 10
    # 训练输出根目录（日志、ckpt、可视化）。
    outputs_root: Path = PROJECT_ROOT / "outputs"

    # 全局随机种子（用于 1:1 对照实验）。
    seed: int = 42
    # 是否开启混合精度（仅 CUDA 下生效）。
    use_amp: bool = True

    # 是否启用稀有类感知采样器（WeightedRandomSampler）。
    use_rare_class_sampler: bool = True
    # 稀有类 ID 列表（按 19 类 id）。
    rare_class_ids: tuple[int, ...] = (3,4,5,6,9,12,16,17)
    # 样本含稀有类时的样本权重倍率。
    rare_class_weight_multiplier: float = 2.0
    # 采样器抽样数量系数（len(dataset) * factor）。
    sampler_num_samples_factor: float = 1.0


    # loss_mode:
    # - "baseline": 保持旧版 CE + Focal 组合，用于对照实验
    # - "ohem": 使用 OHEM CrossEntropy（仅保留困难像素）
    # - "ohem_boundary": 使用 OHEM + Boundary Loss 组合
    loss_mode: str = "ohem"

    # baseline 分支参数（CE + Focal）。
    ce_weight: float = 0.75
    focal_weight: float = 0.25
    focal_gamma: float = 2.0

    # ohem 分支参数：
    # OHEM 保留的困难像素比例
    ohem_ratio: float = 0.26
    # OHEM + Boundary 分支参数：
    # 总损失中 OHEM CE 的权重
    ohem_weight: float = 1
    # 总损失中 Boundary Loss 的权重
    boundary_weight: float = 0.05
    # 边界提取核尺寸（建议奇数）
    boundary_kernel_size: int = 3
    # 每 N epoch 打印一次 loss 子项统计（train/val）。
    report_loss_every: int = 5

@dataclass
class MobileTrainConfig(TrainConfig):
    output_stride: int = 16 #32
    # 蒸馏训练开关。
    use_distillation: bool = True
    # 教师模型 checkpoint（建议使用训练完成的 DeeplabV3+ best.pth）。
    distill_teacher_ckpt: Path = PROJECT_ROOT / "outputs" / "teacher_best.pth"
    # 教师模型结构参数。
    distill_teacher_backbone_name: str = "rsnet-100"
    distill_teacher_backbone_pretrained: bool = False
    distill_teacher_output_stride: int = 8
    # 蒸馏损失类型：
    # - "cwd": Channel-wise Distillation（默认，语义分割更常用）
    # - "kl":  传统 logits KL 蒸馏
    distill_type: str = "cwd"
    # 蒸馏温度参数（KL/CWD 都会使用）。
    distill_temperature: float = 4.0
    distill_loss_weight: float = 0.5
    distill_aux_weight: float = 0.3
