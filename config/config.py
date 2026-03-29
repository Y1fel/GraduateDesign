from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    #目录
    data_root: Path = PROJECT_ROOT / "data"


    num_classes: int = 19
    ignore_index: int = 255

    #训练基本参数
    epochs: int = 50
    batch_size: int = 8
    num_workers: int = 12
    persistent_workers: bool = True
    prefetch_factor: int = 4

    #学习率调整
    lr_0: float = 0.002
    weight_decay: float = 1e-4     #权重衰减
    lr_eta_min: float = 5e-5
    lr_policy: str = "poly"        #学习率策略："poly"/"cosine"。
    poly_power: float = 0.9        #poly 调度幂指数，lr = lr0 * (1-progress)^poly_power。
    warmup_iters: int = 3000       #warmup_ratio*lr0 -> lr0。
    warmup_ratio: float = 0.1      #warmup起始比例

    #BN层相关策略
    freeze_bn: bool = False
    dominant_class_warn_ratio: float = 0.9

    #类别权重策略
    label_smoothing: float = 0.00
    use_class_weights: bool = True
    class_weight_strategy: str = "median_frequency"        #中值频率平衡
    # inverse-frequency 方案的幂指数（仅非 median_frequency 时生效）。
    class_weight_power: float = 0.6
    class_weight_min: float = 0.5                          #权重下限
    class_weight_max: float = 2.5                          #权重上限
    class_weight_rare_cap: float = 3.0                     #对极小类硬上限
    class_weight_boost_low_iou_every: int = 5              #每N个epoch检查
    class_weight_boost_low_iou_threshold: float = 0.2      #触发阈值
    class_weight_boost_factor: float = 1.05                #修改因子

    #稀有类感知采样器
    use_rare_class_sampler: bool = True
    rare_class_ids: tuple[int, ...] = (3,4,5,6,9,12,16,17)    #ID列表
    rare_class_weight_multiplier: float = 2.0                 #稀有类权重倍率
    sampler_num_samples_factor: float = 1.0                   #采样器抽样数量系数（len(dataset) * factor）

    #主干网络相关
    output_stride: int = 8                    #步长
    backbone_pretrained: bool = True          #backbone预训练
    backbone_name: str = "rsnet-100"          #50/100->50/101
    aspp_dropout: float = 0.05
    decoder_dropout: float = 0.15
    use_context_block: bool = True
    context_block_reduction: int = 4
    context_block_dilations: tuple[int, int] = (3, 6)
    context_block_dropout: float = 0.1
    aux_loss_weight: float = 0.3              #辅助损失权重

    #预处理
    crop_h: int = 768
    crop_w: int = 768                         #随即裁剪尺寸
    crop_retry: int = 10                      #重试次数
    crop_max_class_ratio: float = 0.75        #单类最大占比
    train_multi_scale_min: float = 0.5
    train_multi_scale_max: float = 2          #多尺度放缩比例
    hflip_prob: float = 0.5                   #水平翻转概率


    color_jitter_prob: float = 0.5            #颜色抖动概率
    color_jitter_brightness: float = 0.2      #亮度抖动幅度：[1-v, 1+v]。
    color_jitter_contrast: float = 0.2        #对比度抖动幅度
    color_jitter_saturation: float = 0.2      #饱和度抖动幅度
    gaussian_blur_prob: float = 0.2           #高斯模糊概率
    gaussian_blur_radius_min: float = 0.1
    gaussian_blur_radius_max: float = 1.3     #高斯模糊半径

    #训练输出和可视化
    save_vis_every: int = 20
    save_vis_max_items: int = 10
    outputs_root: Path = PROJECT_ROOT / "outputs"

    # loss_mode:
    # - "baseline":CE+Focal
    # - "ohem":OHEM
    # - "ohem_boundary":OHEM+boundary
    loss_mode: str = "ohem_boundary"

    #CE+Focal
    ce_weight: float = 0.75
    focal_weight: float = 0.25
    focal_gamma: float = 2.0

    #OHEM
    ohem_ratio: float = 0.25               #OHEM保留的困难像素比例
    ohem_weight: float = 1                 #OHEM权重
    boundary_weight: float = 0.15         #boundary_loss权重
    boundary_kernel_size: int = 3          #边界提取核尺寸
    report_loss_every: int = 5

    #其他
    seed: int = 42
    use_amp: bool = True

@dataclass
class MobileTrainConfig(TrainConfig):
    epochs: int = 90

    color_jitter_prob: float = 0.5
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    gaussian_blur_prob: float = 0.2
    gaussian_blur_radius_min: float = 0.1
    gaussian_blur_radius_max: float = 1.3

    class_weight_boost_factor: float = 1.1
    output_stride: int = 16 #32
    # 蒸馏训练开关。
    use_distillation: bool = True
    # 教师模型 checkpoint
    distill_teacher_ckpt: Path = PROJECT_ROOT / "outputs" / "best.pth"
    distill_teacher_arch: str = "resnet"
    distill_teacher_backbone_name: str = "rsnet-100"
    distill_teacher_backbone_pretrained: bool = False
    distill_teacher_output_stride: int = 8
    # 蒸馏损失类型：
    # - "cwd": Channel-wise Distillation
    # - "kl":  logits KL蒸馏
    distill_type: str = "cwd"
    # 蒸馏温度参数（KL/CWD 都会使用）。
    distill_temperature: float = 4.0
    distill_loss_weight: float = 0.7
    distill_aux_weight: float = 0.4
