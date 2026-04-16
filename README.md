# GraduateDesign

这个仓库当前的论文主线已经收敛到三部分：

- 教师模型：`DeepLabV3+` 框架下的 `ASPP` 基线与 `Hybrid Large` 改进结构
- 学生模型：`MobileNetV2 + DeepLabV3+ decoder`，当前论文口径固定为 `no pretrain`
- 对照实验：传统方法 `SLIC + RandomForest`

当前代码已经移除了旧的 `OCR` 路线，分割头只保留：

- `aspp`
- `hybrid`

## 当前模型结构

### 教师模型

文件：`src/models/deeplabv3_plus.py`

支持两种 neck / head 路线：

1. `ASPP`
- 标准 `DeepLabV3+`
- 结构：`ResNetBackbone -> ASPP -> Decoder -> Classifier`

2. `Hybrid`
- 用 `HybridContextNeck` 替代纯 `ASPP`
- 结构：`ResNetBackbone -> HybridContextNeck -> Decoder -> Classifier`

### HybridContextNeck

文件：`src/models/hybrid_context.py`

当前保留的主要变体：

1. `large`
- `ASPP` 主分支
- `LargeKernelContextBranch` 大核上下文增强分支
- `ChannelGate` 通道门控
- 可学习残差缩放

2. `large_v3`
- 在 `large` 基础上增加 `mid kernel` 分支
- 用于更复杂的结构消融

可选分支：

- `strip`
- 通过横向 / 纵向条带卷积建模方向性上下文
- 当前只建议作为消融项，不建议作为最终主模型

### 学生模型

文件：`src/models/deeplabv3_plus_moblie.py`

结构：

- `MobileNetV2Backbone`
- `ASPP`
- `DeepLabV3PlusDecoder`
- `Classifier`

当前代码支持：

- `pretrained` 或 `no pretrain` 的 `MobileNetV2`
- `bilinear` 或 `learnable` decoder upsample
- `aux head`
- `KL` / `CWD` logits distillation

当前论文主线中，学生模型以 `no pretrain` 为准。

## 当前训练入口

### 教师训练

文件：`scripts/train.py`

特点：

- 训练和验证 loss 口径统一
- 训练过程中只维护 `best.pth`
- 训练结束后再加载最优权重做最终可视化
- 支持：
  - `CE`
  - `CE + Focal`（`loss_mode="baseline"`）
  - `OHEM`
  - `OHEM + Boundary`

### 学生训练 / 蒸馏

文件：`scripts/train_mobile.py`

特点：

- 支持 `student baseline`
- 支持 `KL` / `CWD` 蒸馏
- 蒸馏与 baseline 共享同一个学生结构
- 支持使用上采样前 logits 做蒸馏，以降低 `KL` 的尺度和显存压力
- 训练过程中只维护 `best.pth`
- 训练结束后再用最佳权重做最终可视化

### 其他脚本

- `scripts/eval_teacher_ckpt.py`：评估教师 checkpoint，可用于 KITTI / CamVid 泛化测试
- `scripts/train_slic_rf.py`：传统方法基线，`SLIC + RandomForest`

## 当前默认配置

文件：`config/config.py`

### 教师默认主线

```python
segmentation_head = "hybrid"
hybrid_variant = "large"
hybrid_use_strip = False

decoder_upsample_mode = "bilinear"
use_aux_loss = False
loss_mode = "ce"
use_class_weights = False
```

### 学生 baseline 默认主线

```python
output_stride = 16
backbone_pretrained = False
decoder_upsample_mode = "bilinear"
use_aux_loss = False
loss_mode = "ce"
use_class_weights = False
use_distillation = False
```

### 学生蒸馏默认主线

```python
output_stride = 16
backbone_pretrained = False
decoder_upsample_mode = "bilinear"
use_aux_loss = False
loss_mode = "ce"
use_class_weights = False

use_distillation = True
distill_teacher_ckpt = outputs/Teacher_Full_preprocess_classweights/checkpoints/best.pth
distill_type = "kl"
distill_temperature = 4.0
distill_loss_weight = 0.2
distill_aux_weight = 0.4
distill_use_preupsample = True
```

## 已完成实验

### 教师模型

核心可用实验：

- `Teacher_Baseline`：`ASPP` 基线，best `mIoU = 0.775216`
- `Teacher_Baseline_large`：`Hybrid Large`，best `mIoU = 0.782683`
- `Teacher_Baseline_large_v3`：`Hybrid Large V3`，best `mIoU = 0.781115`
- `Teacher_Baseline_large_learnable_aux`：best `mIoU = 0.781033`
- `Teacher_Baseline_large_v3_learnable_aux`：best `mIoU = 0.777274`
- `Teacher_Baseline_OHEM`：best `mIoU = 0.773165`
- `Teacher_No_preprocess`：best `mIoU = 0.750164`
- `Teacher_Full_preprocess`：best `mIoU = 0.773470`
- `Teacher_Full_Baseline_classweights`：best `mIoU = 0.771624`
- `Teacher_Full_preprocess_classweights`：best `mIoU = 0.783334`

探索性实验：

- `Teacher_Baseline_Hybrid_Both`
- `Teacher_Baseline_large_Gate`
- `Teacher_learnable_aux`

### 学生模型

当前论文口径下可直接使用的实验：

- `notrained_Student_Baseline`：scratch student baseline，best `mIoU = 0.649585`

需要谨慎处理的实验：

- `notrained_Student_cwd`：目录名显示为 `cwd`，但保存的 `config.json` 中 `use_distillation = false`，应视为另一组 scratch baseline 重复实验，不应当作正式蒸馏结果引用

历史参考实验：

- `Pretrained_Student_Baseline_1e-2Lr`
- `Pretrained_Student_cwd_1e-2Lr`
- `Pretrained_Student_cwd_5e-3Lr`

这些实验说明代码支持 `pretrained` 路线，但由于论文最终口径已经固定为 `no pretrain`，它们更适合作为早期探索记录，不建议作为主文主结果。

### 传统方法

- `SLIC_RF`：best `mIoU = 0.227116`

### 泛化评估

已完成：

- `Teacher_Full_preprocess_classweights` 在 `KITTI Semantic` 验证集上的评估
- 指标：`mIoU = 0.597338`, `BF1 = 0.654069`, `TrimapIoU = 0.661760`

当前仍缺：

- `Teacher_Baseline_large` 与 `Teacher_Full_preprocess_classweights` 的成对 KITTI 泛化对比

## 当前结论

1. 教师结构主线已经明确
- `Hybrid Large` 是当前最稳的教师结构
- 它相对 `ASPP` 基线有稳定提升

2. 更复杂结构没有带来稳定收益
- `large_v3` 没有稳定超过 `large`
- `Hybrid_Both` 和其他更复杂分支没有成为主线的理由

3. 训练技巧更偏向边界优化
- `learnable + aux` 往往提升 `BF1`
- `OHEM` 也更偏边界和难样本
- 但它们都没有稳定超过 `Teacher_Baseline_large`

4. 数据增强结论比较清楚
- 无预处理最差
- 基线预处理在 Cityscapes 本域最稳
- `Full preprocess + class weights` 拿到了当前保存结果中的最高分 `0.783334`
- 但它相对 `Teacher_Baseline_large` 的优势很小，因此论文中应区分“最佳结构”与“最佳训练组合”

5. 学生模型当前结论
- 学生论文主线已经固定为 `no pretrain`
- 当前可用的正式学生主结果是 `notrained_Student_Baseline`
- 还没有一组干净、可直接写入论文主文的 `no pretrain` 蒸馏结果

## 论文写作建议

推荐主文结构：

1. 教师模型结构改进
- `Teacher_Baseline`
- `Teacher_Baseline_large`
- `Teacher_Baseline_large_v3`

2. 训练策略与数据增强
- `Teacher_No_preprocess`
- `Teacher_Baseline_large`
- `Teacher_Full_preprocess`
- `Teacher_Full_preprocess_classweights`

3. 学生模型
- `notrained_Student_Baseline`
- 待补一组干净的 `KL` / `CWD` 蒸馏实验后再写蒸馏结论

4. 传统基线与泛化
- `SLIC_RF`
- KITTI 泛化评估

## 注意事项

1. 旧的 `OCR` checkpoint
- 当前代码已经不再支持

2. 旧的 `context_block`
- 已删除，不再保留兼容逻辑

3. 输出目录名不一定等于真实配置
- 优先以每个 run 目录下的 `config.json` 为准
- 已知示例：
  - `Teacher_Baseline_large_OHEM` 的目录命名与实际配置存在历史版本差异
  - `notrained_Student_cwd` 的目录名显示为 `cwd`，但保存配置中并未开启 distillation

4. 最新实验状态
- 教师主线：`Hybrid Large`
- 学生主线：`no pretrain`
- 蒸馏主线：待补一组干净的 `scratch + KL` 或 `scratch + CWD` 结果

更详细的实验整理可参考：`thesis_progress_summary.txt`
