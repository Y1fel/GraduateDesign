# GraduateDesign

这个仓库当前的主线已经收敛到两部分：

- 教师模型：`DeepLabV3+` 框架下的 `ASPP` 基线和 `Hybrid Large` 改进版
- 学生模型：`MobileNetV2 + DeepLabV3+ decoder`，支持 baseline 和知识蒸馏

当前代码里已经不再保留 `OCR` 相关逻辑，分割头只支持：

- `aspp`
- `hybrid`

## 当前模型结构

### 教师模型

文件：[src/models/deeplabv3_plus.py](D:/MachineLearning/GraduateDesign/src/models/deeplabv3_plus.py)

支持两种 head：

1. `ASPP`
- 标准 `DeepLabV3+`
- 结构：`ResNetBackbone -> ASPP -> Decoder -> Classifier`

2. `Hybrid`
- 在 `ASPP` 基础上增加上下文增强 neck
- 结构：`ResNetBackbone -> HybridContextNeck -> Decoder -> Classifier`

### HybridContextNeck

文件：[src/models/hybrid_context.py](D:/MachineLearning/GraduateDesign/src/models/hybrid_context.py)

当前支持两种变体：

1. `large`
- `ASPP` 主分支
- `LargeKernelContextBranch` 大核上下文增强分支
- `ChannelGate` 做通道门控
- 可学习缩放系数控制残差注入强度

2. `large_v3`
- 在 `large` 基础上额外加入 `mid kernel` 分支
- 用于做更复杂的上下文增强消融

另外还保留了一个可选的 `strip` 分支：

- `hybrid_use_strip = True` 时启用
- 使用横向和纵向条带卷积建模方向性上下文
- 当前只建议用于消融，不建议作为最终主模型

### 学生模型

文件：[src/models/deeplabv3_plus_moblie.py](D:/MachineLearning/GraduateDesign/src/models/deeplabv3_plus_moblie.py)

结构：

- `MobileNetV2Backbone`
- `ASPP`
- `DeepLabV3PlusDecoder`
- `Classifier`

当前学生模型支持：

- `ImageNet pretrained MobileNetV2`
- `bilinear` 或 `learnable` decoder upsample
- `aux head`
- 蒸馏训练

## 当前训练入口

### 教师训练

文件：[scripts/train.py](D:/MachineLearning/GraduateDesign/scripts/train.py)

特点：

- 训练和验证 loss 口径已统一
- 训练中只维护 `best.pth`
- 训练结束后才加载最优权重做一次最终可视化
- 支持：
  - `CE`
  - `CE + Focal`（`loss_mode="baseline"`）
  - `OHEM`
  - `OHEM + Boundary`

### 学生训练 / 蒸馏

文件：[scripts/train_mobile.py](D:/MachineLearning/GraduateDesign/scripts/train_mobile.py)

特点：

- 支持 student baseline
- 支持 `KL` / `CWD` logits distillation
- 训练中只维护 `best.pth`
- 训练结束后才用最佳权重做最终可视化
- 会自动根据教师 checkpoint 检测：
  - `ASPP` 还是 `Hybrid`
  - `large` 还是 `large_v3`
  - `bilinear` 还是 `learnable`

## 当前配置主线

文件：[config/config.py](D:/MachineLearning/GraduateDesign/config/config.py)

### 教师当前常用配置

最终教师主线通常是：

```python
segmentation_head = "hybrid"
hybrid_variant = "large"
hybrid_use_strip = False

decoder_upsample_mode = "bilinear"
use_aux_loss = False
loss_mode = "ce"
```

### 学生当前常用配置

学生 baseline / 蒸馏主线通常是：

```python
output_stride = 16
backbone_pretrained = True
decoder_upsample_mode = "bilinear"
use_aux_loss = False
loss_mode = "ce"
```

## 当前已经完成的重要实验

根据 `outputs` 目录，已经完成的核心实验包括：

- `Teacher_Baseline`
- `Teacher_Baseline_large`
- `Teacher_Baseline_large_v3`
- `Teacher_Baseline_large_learnable_aux`
- `Teacher_Baseline_OHEM`
- `Teacher_Baseline_large_OHEM`
- `Teacher_No_preprocess`
- `Teacher_Full_preprocess`
- `Teacher_Full_Baseline_classweights`
- `Teacher_Full_preprocess_classweights`
- `Student_Baseline`
- `Student_cwd`

这些实验的详细结论已经整理在：

- [thesis_progress_summary.txt](D:/MachineLearning/GraduateDesign/thesis_progress_summary.txt)

## 当前实验结论

1. 教师最终主模型
- 当前最稳的主结构是 `Hybrid Large`

2. 复杂结构
- `large_v3` 没有稳定超过 `large`
- `strip` 更适合做消融，不适合作为最终方案

3. 训练技巧
- `learnable + aux` 更偏边界增强
- `OHEM` 更偏边界和难样本，但整体 `mIoU` 不如 `CE` 主线稳定

4. 预处理
- 无预处理明显最差
- 基线预处理在本域 Cityscapes 上最稳
- 充分预处理在部分设置下可能带来更好的综合结果和更强泛化潜力

5. 学生模型
- 学生 baseline 已完成
- `CWD` 蒸馏已完成并带来小幅 `mIoU` 提升
- 后续还可以继续做 `KL` 和 feature distillation

## 当前推荐实验顺序

1. `Student + KL`
2. `Teacher / Student` 复杂度指标统计（params, FLOPs, latency）
3. `Teacher_Baseline_large` 与 `Teacher_Full_preprocess_classweights` 的泛化对比
4. 若时间允许，再做 feature distillation

## 注意事项

1. 旧的 `OCR` checkpoint
- 现在已经不再受支持
- 如果以前训练过 OCR 权重，当前代码不会再按 OCR 结构构建模型

2. 旧的 `context_block`
- 已经删除，不再保留兼容逻辑

3. 学生和教师配置
- 当前会共用一部分基础字段
- 但学生配置在 `MobileTrainConfig` 里有单独覆盖项

4. 最新实验总结
- 优先以 [thesis_progress_summary.txt](D:/MachineLearning/GraduateDesign/thesis_progress_summary.txt) 为准
