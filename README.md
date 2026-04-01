# GraduateDesign 改动说明

这份 README 只说明本轮针对教师模型训练链路做的改动，重点是：

- `ResNet101 + OCR` 头接入
- 保留 `ASPP` 作为消融开关
- 支持 `Cityscapes coarse -> fine` 两阶段训练
- 清理无收益的 `context_block`
- 收紧 checkpoint / 可视化保存逻辑

## 1. 这次改了什么

当前教师模型已经不是单一的 `DeepLabV3+ + ASPP` 固定结构，而是变成：

- 主干：`ResNet101`
- 分割头：`ASPP` 或 `OCR`
- 训练流程：`fine-only` 或 `coarse -> fine`

也就是说，现在可以通过配置组合出下面几类实验：

1. `ASPP + fine-only`
2. `ASPP + coarse->fine`
3. `OCR + fine-only`
4. `OCR + coarse->fine`

这就是后面做 ablation 的基础。

## 2. 模块说明

### `src/models/ocr.py`

这是这次新增的 OCR 头实现，包含：

- `SpatialGatherModule`
  - 用辅助分支的类别概率，把像素特征聚合成 object-level context。
- `ObjectAttentionBlock2D`
  - 用 object context 回流增强像素特征。
- `SpatialOCRModule`
  - 对 OCR 的上下文融合做最后输出。

这个文件的作用是把论文里的 OCR head 独立出来，避免把逻辑塞进训练脚本。

### `src/models/deeplabv3_plus.py`

这是教师模型主定义，现在支持两种 head：

- `segmentation_head="aspp"`
- `segmentation_head="ocr"`

行为说明：

- 当选择 `aspp` 时，走原来的 `ASPP -> decoder -> classifier`
- 当选择 `ocr` 时，走 `OCR pre -> gather -> OCR head -> decoder -> classifier`
- `aux_classifier` 仍然保留，用于辅助损失，也给 OCR 聚合上下文提供类别概率

注意：

- 我保留了 `self.aspp` 这个命名，这样旧的 ASPP checkpoint 更容易继续兼容
- `context_block` 相关逻辑已经彻底移除

### `src/datasets/cityscapes.py`

这个文件现在支持：

- `split="train"`
- `split="train_extra"`
- `split="val"`
- `split="test"`

并且新增了：

- `annotation_type="fine" | "coarse" | "auto"`

作用是让同一个数据集类同时兼容：

- `gtFine/train`
- `gtCoarse/train_extra`

自动规则：

- `train` 默认走 `fine`
- `train_extra` 默认走 `coarse`
- `val` 默认走 `fine`

因此 coarse-to-fine 不需要再写第二套数据集类。

### `scripts/train.py`

这是本轮改动最大的文件，主要做了 5 件事：

1. 训练/验证 loss 统一
   - `val_loss` 现在和 `train_loss` 一样，都包含 `main loss + aux loss`

2. 支持 phase 训练
   - 训练会按 phase 执行，而不是只认单一 `train`
   - 现在有两种 phase：
     - `coarse`
     - `fine`

3. 支持 OCR / ASPP 切换
   - 由配置里的 `segmentation_head` 控制

4. checkpoint 收紧
   - 不再每隔若干 epoch 额外存 `epoch_xxx.pth`
   - 只更新 `best.pth`

5. 可视化保存收紧
   - 不再每个 epoch 保存
   - 训练结束后只用 `best.pth` 导出一次可视化

此外，训练脚本里增加了几个辅助函数：

- `build_train_dataset`
- `build_val_dataset`
- `build_train_loader`
- `build_eval_loader`
- `build_teacher_model`
- `build_criterion`
- `build_optimizer`
- `build_training_phases`

目的就是把“模型构建”和“phase 切换”拆开，避免主函数继续膨胀。

### `config/config.py`

这里新增了本轮最重要的配置项：

- `segmentation_head`
  - `"aspp"` 或 `"ocr"`
- `ocr_mid_channels`
- `ocr_key_channels`
- `ocr_dropout`
- `use_coarse_to_fine`
- `coarse_epochs`

现在教师训练最重要的几个配置组合是：

```python
segmentation_head = "aspp"   # 或 "ocr"
use_coarse_to_fine = True    # 或 False
coarse_epochs = 30
```

### `scripts/tst_dataset.py`

这个脚本现在会自动判断教师 checkpoint 用的是哪种 head：

- 如果权重里有 `ocr_pre.*` / `ocr_head.*`，就按 OCR 教师模型加载
- 否则按 ASPP 教师模型加载

这样 OCR 教师训练完以后，不需要手工改推理脚本结构。

### `scripts/train_mobile.py`

虽然你这轮不重点跑学生模型，但我还是补了教师 checkpoint 的 head 自动识别：

- 蒸馏时加载教师模型，会自动判断教师是 `ASPP` 还是 `OCR`

否则后面 OCR 教师权重会直接在蒸馏入口报结构不匹配。

### `src/models/context_block.py`

这个模块已经删除。

原因很简单：

- 你明确说明它没有带来 mIoU 提升
- 它会干扰后面做结构消融
- 继续保留只会让 checkpoint、配置和实验矩阵更混乱

## 3. 现在怎么做消融

建议直接按下面 4 组实验命名：

1. `ASPP + fine-only`
2. `OCR + fine-only`
3. `ASPP + coarse->fine`
4. `OCR + coarse->fine`

配置切法如下：

### A. ASPP + fine-only

```python
segmentation_head = "aspp"
use_coarse_to_fine = False
```

### B. OCR + fine-only

```python
segmentation_head = "ocr"
use_coarse_to_fine = False
```

### C. ASPP + coarse->fine

```python
segmentation_head = "aspp"
use_coarse_to_fine = True
coarse_epochs = 30
```

### D. OCR + coarse->fine

```python
segmentation_head = "ocr"
use_coarse_to_fine = True
coarse_epochs = 30
```

如果你要进一步做更严格 ablation，建议固定下面这些不动：

- backbone：`rsnet-100`
- output stride：`8`
- augmentation
- loss mode
- batch size
- lr policy

否则最后结论会混在一起。

## 4. coarse -> fine 的数据依赖

如果要启用两阶段训练，目录至少要有：

```text
data/
  leftImg8bit/
    train/
    train_extra/
    val/
  gtFine/
    train/
    val/
  gtCoarse/
    train_extra/
```

没有 `train_extra` 或 `gtCoarse/train_extra` 时，`coarse` phase 会直接报找不到数据。

## 5. 输出行为

当前教师训练的输出逻辑是：

- 只维护一个 `best.pth`
- 训练结束后，使用 `best.pth` 导出一次最终可视化
- 输出目录名会带上 head 和 coarse2fine 信息，例如：
  - `cityscapes_deeplabv3plus_aspp`
  - `cityscapes_deeplabv3plus_ocr`
  - `cityscapes_deeplabv3plus_aspp_coarse2fine`
  - `cityscapes_deeplabv3plus_ocr_coarse2fine`

## 6. checkpoint 兼容性

### 旧 ASPP checkpoint

大体上仍然兼容，因为 ASPP 分支的参数命名我保留了。

### 旧 context_block checkpoint

不兼容。

因为：

- `context_block` 文件已经删掉
- `DeepLabV3Plus` 结构里也不再有对应参数

### 新 OCR checkpoint

训练、推理、蒸馏入口都已经补了自动识别逻辑，可以正常按 OCR 教师模型结构加载。

## 7. 推荐你接下来怎么跑

如果你的目标是尽快看见 mIoU 是否能上到 `0.81~0.82`，建议优先顺序：

1. `ASPP + fine-only`
   - 作为当前对照组
2. `OCR + fine-only`
   - 先看 head 本身有没有纯增益
3. `ASPP + coarse->fine`
   - 看 coarse data 是否单独有效
4. `OCR + coarse->fine`
   - 这是当前最值得期待的组合

如果最终要写实验结论，最重要的是拆出两件事：

- OCR 带来的提升
- coarse-to-fine 带来的提升

不要一开始只跑 `OCR + coarse->fine`，否则你最后很难说明增益是从哪来的。
