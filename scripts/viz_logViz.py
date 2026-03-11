import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 1. 基础设置与数据读取 ----------------------
# 设置中文字体（避免中文乱码）
plt.rcParams["font.sans-serif"] = ["SimHei"]
#plt.rcParams["font.sans-serif"] = ["SimSun"]

plt.rcParams['axes.unicode_minus'] = False

# 读取三个数据文件
loss_df = pd.read_csv('D:\MachineLearning\GraduateDesign\outputs\cityscapes_deeplabv3plus_20260308_193910\logs\metrics.csv')    # 损失数据
metrics_df = pd.read_csv('D:\MachineLearning\GraduateDesign\outputs\cityscapes_deeplabv3plus_20260308_193910\logs\metrics.csv')         # 整体指标数据
per_class_df = pd.read_csv('D:\MachineLearning\GraduateDesign\outputs\cityscapes_deeplabv3plus_20260308_193910\logs\per_class_metrics.csv')  # 类别级指标数据

# ---------------------- 2. 关键数据预处理 ----------------------
# 2.1 找出Val_Miou最高的epoch（最佳模型）
max_miou_idx = metrics_df['val_miou'].idxmax()
best_epoch = metrics_df.loc[max_miou_idx, 'epoch']
best_val_miou = metrics_df.loc[max_miou_idx, 'val_miou']

# 2.2 准备各子图数据
# 左上：Total Loss数据（区分训练集/验证集）
train_total_loss = loss_df[['epoch', 'train_loss']]
val_total_loss = loss_df[['epoch', 'val_loss']]

# 右上：Val_Miou数据
val_miou_data = metrics_df[['epoch', 'val_miou']]

# 左下：BF1_Miou数据（对应val_bf1列）
bf1_miou_data = metrics_df[['epoch', 'val_bf1']]

# 右下：最佳模型的19类IoU数据（按IoU降序排序）
best_model_per_class = per_class_df[per_class_df['epoch'] == best_epoch].copy()
best_class_iou = best_model_per_class[['class_name', 'iou']].sort_values('iou', ascending=False)

# ---------------------- 3. 创建四宫格可视化 ----------------------
# 设置画布大小与布局
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

# 定义颜色方案（统一视觉风格）
colors = {
    'train': '#2E86AB',    # 蓝色（训练集）
    'val': '#A23B72',      # 紫红色（验证集）
    'miou': '#F18F01',     # 橙色（Miou）
    'bf1': '#C73E1D'       # 红色（BF1-Miou）
}

# 3.1 左上子图：Total Loss折线图
ax1.plot(train_total_loss['epoch'], train_total_loss['train_loss'],
         color=colors['train'], linewidth=3, marker='o', markersize=4, label='训练集')
ax1.plot(val_total_loss['epoch'], val_total_loss['val_loss'],
         color=colors['val'], linewidth=3, marker='s', markersize=4, label='验证集')
ax1.set_title('Total Loss 变化曲线', fontsize=18, fontweight='bold', pad=20)
ax1.set_xlabel('Epoch', fontsize=14)
ax1.set_ylabel('Loss 值', fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(0, max(metrics_df['epoch']) + 2)
ax1.tick_params(axis='both', labelsize=12)

# 3.2 右上子图：Val_Miou折线图（标记最高点）
ax2.plot(val_miou_data['epoch'], val_miou_data['val_miou'],
         color=colors['miou'], linewidth=3, marker='o', markersize=4)
# 标记最高Miou点
ax2.scatter(best_epoch, best_val_miou, color='red', s=100, zorder=5,
            label=f'最高: {best_val_miou:.4f} (Epoch {best_epoch})')
ax2.set_title('验证集 Miou 变化曲线', fontsize=18, fontweight='bold', pad=20)
ax2.set_xlabel('Epoch', fontsize=14)
ax2.set_ylabel('Val Miou', fontsize=14)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(0, max(metrics_df['epoch']) + 2)
ax2.set_ylim(0, 1)
ax2.tick_params(axis='both', labelsize=12)

# 3.3 左下子图：BF1_Miou折线图
ax3.plot(bf1_miou_data['epoch'], bf1_miou_data['val_bf1'],
         color=colors['bf1'], linewidth=3, marker='s', markersize=4)
ax3.set_title('验证集 BF1-Miou 变化曲线', fontsize=18, fontweight='bold', pad=20)
ax3.set_xlabel('Epoch', fontsize=14)
ax3.set_ylabel('Val BF1-Miou', fontsize=14)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_xlim(0, max(metrics_df['epoch']) + 2)
ax3.set_ylim(0, max(bf1_miou_data['val_bf1']) * 1.1)
ax3.tick_params(axis='both', labelsize=12)

# 3.4 右下子图：最佳模型19类IoU柱状图
classes = best_class_iou['class_name'].values
ious = best_class_iou['iou'].values
# 创建水平柱状图（便于显示类别名称）
bars = ax4.barh(range(len(classes)), ious, color=plt.cm.viridis(np.linspace(0, 1, len(classes))))
# 添加数值标签
for i, (bar, iou_val) in enumerate(zip(bars, ious)):
    ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f'{iou_val:.3f}', va='center', fontsize=10, fontweight='bold')
ax4.set_yticks(range(len(classes)))
ax4.set_yticklabels(classes, fontsize=12)
ax4.set_title(f'各类别IoU排序',
              fontsize=18, fontweight='bold', pad=20)
ax4.set_xlabel('IoU 值', fontsize=14)
ax4.set_xlim(0, 1.1)
ax4.grid(True, alpha=0.3, linestyle='--', axis='x')
ax4.tick_params(axis='both', labelsize=12)
for ax in [ax1, ax2, ax3, ax4]:
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color("black")

# ---------------------- 4. 图表优化与保存 ----------------------
# 调整子图间距（避免标签重叠）
plt.tight_layout()


# 保存图表（高分辨率300dpi，适合汇报使用）
plt.savefig('D:\MachineLearning\GraduateDesign\outputs\cityscapes_deeplabv3plus_20260308_193910\logsmodel_performance_analysis.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# ---------------------- 5. 输出关键信息 ----------------------
print("四宫格图表已生成并保存为: model_performance_analysis.png")
print(f"\n关键性能指标：")
print(f"1. 最佳训练轮次: Epoch {best_epoch}")
print(f"2. 最高Val_Miou: {best_val_miou:.4f}")
print(f"3. 最佳类别表现: {classes[0]} (IoU: {ious[0]:.4f})")
print(f"4. 最差类别表现: {classes[-1]} (IoU: {ious[-1]:.4f})")
print(f"5. 平均类别IoU: {np.mean(ious):.4f}")
