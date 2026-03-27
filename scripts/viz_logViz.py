import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

                                                            
                
plt.rcParams["font.sans-serif"] = ["SimHei"]
                                             

plt.rcParams['axes.unicode_minus'] = False

          
loss_df = pd.read_csv('/outputs/mid-distilledMobile\logs\metrics.csv')
metrics_df = pd.read_csv('/outputs/mid-distilledMobile\logs\metrics.csv')
per_class_df = pd.read_csv('/outputs/mid-distilledMobile\logs\per_class_metrics.csv')

                                                          
                              
max_miou_idx = metrics_df['val_miou'].idxmax()
best_epoch = metrics_df.loc[max_miou_idx, 'epoch']
best_val_miou = metrics_df.loc[max_miou_idx, 'val_miou']

             
                            
train_total_loss = loss_df[['epoch', 'train_loss']]
val_total_loss = loss_df[['epoch', 'val_loss']]

               
val_miou_data = metrics_df[['epoch', 'val_miou']]

                           
bf1_miou_data = metrics_df[['epoch', 'val_bf1']]

                            
best_model_per_class = per_class_df[per_class_df['epoch'] == best_epoch].copy()
best_class_iou = best_model_per_class[['class_name', 'iou']].sort_values('iou', ascending=False)

                                                           
           
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

                
colors = {
    'train': '#2E86AB',             
    'val': '#A23B72',                
    'miou': '#F18F01',               
    'bf1': '#C73E1D'                     
}

                        
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

                             
ax2.plot(val_miou_data['epoch'], val_miou_data['val_miou'],
         color=colors['miou'], linewidth=3, marker='o', markersize=4)
           
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

                      
ax3.plot(bf1_miou_data['epoch'], bf1_miou_data['val_bf1'],
         color=colors['bf1'], linewidth=3, marker='s', markersize=4)
ax3.set_title('验证集 BF1-Miou 变化曲线', fontsize=18, fontweight='bold', pad=20)
ax3.set_xlabel('Epoch', fontsize=14)
ax3.set_ylabel('Val BF1-Miou', fontsize=14)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_xlim(0, max(metrics_df['epoch']) + 2)
ax3.set_ylim(0, max(bf1_miou_data['val_bf1']) * 1.1)
ax3.tick_params(axis='both', labelsize=12)

                        
classes = best_class_iou['class_name'].values
ious = best_class_iou['iou'].values
                   
bars = ax4.barh(range(len(classes)), ious, color=plt.cm.viridis(np.linspace(0, 1, len(classes))))
        
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

                                                          
                
plt.tight_layout()


                         
plt.savefig('D:\MachineLearning\GraduateDesign\outputs\cityscapes_deeplabv3plus_mobile_distill_20260311_233247\logs\logsmodel_performance_analysis.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

                                                         
print("四宫格图表已生成并保存为: model_performance_analysis.png")
print(f"\n关键性能指标：")
print(f"1. 最佳训练轮次: Epoch {best_epoch}")
print(f"2. 最高Val_Miou: {best_val_miou:.4f}")
print(f"3. 最佳类别表现: {classes[0]} (IoU: {ious[0]:.4f})")
print(f"4. 最差类别表现: {classes[-1]} (IoU: {ious[-1]:.4f})")
print(f"5. 平均类别IoU: {np.mean(ious):.4f}")
