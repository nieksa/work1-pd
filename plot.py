import os
import glob
import torch
import hdf5storage
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from models import create_model
from data import MRIDataset


# 混淆矩阵绘制函数
def plot_confusion_matrix(targets, preds, num_classes=2):
    cm = confusion_matrix(targets, preds, labels=range(num_classes))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=range(num_classes), yticklabels=range(num_classes))

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    plt.show()


# ROC 曲线绘制函数
def plot_roc_curve(targets, preds, num_classes=2):
    # 将标签进行二值化处理
    targets = label_binarize(targets, classes=[0, 1])  # 适用于二分类任务
    fpr, tpr, thresholds = roc_curve(targets, preds)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()


# 设置参数
task = 'PDvsNC'
model_name = 'ResNet18'
fold = 1

# 创建模型
model = create_model(model_name)

# 查找匹配的模型权重文件
model_weights_pattern = f'./saved_models/{task}/{model_name}_*_fold_{fold}_*.pth'
model_weights_files = glob.glob(model_weights_pattern)

# 检查是否找到匹配的模型权重文件
if model_weights_files:
    model_weights_path = sorted(model_weights_files)[-1]  # 假设最新的文件名包含最高的版本号
    print(f"Loading model weights from {model_weights_path}")
    checkpoint = torch.load(model_weights_path, map_location=torch.device('cpu'))
    print(checkpoint.keys())
    model.load_state_dict(checkpoint['model_state_dict'])  # 使用模型的state_dict加载权重
else:
    print(f"Error: No model weights found matching the pattern {model_weights_pattern}")

# 将模型移到GPU或CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 加载数据
test_x = torch.from_numpy(hdf5storage.loadmat(f'./data/{task}/datas/test_{task}_x_{str(fold)}.mat')['x'])
test_y = torch.from_numpy(hdf5storage.loadmat(f'./data/{task}/datas/test_{task}_y_{str(fold)}.mat')['y'])
val_dataset = MRIDataset(test_x, test_y, transform=None)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

# 模型评估
model.eval()
all_targets = []
all_preds = []

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # 前向传播
        outputs = model(inputs)

        # 获取预测结果
        _, predicted = torch.max(outputs, 1)

        # 存储所有目标和预测值
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# 绘制混淆矩阵
plot_confusion_matrix(all_targets, all_preds, num_classes=2)

# 获取预测概率并绘制ROC曲线
outputs_prob = torch.softmax(outputs, dim=1)[:, 1]  # 假设二分类，取第二类的概率
all_preds_prob = outputs_prob.cpu().numpy()

plot_roc_curve(all_targets, all_preds_prob, num_classes=2)
