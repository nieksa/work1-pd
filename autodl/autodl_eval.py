import numpy as np
import torch
import logging
import os

def confusion_matrix(y_true, y_pred):
    # 转换为 numpy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = np.zeros((2, 2), dtype=int)

    # 计算混淆矩阵
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def accuracy_score(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    if total == 0:
        return 0.0  # 防止除以零
    return correct / total

def cohen_kappa_score(y_true, y_pred):
    confusion = confusion_matrix(y_true, y_pred)
    total = np.sum(confusion)

    if total == 0:
        return 0.0  # 防止除以零

    observed_agreement = np.trace(confusion) / total
    expected_agreement = (np.sum(confusion, axis=0) * np.sum(confusion, axis=1)) / (total * total)
    expected_agreement = np.sum(expected_agreement)

    if expected_agreement == 1:
        return 0.0

    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    return kappa

def roc_auc_score(y_true, y_probs):
    auc = 0.0
    return auc


def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    if precision + recall == 0:
        return 0.0  # 防止除以零
    return 2 * (precision * recall) / (precision + recall)


def precision_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    if tp + fp == 0:
        return 0.0  # 防止除以零
    return tp / (tp + fp)


def recall_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp + fn == 0:
        return 0.0  # 防止除以零
    return tp / (tp + fn)


def specificity_score(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    if tn + fp == 0:
        return 0.0  # 防止除以零
    return tn / (tn + fp)


def balanced_accuracy_score(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    specificity = specificity_score(y_true, y_pred)
    if recall + specificity == 0:
        return 0.0  # 防止除以零
    return (recall + specificity) / 2


def eval_model(model, dataloader, device, epoch):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 模型推理
            outputs = model(inputs)
            probs = outputs.softmax(dim=1)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    cm = confusion_matrix(all_labels, all_preds)

    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    except ValueError:
        auc = 0.0

    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    specificity = specificity_score(all_labels, all_preds)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)

    avg_metrics = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'kappa': kappa,
        'auc': auc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'specificity': specificity
    }

    logging.info(
        f"Epoch:{epoch} | "
        f"Accuracy: {avg_metrics['accuracy']:.4f} | "
        f"BA: {avg_metrics['balanced_accuracy']:.4f} | "
        f"Kappa: {avg_metrics['kappa']:.4f} | "
        f"AUC: {avg_metrics['auc']:.4f} | "
        f"F1: {avg_metrics['f1']:.4f} | "
        f"Pre: {avg_metrics['precision']:.4f} | "
        f"Recall: {avg_metrics['recall']:.4f} | "
        f"Spec: {avg_metrics['specificity']:.4f}"
    )
    return avg_metrics

def save_best_model(model, eval_metric, best_metric, best_metric_model, args, timestamp, fold, epoch, metric_name):
    model_name = args.model_name
    task = args.task
    if eval_metric[metric_name] >= best_metric[metric_name]:
        best_metric[metric_name] = eval_metric[metric_name]
        model_path = f'./saved_models/{task}/{model_name}_{timestamp}_fold_{fold}_epoch_{epoch}_{metric_name}_{best_metric[metric_name]:.2f}.pth'
        if metric_name in best_metric_model and best_metric_model[metric_name]:
            old_model_path = best_metric_model[metric_name]
            if os.path.exists(old_model_path):
                os.remove(old_model_path)
        best_model = model
        best_metric_model[metric_name] = model_path
        torch.save(best_model.state_dict(),best_metric_model[metric_name])

if __name__ == "__main__":
    outputs = torch.tensor([[2.0, 1.0],  # 第一个样本的预测结果
                            [0.5, 2.5],  # 第二个样本的预测结果
                            [1.5, 0.5],  # 第三个样本的预测结果
                            [1.0, 1.0]])  # 第四个样本的预测结果

    y_true = torch.tensor([0, 1, 1, 0])
    _, y_pred = torch.max(outputs, 1)

    y_probs = torch.softmax(outputs, dim=1)

    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    y_probs_np = y_probs.cpu().numpy()

    cm = confusion_matrix(y_true_np, y_pred_np)
    print("Confusion Matrix:")
    print(cm)

    accuracy = accuracy_score(y_true_np, y_pred_np)
    print("Accuracy:", accuracy)

    kappa = cohen_kappa_score(y_true_np, y_pred_np)
    print("Kappa:", kappa)

    auc = roc_auc_score(y_true_np, y_probs_np)
    print("AUC:", auc)

    f1 = f1_score(y_true_np, y_pred_np)
    print("F1 Score:", f1)

    precision = precision_score(y_true_np, y_pred_np)
    print("Precision:", precision)

    recall = recall_score(y_true_np, y_pred_np)
    print("Recall:", recall)

    specificity = specificity_score(y_true_np, y_pred_np)
    print("Specificity:", specificity)

    balanced_accuracy = balanced_accuracy_score(y_true_np, y_pred_np)
    print("Balanced Accuracy:", balanced_accuracy)
