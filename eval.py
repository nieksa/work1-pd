import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score, precision_score, \
    recall_score, confusion_matrix, roc_auc_score
import torch
import logging
import os

def eval_model(model, dataloader, device, epoch):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 模型推理
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = torch.tensor(all_labels)
    all_preds = torch.tensor(all_preds)

    cm = confusion_matrix(all_labels, all_preds)

    accuracy = accuracy_score(all_labels, all_preds)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_preds, average='macro', multi_class='ovr')
    except ValueError:
        auc = 0.0

    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    specificity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if cm.shape == (2, 2) else 0.0  # 计算特异性

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



def save_best_model(model, eval_metric, best_metric, best_metric_model, model_name, timestamp, fold, epoch, metric_name):
    if eval_metric[metric_name] > best_metric[metric_name]:
        best_metric[metric_name] = eval_metric[metric_name]
        model_path = f'./saved_models/{model_name}_{timestamp}_fold_{fold}_epoch_{epoch}_{metric_name}_{best_metric[metric_name]:.2f}.pth'
        if metric_name in best_metric_model and best_metric_model[metric_name]:
            old_model_path = best_metric_model[metric_name]
            if os.path.exists(old_model_path):
                print(f'Deleting old model: {old_model_path}')
                os.remove(old_model_path)
        best_model = model
        best_metric_model[metric_name] = model_path
        torch.save(best_model.state_dict(),best_metric_model[metric_name])
        print(f"Saved new best model for {metric_name}: {model_path}")
