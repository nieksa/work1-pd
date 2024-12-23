# 这里应该才是写好训练过程的函数
# only for autodl in schoole
import torch
import logging
from data import load_data
import argparse
import os
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingLR,LambdaLR
from statistics import mean, stdev

from autodl_models import create_model
from autodl_eval import save_best_model, eval_model
# 这里创建models

parser = argparse.ArgumentParser(description='Training script for models.')

parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--seed', type=int, default=1337, help='Random seed for reproducibility.')
parser.add_argument('--device_ids', nargs='+', type=int, default=[0], help='List of device IDs to use.')
parser.add_argument('--model_name', type=str, default='Design1', help='Name of the model to use.')
parser.add_argument('--task', type=str, default='PDvsNC', choices = ['PDvsNC', 'PDvsSWEDD', 'NCvsSWEDD'])
parser.add_argument('--train_bs', type=int, default=16, help='I3D C3D cuda out of memory.')
parser.add_argument('--val_bs', type=int, default=16, help='densenet cuda out of memory.')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_dir = f'./logs/{args.task}'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(f'./saved_models/{args.task}', exist_ok=True)

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log_file = os.path.join(log_dir, f'{args.model_name}_{timestamp}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),  # 将日志输出到文件
        logging.StreamHandler()          # 同时输出到控制台
    ]
)

logging.info(f"Training with {device}")
logging.info(f"Model: {args.model_name}")
logging.info(f"Task: {args.task}")
all_metrics = {metric: [] for metric in ['accuracy', 'balanced_accuracy', 'kappa', 'auc', 'f1',
                                         'precision', 'recall', 'specificity']}
for n in range(5):
    logging.info(f'Fold {n + 1} Starting')

    train_dataset, val_dataset, train_loader, val_loader = load_data(args, n=0,
                                                                     batch_size_train=args.train_bs,
                                                                     batch_size_val=args.val_bs,
                                                                     test=False)
    loss_function = torch.nn.CrossEntropyLoss()

    model = create_model(args.model_name).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.8)

    epoch_loss_values = []
    metric_values = []

    best_metric = {
        'accuracy': 0,
        'f1': 0,
    }  # 用于记录当前最佳指标
    best_metric_model = {
        'accuracy': None,
        'f1': None,
    }
    # 可以通过添加指标项来保存最好的模型

    max_epochs = args.epochs
    val_interval = 1  # 每5个epochs做一次验证
    for epoch in range(max_epochs):
        # print("-" * 10)
        # print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            labels = labels.long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_dataset) // train_loader.batch_size
            # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        logging.info(f"epoch {epoch + 1} ---------------------------------------------- average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0 and (epoch + 1) >= 0:
            eval_metrics = eval_model(model=model, dataloader=val_loader, device=device, epoch=epoch + 1)

            save_best_model(model, eval_metrics, best_metric, best_metric_model, args, timestamp,
                            fold=n, epoch=epoch, metric_name='accuracy')

    avg_metrics = eval_model(model=model, dataloader=val_loader, device=device, epoch='FINAL')
    logging.info(
        f"Fold {n + 1} : {avg_metrics['accuracy']:.4f} | "
        f"BA: {avg_metrics['balanced_accuracy']:.4f} | "
        f"Kappa: {avg_metrics['kappa']:.4f} | "
        f"AUC: {avg_metrics['auc']:.4f} | "
        f"F1: {avg_metrics['f1']:.4f} | "
        f"Pre: {avg_metrics['precision']:.4f} | "
        f"Recall: {avg_metrics['recall']:.4f} | "
        f"Spec: {avg_metrics['specificity']:.4f}"
    )

    for metric, value in avg_metrics.items():
        all_metrics[metric].append(value)

result_message = ''
for metric, values in all_metrics.items():
    avg = mean(values)
    std = stdev(values)
    result_message += f"{avg * 100:.2f}±{std * 100:.2f}\t"

avg_acc = mean(all_metrics['accuracy']) * 100
logging.info(f"\n{result_message}")
logging.shutdown()

new_logfilename = os.path.join(log_dir, f'{args.model_name}_{timestamp}_{avg_acc:.2f}.log')
os.rename(log_file, new_logfilename)