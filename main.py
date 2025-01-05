import torch
from data import load_data
from torch.nn import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingLR,LambdaLR
import torch.backends.cudnn as cudnn
import random
from models import create_model
from tensorboardX import SummaryWriter
from utils import setup_training_environment, rename_log_file
from tqdm import tqdm
import logging
import os
from statistics import mean, stdev
from eval import eval_model, save_best_model

test = False
val_start = 10
val_interval = 1
patience = 5

args ,device, log_file, timestamp = setup_training_environment()
all_metrics = {metric: [] for metric in ['accuracy', 'balanced_accuracy', 'kappa', 'auc', 'f1',
                                         'precision', 'recall', 'specificity']}
for n in range(5):
    logging.info(f'Fold {n+1} Starting')

    train_dataset, val_dataset, train_loader, val_loader = load_data(args, n=n,
                                                                     batch_size_train=args.train_bs,
                                                                     batch_size_val=args.val_bs,
                                                                     test = test, num_workers=args.num_workers)
    loss_function = torch.nn.CrossEntropyLoss()
    
    model = create_model(args.model_name).to(device)
    if torch.cuda.device_count() > 1:
        device_ids = list(range(torch.cuda.device_count()))
        model = DataParallel(model, device_ids=device_ids).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.8)

    epoch_loss_values = []
    metric_values = []
    # writer = SummaryWriter()

    best_metric = {
        'accuracy':0,
        'f1':0,
    }
    best_metric_model = {
        'accuracy':None,
        'f1':None,
    }
    max_epochs = args.epochs
    min_delta = 0.001
    best_val_metric = 0
    epochs_without_improvement = 0
    # best_model_weights = model.state_dict().copy()
    result_metric = None
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        step = 0
        epoch_preds = []
        epoch_labels = []

        for batch_idx, (data, labels) in tqdm(enumerate(train_loader)):
            step += 1
            data = data.to(device)
            labels = labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_dataset) // train_loader.batch_size
            # writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

            _, preds = torch.max(outputs, 1)
            epoch_preds.extend(preds.cpu().numpy())
            epoch_labels.extend(labels.cpu().numpy())


        if (epoch + 1) % val_interval == 0 and (epoch + 1) >= val_start:
            eval_metrics = eval_model(model=model, dataloader=val_loader, device=device, epoch=epoch + 1)
            current_val_metric = eval_metrics['accuracy']
            if current_val_metric > best_val_metric + min_delta:
                best_val_metric = current_val_metric
                epochs_without_improvement = 0
                # best_model_weights = model.state_dict().copy()
                # save_best_model(model, eval_metrics, best_metric, best_metric_model, args, timestamp,
                #                 fold=fold, epoch=epoch+1, metric_name='accuracy')
                result_metric = eval_metrics
            else:
                epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logging.info(f"Early Stopping at Epoch {epoch + 1}. Val Metric did not improve for {patience} epochs.")
            break

    # model.load_state_dict(best_model_weights)
    # avg_metrics = eval_model(model=model, dataloader=val_loader, device=device, epoch='FINAL')
    avg_metrics = result_metric
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
    # writer.close()

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

rename_log_file(log_file, avg_acc, args.task, args.model_name, timestamp)