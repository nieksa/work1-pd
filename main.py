import numpy as np
import torch
from data import MRIDataset, DataLoader
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingLR,LambdaLR
import torch.backends.cudnn as cudnn
import random
from models import create_model
from tensorboardX import SummaryWriter
import argparse
import hdf5storage
import logging
import os
import time

parser = argparse.ArgumentParser(description='Training script for models.')

# 添加参数和默认值
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--seed', type=int, default=1337, help='Random seed for reproducibility.')
parser.add_argument('--device_ids', nargs='+', type=int, default=[0], help='List of device IDs to use.')
parser.add_argument('--model_name', type=str, default='C3D', help='Name of the model to use.')
parser.add_argument('--task', type=str, default='PDvsNC', choice = ['PDvsNC', 'PDvsSWEDD', 'NCvsSWEDD'])

args = parser.parse_args()

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device is not 'cpu':
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.95, 0)
    torch.backends.cuda.max_split_size_mb = 4096

log_dir = './logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
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

logging.info(f"Training with {args.device}")
logging.info(f"Model: {args.model_name}")
all_metrics = {metric: [] for metric in ['acc', 'ba', 'kappa', 'auc', 'f1', 'precision', 'recall', 'specificity']}
for n in range(5):
    logging.info(f'Fold {n+1} Starting')
    train_x = torch.from_numpy(hdf5storage.loadmat(f'.data/{args.task}/datas/train_{args.task}_x_{str(n + 1)}.mat')['x'])
    train_y = torch.from_numpy(hdf5storage.loadmat(f'.data/{args.task}/datas/train_{args.task}_y_{str(n + 1)}.mat')['y'])

    test_x = torch.from_numpy(hdf5storage.loadmat(f'.data/{args.task}/datas/test_{args.task}_x_{str(n + 1)}.mat')['x'])
    test_y = torch.from_numpy(hdf5storage.loadmat(f'.data/{args.task}/datas/test_{args.task}_y_{str(n + 1)}.mat')['y'])

    train_labels_unique, train_labels_count = torch.unique(train_y, return_counts=True)
    test_labels_unique, test_labels_count = torch.unique(test_y, return_counts=True)

    logging.info(f'Training labels distribution: {dict(zip(train_labels_unique.tolist(), train_labels_count.tolist()))}')
    logging.info(f'Testing labels distribution: {dict(zip(test_labels_unique.tolist(), test_labels_count.tolist()))}')

    train_dataset = MRIDataset(train_x, train_y, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=16
                                                   , shuffle=True,drop_last=True)

    val_dataset = MRIDataset(test_x, test_y, transform=None)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.8)
    loss_function = torch.nn.CrossEntropyLoss()
    
    model = create_model(args.model_name).to(device)

    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    writer = SummaryWriter()
    max_epochs = 5

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_dataset) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()

            num_correct = 0.0
            metric_count = 0
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                with torch.no_grad():
                    val_outputs = model(val_images)
                    value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
                    metric_count += len(value)
                    num_correct += value.sum().item()

            metric = num_correct / metric_count
            metric_values.append(metric)

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
                print("saved new best metric model")

            print(f"Current epoch: {epoch+1} current accuracy: {metric:.4f} ")
            print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
            writer.add_scalar("val_accuracy", metric, epoch + 1)

    print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()

result_message = ''
for metric, values in all_metrics.items():
    avg = mean(values)
    std = stdev(values)
    result_message += f"{avg * 100:.2f}±{std * 100:.2f}\t"

avg_acc = mean(all_metrics['acc']) * 100
logging.info(f"\n{result_message}")
logging.shutdown()

new_logfilename = os.path.join(log_dir, f'{args.model_name}_{timestamp}_{avg_acc:.2f}.log')
os.rename(log_file,new_logfilename)