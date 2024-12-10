from monai.data.dataset import Dataset
from torch.utils.data import DataLoader
import hdf5storage
import torch
import logging


class MRIDataset(Dataset):
    def __init__(self, x, y, transform = None):
        self.x = x
        self.y = y
        self.transform = transform
        if transform is not None:
            self.transform = transform

    def __getitem__(self, index):
        xi = self.x[index]#.permute(0,1,3,2)
        #print(xi.shape)
        if self.transform is not None:
            xi = self.transform(xi)
        yi = self.y[index]/1.0
        return xi, yi

    def __len__(self):
        return len(self.x)


def load_data(args, n=0, batch_size_train=16, batch_size_val=16, test = False):
    """
    加载数据并返回 DataLoader
    :param args: 任务参数
    :param n: 数据集索引
    :param batch_size_train: 训练数据的批大小
    :param batch_size_val: 验证数据的批大小
    :return: train_loader, val_loader
    """
    # 加载训练数据和标签
    train_x = torch.from_numpy(hdf5storage.loadmat(f'./data/{args.task}/datas/train_{args.task}_x_{str(n + 1)}.mat')['x'])
    train_y = torch.from_numpy(hdf5storage.loadmat(f'./data/{args.task}/datas/train_{args.task}_y_{str(n + 1)}.mat')['y'])

    # 加载测试数据和标签
    test_x = torch.from_numpy(hdf5storage.loadmat(f'./data/{args.task}/datas/test_{args.task}_x_{str(n + 1)}.mat')['x'])
    test_y = torch.from_numpy(hdf5storage.loadmat(f'./data/{args.task}/datas/test_{args.task}_y_{str(n + 1)}.mat')['y'])

    if test:
        train_x = train_x[:10]
        train_y = train_y[:10]
        test_x = test_x[:10]
        test_y = test_y[:10]

    # 打印训练集和测试集的标签分布
    train_labels_unique, train_labels_count = torch.unique(train_y, return_counts=True)
    test_labels_unique, test_labels_count = torch.unique(test_y, return_counts=True)

    logging.info(f'Training labels distribution: {dict(zip(train_labels_unique.tolist(), train_labels_count.tolist()))}')
    logging.info(f'Testing labels distribution: {dict(zip(test_labels_unique.tolist(), test_labels_count.tolist()))}')

    # 创建训练数据集和验证数据集
    train_dataset = MRIDataset(train_x, train_y, transform=None)
    val_dataset = MRIDataset(test_x, test_y, transform=None)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)

    return train_dataset, val_dataset, train_loader, val_loader