from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义一个3x3x3卷积层
def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     dilation=1,
                     bias=False)
def conv7x7x7(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=5,
                     dilation=1,
                     bias=False)
class BasicBlock_SmallVoxel(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock_SmallVoxel, self).__init__()
        self.conv1 = conv3x3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm3d(out_planes)

        # 残差连接：如果输入和输出的通道数不一致，或者步幅不为1，使用1x1x1卷积调整维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_planes)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))  # 第一层卷积+激活
        out = self.bn2(self.conv2(out))  # 第二层卷积
        out += self.shortcut(x)  # 残差连接
        out = self.relu(out)  # 激活
        return out

class BasicBlock_LargeVoxel(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock_LargeVoxel, self).__init__()
        self.conv1 = conv7x7x7(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm3d(out_planes)

        # 残差连接：如果输入和输出的通道数不一致，或者步幅不为1，使用1x1x1卷积调整维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_planes)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))  # 第一层卷积+激活
        out = self.bn2(self.conv2(out))  # 第二层卷积
        out += self.shortcut(x)  # 残差连接
        out = self.relu(out)  # 激活
        return out

class ResNet3D(nn.Module):
    def __init__(self, block, in_channels, out_channels, num_blocks):
        super(ResNet3D, self).__init__()
        self.block = block
        self.conv1 = conv3x3x3(in_channels, 64)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks[3], stride=2)

        self.fc = nn.Linear(512, out_channels)

    def _make_layer(self,in_planes, out_planes, num_blocks, stride):
        layers = []
        layers.append(self.block(in_planes, out_planes, stride))
        for _ in range(1, num_blocks):
            layers.append(self.block(out_planes, out_planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        print(x.shape)
        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)
        x = self.layer4(x)
        print(x.shape)
        return x

small_voxel_block = BasicBlock_SmallVoxel
large_voxel_block = BasicBlock_LargeVoxel
# 测试模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'training with {device}')
model = ResNet3D(small_voxel_block, in_channels=1, out_channels=16, num_blocks=[2, 2, 2, 2]).to(device)
input_tensor = torch.randn(2, 1, 128, 128, 128).to(device)
output_tensor = model(input_tensor)
print(output_tensor.shape)
