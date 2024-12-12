import torch
import torch.nn as nn
import torch.nn.functional as F


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
                     kernel_size=7,
                     stride=stride,
                     padding=3,
                     dilation=1,
                     bias=False)


class SmallVoxelResNet(nn.Module):
    def __init__(self, in_planes, out_planes, num_blocks):
        super(SmallVoxelResNet, self).__init__()
        self.conv1 = conv3x3x3(in_planes, 64)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 128, num_blocks[0], stride=2)  # 64 -> 128, stride 2
        self.layer2 = self._make_layer(128, 128, num_blocks[1], stride=2)  # 128 -> 128, stride 2
        self.layer3 = self._make_layer(128, 128, num_blocks[2], stride=1)  # 128 -> 128, stride 1 (保持大小不变)

    def _make_layer(self, in_planes, out_planes, num_blocks, stride):
        layers = []
        layers.append(self._make_block(in_planes, out_planes, stride))
        for _ in range(1, num_blocks):
            layers.append(self._make_block(out_planes, out_planes))
        return nn.Sequential(*layers)

    def _make_block(self, in_planes, out_planes, stride=1):
        return nn.Sequential(
            conv3x3x3(in_planes, out_planes, stride),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True),
            conv3x3x3(out_planes, out_planes),
            nn.BatchNorm3d(out_planes)
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class LargeVoxelResNet(nn.Module):
    def __init__(self, in_planes, out_planes, num_blocks):
        super(LargeVoxelResNet, self).__init__()
        self.conv1 = conv7x7x7(in_planes, 64)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 128, num_blocks[0], stride=2)  # 64 -> 128, stride 2
        self.layer2 = self._make_layer(128, 128, num_blocks[1], stride=2)  # 128 -> 128, stride 2
        self.layer3 = self._make_layer(128, 128, num_blocks[2], stride=1)  # 128 -> 128, stride 1 (保持大小不变)

    def _make_layer(self, in_planes, out_planes, num_blocks, stride):
        layers = []
        layers.append(self._make_block(in_planes, out_planes, stride))
        for _ in range(1, num_blocks):
            layers.append(self._make_block(out_planes, out_planes))
        return nn.Sequential(*layers)

    def _make_block(self, in_planes, out_planes, stride=1):
        return nn.Sequential(
            conv7x7x7(in_planes, out_planes, stride),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True),
            conv7x7x7(out_planes, out_planes),
            nn.BatchNorm3d(out_planes)
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class DualBranchResNet(nn.Module):
    # (batch, channel = 1, 128, 128, 128) -> (batch, 256, 32, 32, 32)
    # 一半通道是大体素特征， 一半是小体素特征
    def __init__(self, in_channels, out_channels, num_blocks):
        super(DualBranchResNet, self).__init__()
        # 小体素分支
        self.small_voxel_branch = SmallVoxelResNet(in_channels, out_channels, num_blocks)
        # 大体素分支
        self.large_voxel_branch = LargeVoxelResNet(in_channels, out_channels, num_blocks)
    def forward(self, x):
        small_out = self.small_voxel_branch(x)
        large_out = self.large_voxel_branch(x)
        out = torch.cat((small_out, large_out), dim=1)  # 拼接通道维度
        return out

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualBranchResNet(in_channels=1, out_channels=128, num_blocks=[1, 1, 1]).to(device)
    input_tensor = torch.randn(4, 1, 128, 128, 128).to(device)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)
