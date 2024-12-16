import torch
from torch import nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2 if downsample else 1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = nn.Sequential()

        if in_channels != out_channels or downsample:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2 if downsample else 1, padding=0)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + residual
        out = self.relu(out)
        return out


class Block1(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(Block1, self).__init__()
        self.resnet_block1 = ResNetBlock(in_channels, out_channels, downsample=True)
        self.resnet_block2 = ResNetBlock(out_channels, out_channels, downsample=True)
    def forward(self, x):
        out = self.resnet_block1(x)
        out = self.resnet_block2(out)
        return out


class Block2(nn.Module): # out (b, patch的个数, dim)
    def __init__(self, *, image_size=64, image_patch_size=4, frames=64, frame_patch_size=4, dim=256, channels=64):
        super(Block2, self).__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        patch_dim = channels * patch_height * patch_width * frame_patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1=patch_height, p2=patch_width,
                      pf=frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
    def forward(self, x):
        x = self.to_patch_embedding(x)
        return x

class TransformerClassifier(nn.Module):
    def __init__(self,
                 feature_dim=256,  # 每个位置的特征维度
                 num_classes=10,  # 类别数
                 seq_length=128,  # 序列长度
                 num_heads=8,  # 注意力头数
                 num_layers=6,  # Transformer 编码器层数
                 mlp_ratio=4.0,  # MLP 比例
                 dropout_rate=0.1,  # Dropout 比例
                 positional_embedding='learnable'):  # 位置编码类型（sinusoidal / learnable）
        super(TransformerClassifier, self).__init__()

        # 位置编码
        self.positional_embedding_type = positional_embedding
        self.positional_embedding = None

        if positional_embedding == 'learnable':
            self.positional_embedding = nn.Parameter(torch.zeros(1, seq_length, feature_dim))
        elif positional_embedding == 'sinusoidal':
            self.register_buffer('positional_embedding', self.get_sinusoidal_embedding(seq_length, feature_dim))

        # Transformer Encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=int(feature_dim * mlp_ratio),
            dropout=dropout_rate,
            batch_first=True)

        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 分类头
        self.fc = nn.Linear(feature_dim, num_classes)

    def get_sinusoidal_embedding(self, seq_length, dim):
        """
        生成 sinusoidal 位置编码
        """
        position = torch.arange(0, seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / dim))
        pos_embedding = torch.zeros(seq_length, dim)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        return pos_embedding.unsqueeze(0)  # Shape: (1, seq_length, feature_dim)

    def forward(self, x):
        """
        输入形状: (batch_size, seq_length, feature_dim)
        输出: 分类结果 (batch_size, num_classes)
        """
        b, seq_len, _ = x.shape

        # 加入位置编码
        if self.positional_embedding_type == 'learnable':
            x += self.positional_embedding
        else:
            x += self.positional_embedding

        # 使用 Transformer 编码器
        x = x.permute(1, 0, 2)  # Shape: (seq_length, batch_size, feature_dim)
        x = self.transformer_encoder(x)

        # 取序列最后一个位置的特征 (通常用于分类任务)
        x = x[-1, :, :]  # Shape: (batch_size, feature_dim)

        # 分类头
        x = self.fc(x)  # Shape: (batch_size, num_classes)
        return x


class Design2(nn.Module):
    def __init__(self, in_channels=64, image_size=128, patch_size=4, frames=128, frame_patch_size=4,
                 embedding_dim=256):
        super(Design2, self).__init__()
        self.block1 = Block1(in_channels=in_channels, out_channels=64)
        self.block2 = Block2(image_size=image_size, image_patch_size=patch_size, frames=frames,
                             frame_patch_size=frame_patch_size, dim=embedding_dim, channels=64)
        self.transformerclassifier = TransformerClassifier(feature_dim=embedding_dim, num_classes=2,seq_length=512)
    def forward(self, x):
        x = self.block1(x)  # 64, 32, 32, 32
        x = self.block2(x)  # 32/4**3, embedding_dim
        x = self.transformerclassifier(x)
        return x

if __name__ == '__main__':
    torch.manual_seed(42)
    x = torch.randn(4, 1, 128, 128, 128)
    model = Design2(in_channels=1, image_size=128, patch_size=4, frames=128, frame_patch_size=4,embedding_dim=1024)
    out = model(x)
    print(out.shape)