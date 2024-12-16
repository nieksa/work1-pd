import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# positional

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')

# modules

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.heads = num_heads
        head_dim = dim // self.heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        q = q * self.scale

        attn = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')

        return self.proj_drop(self.proj(x))


class TransformerEncoderLayer(nn.Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and
    rwightman's timm package.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super().__init__()

        self.pre_norm = nn.LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1  = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1    = nn.LayerNorm(d_model)
        self.linear2  = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate)

        self.activation = F.gelu

    def forward(self, src, *args, **kwargs):
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        batch, drop_prob, device, dtype = x.shape[0], self.drop_prob, x.device, x.dtype

        if drop_prob <= 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (batch, *((1,) * (x.ndim - 1)))

        keep_mask = torch.zeros(shape, device = device).float().uniform_(0, 1) < keep_prob
        output = x.div(keep_prob) * keep_mask.float()
        return output


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        seq_pool=True,
        embedding_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=1000,
        dropout_rate=0.1,
        attention_dropout=0.1,
        stochastic_depth_rate=0.1,
        positional_embedding='sine',
        sequence_length=None,
        *args, **kwargs
    ):
        super().__init__()
        assert positional_embedding in {'sine', 'learnable', 'none'}

        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool

        assert exists(sequence_length) or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        else:
            self.attention_pool = nn.Linear(self.embedding_dim, 1)

        if positional_embedding == 'none':
            self.positional_emb = None
        elif positional_embedding == 'learnable':
            self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length, embedding_dim))
            nn.init.trunc_normal_(self.positional_emb, std = 0.2)
        else:
            self.register_buffer('positional_emb', sinusoidal_embedding(sequence_length, embedding_dim))

        self.dropout = nn.Dropout(p=dropout_rate)

        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout_rate,
                                    attention_dropout=attention_dropout, drop_path_rate=layer_dpr)
            for layer_dpr in dpr])

        self.norm = nn.LayerNorm(embedding_dim)

        self.fc = nn.Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and exists(m.bias):
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        b = x.shape[0]
        if not exists(self.positional_emb) and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)
        if not self.seq_pool:
            cls_token = repeat(self.class_emb, '1 1 d -> b 1 d', b = b)
            x = torch.cat((cls_token, x), dim=1)
        if exists(self.positional_emb):
            x += self.positional_emb
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.seq_pool:
            attn_weights = rearrange(self.attention_pool(x), 'b n 1 -> b n')
            x = einsum('b n, b n d -> b d', attn_weights.softmax(dim = 1), x)
        else:
            x = x[:, 0]
        return self.fc(x)


class Block1(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1):
        super(Block1, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu2 = nn.ReLU()
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        shortcut = self.shortcut(x)
        out += shortcut
        out = torch.relu(out)
        out = self.pool(out)
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

class Design2(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, image_size=128, patch_size=4, frames=128, frame_patch_size=4,
                 embedding_dim=256, num_heads=8, positional_embedding='sine',sequence_length=None):
        super(Design2, self).__init__()
        self.block1 = Block1(in_channels=in_channels, out_channels=64)
        self.block2 = Block2(image_size=image_size, image_patch_size=patch_size, frames=frames,
                             frame_patch_size=frame_patch_size, dim=embedding_dim, channels=64)
        self.transformer = TransformerClassifier(embedding_dim=embedding_dim,
                                                 num_heads=num_heads,num_classes=num_classes,
                                                 positional_embedding=positional_embedding,
                                                 sequence_length=sequence_length)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.transformer(x)
        return x


if __name__ == '__main__':
    torch.manual_seed(42)
    x = torch.randn(4, 1, 128, 128, 128)
    model = Design2(positional_embedding='sine',sequence_length=4096)
    out = model(x)
    print(out.shape)