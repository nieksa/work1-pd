import torch
import torch.nn as nn
import torch.nn.functional as F

class LogisticRegressionClassifier(nn.Module):
    def __init__(self, input_dim=64, output_dim=2):
        super(LogisticRegressionClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)

class WeightedAveragePoolingClassifier(nn.Module):
    def __init__(self, input_dim=64, output_dim=2, heads=4, num_layers=2, hidden_dim=128):
        super(WeightedAveragePoolingClassifier, self).__init__()

        # 使用自注意力计算权重
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=heads)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(0)  # 变成 (1, batch_size, dim) 以便做transformer处理
        attn_output, _ = self.attn(x, x, x)  # 自注意力
        x = attn_output.squeeze(0)  # 恢复为 (batch_size, dim)
        x = self.fc(x)
        return F.softmax(x, dim=-1)
class FCClassifier(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=2, dropout=0.5):
        super(FCClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=-1)



if __name__ == '__main__':
    input_tensor = torch.randn(4, 64)  # 假设输入是 (batch_size=4, dim=64)
    model = LogisticRegressionClassifier(input_dim=64, output_dim=2)
    output = model(input_tensor)
    print("输出结果: ", output)
    print("输出尺寸: ", output.shape)  # 应该是 (batch_size=4, 2)
