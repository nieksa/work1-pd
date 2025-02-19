import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
qwen_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)  # 将qwen_model移到GPU

class Image2Token(nn.Module):
    """
    将3D医学影像数据转换为Qwen2可以接受的输入
    batch, seq_len, 896
    """

    def __init__(self):
        super(Image2Token, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d((64, 64, 64))
        self.linear = nn.Linear(512, 896)

    def forward(self, x):
        x = self.pool(x)
        batch_size, channels, depth, height, width = x.size()
        x = x.view(batch_size, channels, depth // 8, 8, height // 8, 8, width // 8, 8)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.view(batch_size, -1, 512)
        x = self.linear(x)
        return x


class QwenForBinaryClassification(nn.Module):
    """
    下一步是考虑改冻结哪些层
    """
    def __init__(self, qwen_model):
        super(QwenForBinaryClassification, self).__init__()
        self.qwen_model = qwen_model
        self.classifier = nn.Linear(896, 2)  # 假设Qwen的隐藏层大小为896

    def forward(self, features):
        outputs = self.qwen_model(inputs_embeds=features, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        logits = self.classifier(last_hidden_state[:, 0, :])
        return logits


class MedicalImageClassifier(nn.Module):
    def __init__(self, qwen_model, freeze_qwen_layers=6):
        super(MedicalImageClassifier, self).__init__()

        # 初始化CNN提取器
        self.cnn_extractor = Image2Token()
        # 初始化Qwen分类器
        self.qwen_classifier = QwenForBinaryClassification(qwen_model)

        # 冻结Qwen模型的前`freeze_qwen_layers`层
        for i in range(freeze_qwen_layers):
            for param in self.qwen_classifier.qwen_model.model.layers[i].parameters():
                param.requires_grad = False

        # 解冻Qwen的后几层
        for i in range(freeze_qwen_layers, len(self.qwen_classifier.qwen_model.model.layers)):
            for param in self.qwen_classifier.qwen_model.model.layers[i].parameters():
                param.requires_grad = True

        # 其他Qwen模型部分的冻结策略
        for param in self.qwen_classifier.qwen_model.model.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.qwen_classifier.qwen_model.model.norm.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 通过CNN处理图像，得到特征
        features = self.cnn_extractor(x)
        logits = self.qwen_classifier(features)

        return logits

def create_qwen2():
    model = MedicalImageClassifier(qwen_model)
    return model

if __name__ == '__main__':
    model = MedicalImageClassifier(qwen_model).to(device)
    # model = Image2Token()
    # print(model)
    x = torch.randn(2, 1, 128, 128, 128).to(device)  # 示例医学影像数据并将其移到GPU
    logits = model(x)

    print(logits)  # 输出二分类结果
