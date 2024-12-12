from classifier import LogisticRegressionClassifier
from dual_resnet import DualBranchResNet
from vit import ViT
import torch
import torch.nn as nn

class Design1 (nn.Module):
    def __init__ (self, in_channels=1, out_channel=128, class_num=2, num_blocks=[1,1,1]):
        super().__init__()
        self.dual_resnet = DualBranchResNet(in_channels=in_channels, out_channels=out_channel, num_blocks=num_blocks)
        self.vit = ViT(channels=out_channel*2)
        self.classifier = LogisticRegressionClassifier(input_dim=64, output_dim=class_num)
    def forward(self, x):
        x = self.dual_resnet(x)
        x = self.vit(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Design1(in_channels=1, class_num=2, num_blocks=[1,1,1]).to(device)
    x = torch.randn(8, 1, 128, 128, 128).to(device)
    out = model(x)
    print(out.shape)