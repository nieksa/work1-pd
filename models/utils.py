# 上面导入各式各样的model，同时在这里就写死参数
import torch
from torch.onnx.symbolic_opset9 import numel

from models.C3D import C3D
from monai.networks.nets.classifier import Classifier, Discriminator, Critic
# from mamba_ssm import Mamba
from design1 import ViT
from resnet import ResNet, BasicBlock, get_inplanes, Bottleneck
from C3D import C3D
from I3D import InceptionI3d
from densnet import DenseNet
from slowfast import SlowFast
from vgg import VGG

def create_model(model_name):
    if model_name == 'Classifier':
        model = Classifier(
            in_shape = (128, 128, 128),
            classes = 2,
            channels = (32, 64, 128),
            strides = (1, 2, 2),
            kernel_size = 3,
            num_res_units = 2,
            # act=Act.PRELU,
            # norm=Norm.INSTANCE,
            dropout = 0.1,
            bias = True,
            last_act = 'softmax',
        )
    elif model_name == 'ViT':
        model = ViT(image_size=128, image_patch_size=16, frames=128, frame_patch_size=16,
                    num_classes=2, dim=1024, depth=2, heads=4, mlp_dim=64, pool='cls',
                    channels=1, dim_head=32, dropout=0.2, emb_dropout=0.1)
    elif model_name == 'ResNet18':
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), n_input_channels=1, n_classes=2)
    elif model_name == 'ResNet50':
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), n_input_channels=1, n_classes=2)
    elif model_name == 'C3D':
        model = C3D(num_classes=2)
    elif model_name == 'I3D':
        model = InceptionI3d(num_classes=2, spatial_squeeze=True,
                     final_endpoint='Logits', name='inception_i3d', in_channels=1, dropout_keep_prob=0.5)
    elif model_name == 'DenseNet264':#一共有四种[121, 169, 201, 264]
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 64, 48),
                         n_input_channels=1, num_classes=2)
    elif model_name == 'DenseNet121':#一共有四种[121, 169, 201, 264]
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 24, 16),
                         n_input_channels=1, num_classes=2)
    elif model_name == 'SlowFast':
        model = SlowFast(layers=[3, 4, 6, 3], class_num=2, dropout=0.5)
    elif model_name == 'VGG':
        model = VGG(dropout=0.5, n_classes=2)

    else:
        raise ValueError(f'Unsupported model: {model_name}')
    return model


if __name__ == '__main__':
    model = create_model('VGG')
    x = torch.rand(4, 1, 128, 128, 128)
    label = torch.randint(0, 2, (4,))
    out = model(x)
    print(label)
    print(out)