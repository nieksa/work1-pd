import torch
from torch.onnx.symbolic_opset9 import numel
from monai.networks.nets.classifier import Classifier, Discriminator, Critic
from vit_pytorch.vit_3d import ViT
from vit_pytorch.cct_3d import CCT,cct_4
from vit_pytorch.vivit import ViT as ViViT
from vit_pytorch.simple_vit_3d import SimpleViT

from .resnet import ResNet, BasicBlock, get_inplanes, Bottleneck
from .C3D import C3D
from .I3D import InceptionI3d
from .densnet import DenseNet
from .slowfast import SlowFast
from .vgg import VGG
from model_design import Design1


def create_model(model_name):
    if model_name == 'ViT':
        model = ViT(image_size=128, image_patch_size=16, frames=128, frame_patch_size=16,
                    num_classes=2, dim=1024, depth=2, heads=4, mlp_dim=64, pool='cls',
                    channels=1, dim_head=32, dropout=0.2, emb_dropout=0.1)
    # elif model_name == 'ResNet18':
    #     model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), n_input_channels=1, n_classes=2)
    # elif model_name == 'ResNet50':
    #     model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), n_input_channels=1, n_classes=2)
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
    # elif model_name == 'SlowFast':
    #     model = SlowFast(layers=[3, 4, 6, 3], class_num=2, dropout=0.5)
    # elif model_name == 'VGG':
    #     model = VGG(dropout=0.5, n_classes=2)
    elif model_name == 'cct4': #复杂度高，4090 24G 跑不了
        model = cct_4(img_size=128, num_frames=128, num_classes=2, n_input_channels= 1)
    # elif model_name == 'ViViT': # 效果不是很好，不知道是不是参数的问题
    #     model = ViViT(
    #         image_size=128,  # image size
    #         frames=128,  # number of frames
    #         image_patch_size=16,  # image patch size
    #         frame_patch_size=2,  # frame patch size
    #         num_classes=2,
    #         dim=512,
    #         spatial_depth=5,  # depth of the spatial transformer
    #         temporal_depth=5,  # depth of the temporal transformer
    #         heads=5,
    #         mlp_dim=1024,
    #         channels = 1,
    #         variant='factorized_encoder',  # or 'factorized_self_attention'
    #     )
    elif model_name == 'SimpleViT': #复杂度高，4090 24G 跑不了
        model = SimpleViT(
            image_size = 128,          # image size
            frames = 128,               # number of frames
            image_patch_size = 16,     # image patch size
            frame_patch_size = 2,      # frame patch size
            num_classes = 2,
            dim = 512,
            depth = 5,
            heads = 8,
            mlp_dim = 1024,
            channels = 1,
            dim_head = 64
        )
    elif model_name == 'Design1':
        model = Design1(in_channels=1, out_channel=128, class_num=2, num_blocks=[1,1,1])
    else:
        raise ValueError(f'Unsupported model: {model_name}')
    return model


# model_name = ['Classifier','ResNet18','ResNet50','C3D','I3D','DenseNet264','DenseNet121','SlowFast','VGG',
#               'ViT','cct4','ViViT','SimpleViT']

if __name__ == '__main__':
    model = create_model('SimpleViT')
    x = torch.rand(4, 1, 128, 128, 128)
    label = torch.randint(0, 2, (4,))
    out = model(x)
    print(label)
    print(out)