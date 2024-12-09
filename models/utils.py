# 上面导入各式各样的model，同时在这里就写死参数
from monai.networks.nets.classifier import Classifier, Discriminator, Critic
# from mamba_ssm import Mamba
from .design1 import ViT
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
    if model_name == 'ViT':
        model = ViT(image_size=128, image_patch_size=16, frames=128, frame_patch_size=16,
                    num_classes=2, dim=128, depth=4, heads=4, mlp_dim=128, pool='cls',
                    channels=1, dim_head=32, dropout=0.2, emb_dropout=0.1)
    else:
        raise ValueError(f'Unsupported model: {model_name}')
    return model