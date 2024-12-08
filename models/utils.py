# 上面导入各式各样的model，同时在这里就写死参数
from monai.networks.nets.classifier import Classifier, Discriminator, Critic
def create_model(model_name):
    if model_name == 'SwinTransformer':
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
    else:
        raise ValueError(f'Unsupported model: {model_name}')
    return model