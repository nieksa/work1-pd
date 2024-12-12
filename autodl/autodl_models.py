from models.model_design import Design1

def create_model(model_name):
    if model_name == 'Design1':
        model = Design1(in_channels=1, out_channel=128, class_num=2, num_blocks=[1,1,1])
    else:
        raise ValueError(f'Unsupported model: {model_name}')
    return model