from utils import pickle_load

from .CNN import ShallowCNN
from .ResNet18 import ResNet18
from .ResNet20 import ResNet20
from .ResNet50 import ResNet50
from .ViT import ViT  # 新增 ViT 导入
from .CCT import CCT4  # 新增CCT4导入

def create_model(args):
    """
    Create model
    :param args:
    :return:
    """
    shape_in = args.shape_in
    shape_out = args.shape_out

    if args.model == 'resnet18':
        model = ResNet18(shape_out=shape_out)
    elif args.model == 'resnet20':
        model = ResNet20(shape_out=shape_out)
    elif args.model == 'resnet50':
        model = ResNet50(shape_out=shape_out)
    elif args.model == 'cnn':
        model = ShallowCNN(shape_in=shape_in, shape_out=shape_out)
    elif args.model == 'vit':  # 新增 ViT 选项
        model = ViT(shape_out=shape_out)
    elif args.model == 'cct4':  # 新增CCT4选项
        model = CCT4(shape_out=shape_out)
    else:
        raise NotImplementedError('Unknown model. ')

    model.to(args.device)

    if args.load_model_path != 'none':
        state = pickle_load(args.load_model_path)
        model.load_state_dict(state)
        print('Loaded model from {}'.format(args.load_model_path))

    return model