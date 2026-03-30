import torch
import torch.nn as nn
import math
import numpy as np
from collections import OrderedDict

from .Model import Model
from .MyBatchNorm2d import MyBatchNorm2d, ModifiedBatchNorm2d

# 保留原始DeepLab的常量和基础函数
affine_par = True


def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return i


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# 保留原始的基础模块（修正Classifier_Module的forward缩进错误）
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Classifier_Module(nn.Module):
    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, num_classes, kernel_size=3, stride=1,
                                              padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out  # 修正缩进错误


# 核心：改写成和ResNet18一致的类格式
class ResDeeplab(Model):
    """
    DeepLabv2 with ResNet101 backbone (Res_Deeplab)
    适配ResNet18示例的代码格式，支持BN层替换、梯度配置等功能
    """

    def __init__(self, shape_out):
        super(ResDeeplab, self).__init__()

        # 构建原始ResNet+DeepLabv2 backbone（对应ResNet18的self.backbone）
        self.backbone = self._build_resdeeplab_backbone(num_classes=shape_out)

        # 分割任务默认不丢弃最后一个batch（和分类任务区分）
        self.drop_last = False

    def _build_resdeeplab_backbone(self, num_classes):
        """构建原始的ResNet+DeepLabv2网络作为backbone"""

        class _ResNet(nn.Module):
            def __init__(self, block, layers, num_classes):
                self.inplanes = 64
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
                for i in self.bn1.parameters():
                    i.requires_grad = False
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
                self.layer1 = self._make_layer(block, 64, layers[0])
                self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
                self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
                self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

                # 初始化权重
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(0, 0.01)
                    elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()

            def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
                downsample = None
                if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
                for i in downsample._modules['1'].parameters():
                    i.requires_grad = False
                layers = []
                layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
                self.inplanes = planes * block.expansion
                for i in range(1, blocks):
                    layers.append(block(self.inplanes, planes, dilation=dilation))
                return nn.Sequential(*layers)

            def _make_pred_layer(self, block, dilation_series, padding_series, num_classes):
                return block(dilation_series, padding_series, num_classes)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.layer5(x)
                return x

        # 构建ResNet101+DeepLabv2 backbone（对应原始Res_Deeplab）
        return _ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

    def forward(self, x):
        # 兼容单通道输入（和ResNet18示例一致）
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # 调用backbone的forward（和ResNet18示例逻辑一致）
        return self.backbone(x)

    def change_bn(self, mode='grad', prior=0):
        """替换所有BN层（适配ResNet101的Bottleneck结构）"""
        model = self.backbone

        if mode == 'grad':
            # 替换根节点BN
            model.bn1 = MyBatchNorm2d(model.bn1)

            # 替换Layer1（Bottleneck结构：bn1/bn2/bn3 + downsample的bn）
            for idx, bottleneck in enumerate(model.layer1):
                bottleneck.bn1 = MyBatchNorm2d(bottleneck.bn1)
                bottleneck.bn2 = MyBatchNorm2d(bottleneck.bn2)
                bottleneck.bn3 = MyBatchNorm2d(bottleneck.bn3)
                if bottleneck.downsample is not None:
                    bottleneck.downsample[1] = MyBatchNorm2d(bottleneck.downsample[1])

            # 替换Layer2
            for idx, bottleneck in enumerate(model.layer2):
                bottleneck.bn1 = MyBatchNorm2d(bottleneck.bn1)
                bottleneck.bn2 = MyBatchNorm2d(bottleneck.bn2)
                bottleneck.bn3 = MyBatchNorm2d(bottleneck.bn3)
                if bottleneck.downsample is not None:
                    bottleneck.downsample[1] = MyBatchNorm2d(bottleneck.downsample[1])

            # 替换Layer3
            for idx, bottleneck in enumerate(model.layer3):
                bottleneck.bn1 = MyBatchNorm2d(bottleneck.bn1)
                bottleneck.bn2 = MyBatchNorm2d(bottleneck.bn2)
                bottleneck.bn3 = MyBatchNorm2d(bottleneck.bn3)
                if bottleneck.downsample is not None:
                    bottleneck.downsample[1] = MyBatchNorm2d(bottleneck.downsample[1])

            # 替换Layer4
            for idx, bottleneck in enumerate(model.layer4):
                bottleneck.bn1 = MyBatchNorm2d(bottleneck.bn1)
                bottleneck.bn2 = MyBatchNorm2d(bottleneck.bn2)
                bottleneck.bn3 = MyBatchNorm2d(bottleneck.bn3)
                if bottleneck.downsample is not None:
                    bottleneck.downsample[1] = MyBatchNorm2d(bottleneck.downsample[1])

        elif mode == 'prior':
            # 替换为带prior的ModifiedBatchNorm2d
            model.bn1 = ModifiedBatchNorm2d(model.bn1, prior=prior)

            # 替换Layer1
            for idx, bottleneck in enumerate(model.layer1):
                bottleneck.bn1 = ModifiedBatchNorm2d(bottleneck.bn1, prior=prior)
                bottleneck.bn2 = ModifiedBatchNorm2d(bottleneck.bn2, prior=prior)
                bottleneck.bn3 = ModifiedBatchNorm2d(bottleneck.bn3, prior=prior)
                if bottleneck.downsample is not None:
                    bottleneck.downsample[1] = ModifiedBatchNorm2d(bottleneck.downsample[1], prior=prior)

            # 替换Layer2
            for idx, bottleneck in enumerate(model.layer2):
                bottleneck.bn1 = ModifiedBatchNorm2d(bottleneck.bn1, prior=prior)
                bottleneck.bn2 = ModifiedBatchNorm2d(bottleneck.bn2, prior=prior)
                bottleneck.bn3 = ModifiedBatchNorm2d(bottleneck.bn3, prior=prior)
                if bottleneck.downsample is not None:
                    bottleneck.downsample[1] = ModifiedBatchNorm2d(bottleneck.downsample[1], prior=prior)

            # 替换Layer3
            for idx, bottleneck in enumerate(model.layer3):
                bottleneck.bn1 = ModifiedBatchNorm2d(bottleneck.bn1, prior=prior)
                bottleneck.bn2 = ModifiedBatchNorm2d(bottleneck.bn2, prior=prior)
                bottleneck.bn3 = ModifiedBatchNorm2d(bottleneck.bn3, prior=prior)
                if bottleneck.downsample is not None:
                    bottleneck.downsample[1] = ModifiedBatchNorm2d(bottleneck.downsample[1], prior=prior)

            # 替换Layer4
            for idx, bottleneck in enumerate(model.layer4):
                bottleneck.bn1 = ModifiedBatchNorm2d(bottleneck.bn1, prior=prior)
                bottleneck.bn2 = ModifiedBatchNorm2d(bottleneck.bn2, prior=prior)
                bottleneck.bn3 = ModifiedBatchNorm2d(bottleneck.bn3, prior=prior)
                if bottleneck.downsample is not None:
                    bottleneck.downsample[1] = ModifiedBatchNorm2d(bottleneck.downsample[1], prior=prior)

    def set_running_stat_grads(self):
        """和ResNet18示例逻辑一致"""
        for m in self.backbone.modules():
            if isinstance(m, MyBatchNorm2d):
                m.set_running_stat_grads()

    def clip_bn_running_vars(self):
        """和ResNet18示例逻辑一致"""
        for m in self.backbone.modules():
            if isinstance(m, MyBatchNorm2d):
                m.clip_running_var()

    def freeze_bn_stats(self):
        """冻结BN层的running stats（和ResNet18示例一致）"""
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                m.eval()

    def set_layers_to_adapt(self, mode='all'):
        """配置不同适配模式的梯度和BN行为（完全对齐ResNet18示例）"""
        print(mode)
        self.backbone.train()

        if mode in ['bn_all', 'bn_stat', 'bn_params']:
            # 全局关闭梯度，按需开启BN层梯度
            self.backbone.requires_grad_(False)
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    if mode in ['bn_all', 'bn_params']:
                        m.requires_grad_(True)
                    if mode in ['bn_all', 'bn_stat']:
                        m.track_running_stats = True
                        m.momentum = 1.0
                    else:
                        m.track_running_stats = False

        elif mode == 'tent':
            self.backbone.requires_grad_(False)
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None

        elif mode == 'last_layer':
            # 仅开启分类层（layer5）梯度
            self.backbone.requires_grad_(False)
            self.backbone.layer5.requires_grad_(True)
            # 冻结所有BN的running stats
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False

        elif mode == 'last_bias':
            # 仅开启分类层偏置梯度
            self.backbone.requires_grad_(False)
            for m in self.backbone.layer5.modules():
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.requires_grad_(True)
            # 冻结BN stats
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False

        elif mode == 'first_conv':
            # 仅开启第一层卷积梯度
            self.backbone.requires_grad_(False)
            self.backbone.conv1.requires_grad_(True)
            # 冻结BN stats
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False

        elif mode == 'block1':
            self.backbone.requires_grad_(False)
            self.backbone.layer1.requires_grad_(True)
            for m in self.backbone.layer1.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'block2':
            self.backbone.requires_grad_(False)
            self.backbone.layer2.requires_grad_(True)
            for m in self.backbone.layer2.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'block3':
            self.backbone.requires_grad_(False)
            self.backbone.layer3.requires_grad_(True)
            for m in self.backbone.layer3.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'block4':
            self.backbone.requires_grad_(False)
            self.backbone.layer4.requires_grad_(True)
            for m in self.backbone.layer4.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'all':
            pass

        else:
            raise NotImplementedError

    def surgical(self, mode='all'):
        """精细化梯度配置（完全对齐ResNet18示例）"""
        self.backbone.requires_grad_(False)
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False

        if mode == 'block1':
            self.backbone.layer1.requires_grad_(True)
            for m in self.backbone.layer1.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'block2':
            self.backbone.layer2.requires_grad_(True)
            for m in self.backbone.layer2.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'block3':
            self.backbone.layer3.requires_grad_(True)
            for m in self.backbone.layer3.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'block4':
            self.backbone.layer4.requires_grad_(True)
            for m in self.backbone.layer4.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'last_layer':
            self.backbone.layer5.requires_grad_(True)

        else:
            raise NotImplementedError

    @property
    def featurizer(self):
        """动态生成特征提取器（对应ResNet101 backbone，不含分类层）"""
        resnet = self.backbone
        return nn.Sequential(OrderedDict([
            ('conv1', resnet.conv1),
            ('bn1', resnet.bn1),
            ('relu', resnet.relu),
            ('maxpool', resnet.maxpool),
            ('layer1', resnet.layer1),
            ('layer2', resnet.layer2),
            ('layer3', resnet.layer3),
            ('layer4', resnet.layer4),
        ]))

    @property
    def classifier(self):
        """返回分类层（DeepLabv2的ASPP模块）"""
        return self.backbone.layer5


def test():
    """测试函数（和ResNet18示例格式一致）"""
    model = ResDeeplab(shape_out=21)  # VOC数据集21类
    model.change_bn()

    # 打印参数总数
    total_num = sum(p.numel() for name, p in model.state_dict().items())
    print(f"Total parameters: {total_num}")

    # 打印参数名和数量
    param_keys = list(dict(model.named_parameters()).keys())
    print(f"First 5 parameter keys: {param_keys[:5]}...")
    print(f"Total parameter count: {len(param_keys)}")

    # 测试前向传播
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    print(f"Output shape: {out.shape}")  # 输出: (2, 21, 64, 64)


if __name__ == "__main__":
    test()