import torch
import torch.nn as nn
from collections import OrderedDict

from .Model import Model
from .MyBatchNorm2d import MyBatchNorm2d, ModifiedBatchNorm2d


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet20(Model):
    """
    ResNet20
    """

    def __init__(self, shape_out, in_channels=3):
        super(ResNet20, self).__init__()

        self.in_channels = in_channels
        self.shape_out = shape_out

        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Residual layers
        self.layer1 = self._make_layer(16, 16, 3, stride=1)
        self.layer2 = self._make_layer(16, 32, 3, stride=2)
        self.layer3 = self._make_layer(32, 64, 3, stride=2)

        # Average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, shape_out)

        self.drop_last = True

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_planes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(BasicBlock(in_planes, planes, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.shape[1] == 1 and self.in_channels == 3:  # convert 1-channel image to 3 channel
            x = x.repeat(1, 3, 1, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def change_bn(self, mode='grad', prior=0):
        """
        按照ResNet18样式替换BatchNorm层
        """
        if mode == 'grad':
            # 替换初始BN层
            self.bn1 = MyBatchNorm2d(self.bn1)

            # 替换layer1中的BN层 (3个block)
            self.layer1[0].bn1 = MyBatchNorm2d(self.layer1[0].bn1)
            self.layer1[0].bn2 = MyBatchNorm2d(self.layer1[0].bn2)
            if self.layer1[0].downsample is not None:
                self.layer1[0].downsample[1] = MyBatchNorm2d(self.layer1[0].downsample[1])

            self.layer1[1].bn1 = MyBatchNorm2d(self.layer1[1].bn1)
            self.layer1[1].bn2 = MyBatchNorm2d(self.layer1[1].bn2)

            self.layer1[2].bn1 = MyBatchNorm2d(self.layer1[2].bn1)
            self.layer1[2].bn2 = MyBatchNorm2d(self.layer1[2].bn2)

            # 替换layer2中的BN层 (3个block)
            self.layer2[0].bn1 = MyBatchNorm2d(self.layer2[0].bn1)
            self.layer2[0].bn2 = MyBatchNorm2d(self.layer2[0].bn2)
            if self.layer2[0].downsample is not None:
                self.layer2[0].downsample[1] = MyBatchNorm2d(self.layer2[0].downsample[1])

            self.layer2[1].bn1 = MyBatchNorm2d(self.layer2[1].bn1)
            self.layer2[1].bn2 = MyBatchNorm2d(self.layer2[1].bn2)

            self.layer2[2].bn1 = MyBatchNorm2d(self.layer2[2].bn1)
            self.layer2[2].bn2 = MyBatchNorm2d(self.layer2[2].bn2)

            # 替换layer3中的BN层 (3个block)
            self.layer3[0].bn1 = MyBatchNorm2d(self.layer3[0].bn1)
            self.layer3[0].bn2 = MyBatchNorm2d(self.layer3[0].bn2)
            if self.layer3[0].downsample is not None:
                self.layer3[0].downsample[1] = MyBatchNorm2d(self.layer3[0].downsample[1])

            self.layer3[1].bn1 = MyBatchNorm2d(self.layer3[1].bn1)
            self.layer3[1].bn2 = MyBatchNorm2d(self.layer3[1].bn2)

            self.layer3[2].bn1 = MyBatchNorm2d(self.layer3[2].bn1)
            self.layer3[2].bn2 = MyBatchNorm2d(self.layer3[2].bn2)

        elif mode == 'prior':
            # 替换初始BN层
            self.bn1 = ModifiedBatchNorm2d(self.bn1, prior=prior)

            # 替换layer1中的BN层 (3个block)
            self.layer1[0].bn1 = ModifiedBatchNorm2d(self.layer1[0].bn1, prior=prior)
            self.layer1[0].bn2 = ModifiedBatchNorm2d(self.layer1[0].bn2, prior=prior)
            if self.layer1[0].downsample is not None:
                self.layer1[0].downsample[1] = ModifiedBatchNorm2d(self.layer1[0].downsample[1], prior=prior)

            self.layer1[1].bn1 = ModifiedBatchNorm2d(self.layer1[1].bn1, prior=prior)
            self.layer1[1].bn2 = ModifiedBatchNorm2d(self.layer1[1].bn2, prior=prior)

            self.layer1[2].bn1 = ModifiedBatchNorm2d(self.layer1[2].bn1, prior=prior)
            self.layer1[2].bn2 = ModifiedBatchNorm2d(self.layer1[2].bn2, prior=prior)

            # 替换layer2中的BN层 (3个block)
            self.layer2[0].bn1 = ModifiedBatchNorm2d(self.layer2[0].bn1, prior=prior)
            self.layer2[0].bn2 = ModifiedBatchNorm2d(self.layer2[0].bn2, prior=prior)
            if self.layer2[0].downsample is not None:
                self.layer2[0].downsample[1] = ModifiedBatchNorm2d(self.layer2[0].downsample[1], prior=prior)

            self.layer2[1].bn1 = ModifiedBatchNorm2d(self.layer2[1].bn1, prior=prior)
            self.layer2[1].bn2 = ModifiedBatchNorm2d(self.layer2[1].bn2, prior=prior)

            self.layer2[2].bn1 = ModifiedBatchNorm2d(self.layer2[2].bn1, prior=prior)
            self.layer2[2].bn2 = ModifiedBatchNorm2d(self.layer2[2].bn2, prior=prior)

            # 替换layer3中的BN层 (3个block)
            self.layer3[0].bn1 = ModifiedBatchNorm2d(self.layer3[0].bn1, prior=prior)
            self.layer3[0].bn2 = ModifiedBatchNorm2d(self.layer3[0].bn2, prior=prior)
            if self.layer3[0].downsample is not None:
                self.layer3[0].downsample[1] = ModifiedBatchNorm2d(self.layer3[0].downsample[1], prior=prior)

            self.layer3[1].bn1 = ModifiedBatchNorm2d(self.layer3[1].bn1, prior=prior)
            self.layer3[1].bn2 = ModifiedBatchNorm2d(self.layer3[1].bn2, prior=prior)

            self.layer3[2].bn1 = ModifiedBatchNorm2d(self.layer3[2].bn1, prior=prior)
            self.layer3[2].bn2 = ModifiedBatchNorm2d(self.layer3[2].bn2, prior=prior)

    def set_running_stat_grads(self):
        for m in self.modules():
            if isinstance(m, MyBatchNorm2d):
                m.set_running_stat_grads()

    def clip_bn_running_vars(self):
        for m in self.modules():
            if isinstance(m, MyBatchNorm2d):
                m.clip_running_var()

    def freeze_bn_stats(self):
        """
        Do not update running stats of batch norm layers.
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                m.eval()

    def set_layers_to_adapt(self, mode='all'):
        print(mode)
        self.train()
        if mode in ['bn_all', 'bn_stat', 'bn_params']:
            # disable grad, to (re-)enable later
            self.requires_grad_(False)
            # configure norm for tent updates: enable grad + force batch statisics
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    if mode in ['bn_all', 'bn_params']:
                        m.requires_grad_(True)
                    if mode in ['bn_all', 'bn_stat']:
                        m.track_running_stats = True
                        m.momentum = 1.0
                    else:
                        m.track_running_stats = False

        elif mode == 'tent':
            self.requires_grad_(False)
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None

        elif mode == 'last_layer':
            # disable grad, to (re-)enable later
            self.requires_grad_(False)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.requires_grad_(True)
                elif isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False

        elif mode == 'last_bias':
            # disable grad, to (re-)enable later
            self.requires_grad_(False)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.bias.requires_grad_(True)
                elif isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False

        elif mode == 'first_conv':
            # disable grad, to (re-)enable later
            self.requires_grad_(False)
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False

            self.conv1.requires_grad_(True)

        elif mode == 'block1':
            self.requires_grad_(False)
            self.layer1.requires_grad_(True)
            for m in self.layer1.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'block2':
            self.requires_grad_(False)
            self.layer2.requires_grad_(True)
            for m in self.layer2.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'block3':
            self.requires_grad_(False)
            self.layer3.requires_grad_(True)
            for m in self.layer3.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'all':
            pass

        else:
            raise NotImplementedError

    def surgical(self, mode='all'):
        self.requires_grad_(False)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False

        if mode == 'block1':
            self.layer1.requires_grad_(True)
            for m in self.layer1.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'block2':
            self.layer2.requires_grad_(True)
            for m in self.layer2.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'block3':
            self.layer3.requires_grad_(True)
            for m in self.layer3.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True

        elif mode == 'last_layer':
            self.fc.requires_grad_(True)

        else:
            raise NotImplementedError

    @property
    def featurizer(self):
        """动态生成特征提取器"""
        return nn.Sequential(OrderedDict([
            ('conv1', self.conv1),
            ('bn1', self.bn1),
            ('relu', self.relu),
            ('layer1', self.layer1),
            ('layer2', self.layer2),
            ('layer3', self.layer3),
            ('avgpool', self.avgpool),
            ('flatten', nn.Flatten()),
        ]))

    @property
    def classifier(self):
        """直接返回分类器层"""
        return self.fc


def test():
    model = ResNet20(shape_out=10)
    model.change_bn()
    total_num = sum(p.numel() for name, p in model.state_dict().items())
    print(f"Total parameters: {total_num}")
    print("Parameter names:", dict(model.named_parameters()).keys())
    print("Number of parameter groups:", len(dict(model.named_parameters())))

    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    test()