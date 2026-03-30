import torch
import torch.nn as nn
import copy
from .cct_utils import Tokenizer, TransformerClassifier
from .MyLayerNorm import MyLayerNorm, ModifiedLayerNorm


# ATP框架基础Model类
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop_last = True

    def change_bn(self, mode='grad', prior=0):
        pass

    def set_running_stat_grads(self):
        pass

    def clip_bn_running_vars(self):
        pass

    def surgical(self, mode='all'):
        pass

    @property
    def featurizer(self):
        return nn.Identity()

    def updated_state_dict(self):
        return copy.deepcopy(self.state_dict())

    def load_updated_state_dict(self, state_dict):
        self.load_state_dict(state_dict, strict=True)


# 适配ATP框架的CCT类
class CCT(Model):
    def __init__(self, shape_out, img_size=32, positional_embedding='learnable', num_classes=10):
        super(CCT, self).__init__()
        # CCT-4 3x2 32配置（贴合你的官方版）
        self.embedding_dim = 128
        self.num_layers = 4
        self.num_heads = 2
        self.mlp_ratio = 1.0
        self.kernel_size = 3
        self.n_conv_layers = 2
        self.img_size = img_size
        self._classifier = None  # 避免递归

        # 初始化Tokenizer和TransformerClassifier
        self.tokenizer = Tokenizer(
            n_input_channels=3,
            n_output_channels=self.embedding_dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=1,
            pooling_kernel_size=3,
            pooling_stride=2,
            pooling_padding=1,
            n_conv_layers=self.n_conv_layers
        )

        self.classifier_module = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(height=img_size, width=img_size),
            embedding_dim=self.embedding_dim,
            seq_pool=True,
            dropout=0.,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            num_classes=shape_out,  # 直接指定最终分类数
            positional_embedding=positional_embedding
        )

        # 分类头（适配框架）
        self._classifier = self.classifier_module.fc
        self.drop_last = True

    def forward(self, x):
        # 处理单通道图像
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # 前向传播（核心简化：直接返回classifier_module结果）
        x = self.tokenizer(x)
        x = self.classifier_module(x)  # 内部已包含fc分类，无需额外处理
        return x

    def change_bn(self, mode='grad', prior=0):
        """替换LayerNorm/BN为自定义版本"""
        for name, module in self.named_modules():
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                child_name = name.rsplit('.', 1)[-1]
                parent = self
                if parent_name:
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                if mode == 'grad':
                    setattr(parent, child_name,
                            MyLayerNorm(module) if isinstance(module, nn.LayerNorm) else MyLayerNorm(module))
                elif mode == 'prior':
                    setattr(parent, child_name, ModifiedLayerNorm(module, prior=prior) if isinstance(module,
                                                                                                     nn.LayerNorm) else ModifiedLayerNorm(
                        module, prior=prior))

    def set_running_stat_grads(self):
        for m in self.modules():
            if isinstance(m, MyLayerNorm):
                m.set_running_stat_grads()

    def clip_bn_running_vars(self):
        for m in self.modules():
            if isinstance(m, MyLayerNorm):
                m.clip_running_var()

    def surgical(self, mode='all'):
        """选择性冻结参数"""
        self.requires_grad_(False)
        if mode == 'encoder':
            self.classifier_module.blocks.requires_grad_(True)
        elif mode == 'last_layer':
            self._classifier.requires_grad_(True)
        elif mode == 'patch_embed':
            self.tokenizer.requires_grad_(True)
        elif mode == 'attention':
            for block in self.classifier_module.blocks:
                block.attn.requires_grad_(True)
        elif mode == 'mlp':
            for block in self.classifier_module.blocks:
                block.mlp.requires_grad_(True)
        elif mode == 'all':
            self.requires_grad_(True)
        else:
            raise NotImplementedError(f"Unsupported mode: {mode}")

    @property
    def featurizer(self):
        """特征提取器（适配框架）"""
        return nn.Sequential(
            self.tokenizer,
            self.classifier_module.pos_emb,
            self.classifier_module.dropout,
            self.classifier_module.blocks,
            self.classifier_module.norm
        )

    @property
    def classifier(self):
        """分类器属性（避免递归）"""
        return self._classifier

    @classifier.setter
    def classifier(self, new_classifier):
        self._classifier = new_classifier
        self.classifier_module.fc = new_classifier


# 兼容原有create_model的接口
def CCT4(shape_out):
    return CCT(shape_out=shape_out, img_size=32, num_classes=shape_out)


# 测试函数
def test():
    model = CCT4(shape_out=100)  # CIFAR-100
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    print(f"Output shape: {output.shape}")  # 应输出 torch.Size([2, 100])
    print(f"Updated state dict keys: {list(model.updated_state_dict().keys())[:5]}")
    print("Test passed!")


if __name__ == "__main__":
    test()