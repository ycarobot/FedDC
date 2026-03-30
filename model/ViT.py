#
# import torch
# import torch.nn as nn
# from torchvision.transforms import Resize
# from collections import OrderedDict
# import timm
# from .Model import Model
# from .MyLayerNorm import StandardLayerNormAdapter
#
#
# class ViT(Model):
#     def __init__(self, shape_out):
#         super(ViT, self).__init__()
#
#         # 使用 timm 的 vit_small (ViT-S/16)
#         self.backbone = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
#
#         # 添加自定义分类头
#         self.head = nn.Linear(384, shape_out)  # vit_small 的隐藏维度是384
#
#         # 图像尺寸调整
#         self.resize = Resize((224, 224))
#         self.drop_last = True
#
#     def forward(self, x):
#         # 单通道转三通道
#         if x.shape[1] == 1:
#             x = x.repeat(1, 3, 1, 1)
#
#         # 调整尺寸后前向传播
#         x = self.resize(x)
#         features = self.backbone(x)
#         return self.head(features)
#
#     # ========== LayerNorm 适配相关方法 ==========
#     def change_bn(self, mode='grad', prior=0):
#         if mode != 'grad':
#             return
#
#         def _wrap_ln(module):
#             for name, child in module.named_children():
#                 if isinstance(child, nn.LayerNorm):
#                     setattr(module, name, StandardLayerNormAdapter(child))
#                 else:
#                     _wrap_ln(child)
#
#         _wrap_ln(self.backbone)
#
#     def set_running_stat_grads(self):
#         for m in self.backbone.modules():
#             if isinstance(m, StandardLayerNormAdapter):
#                 m.set_running_stat_grads()
#
#     def clip_bn_running_vars(self):
#         for m in self.backbone.modules():
#             if isinstance(m, StandardLayerNormAdapter):
#                 m.clip_running_var()
#
#     def freeze_bn_stats(self):
#         for m in self.backbone.modules():
#             if isinstance(m, StandardLayerNormAdapter):
#                 m.eval()
#
#     # ========== 参数冻结控制 ==========
#     def set_layers_to_adapt(self, mode='all'):
#         self.backbone.train()
#
#         if mode in ['bn_all', 'bn_stat', 'bn_params']:
#             self.backbone.requires_grad_(False)
#             for m in self.backbone.modules():
#                 if isinstance(m, StandardLayerNormAdapter):
#                     if mode in ['bn_all', 'bn_params']:
#                         m.meta_scale.requires_grad_(True)
#                         m.meta_shift.requires_grad_(True)
#
#         elif mode == 'tent':
#             self.backbone.requires_grad_(False)
#             for m in self.backbone.modules():
#                 if isinstance(m, StandardLayerNormAdapter):
#                     m.meta_scale.requires_grad_(True)
#                     m.meta_shift.requires_grad_(True)
#
#         elif mode == 'last_layer':
#             self.backbone.requires_grad_(False)
#             self.head.requires_grad_(True)
#
#         elif mode == 'last_bias':
#             self.backbone.requires_grad_(False)
#             self.head.bias.requires_grad_(True)
#
#         elif mode == 'first_conv':
#             self.backbone.requires_grad_(False)
#             self.backbone.patch_embed.proj.requires_grad_(True)
#
#         elif mode == 'encoder_layers':
#             self.backbone.requires_grad_(False)
#             self.backbone.blocks.requires_grad_(True)
#
#         elif mode == 'attention_layers':
#             self.backbone.requires_grad_(False)
#             for block in self.backbone.blocks:
#                 block.attn.requires_grad_(True)
#
#         elif mode == 'mlp_layers':
#             self.backbone.requires_grad_(False)
#             for block in self.backbone.blocks:
#                 block.mlp.requires_grad_(True)
#
#         elif mode == 'all':
#             pass
#
#         else:
#             raise NotImplementedError
#
#     # ========== 特征提取接口 ==========
#     @property
#     def featurizer(self):
#         return nn.Sequential(OrderedDict([
#             ('resize', self.resize),
#             ('patch_embed', self.backbone.patch_embed),
#             ('blocks', self.backbone.blocks),
#             ('norm', self.backbone.norm),
#             ('flatten', nn.Flatten()),
#         ]))
#
#     @property
#     def classifier(self):
#         return self.head
#
#
# def test():
#     model = ViT(shape_out=10)
#     print("模型结构:")
#     print(model)
#
#     # 测试参数
#     total = sum(p.numel() for p in model.parameters())
#     print(f"\n总参数: {total:,}")
#
#     # 测试前向传播
#     x = torch.randn(2, 3, 32, 32)
#     out = model(x)
#     print(f"\n输入形状: {x.shape} -> 输出形状: {out.shape}")
#
#     # 测试LayerNorm适配
#     model.change_bn(mode='grad')
#     adapted = len([m for m in model.modules() if isinstance(m, StandardLayerNormAdapter)])
#     print(f"\n适配后的LayerNorm数量: {adapted}")
#
#
# # if __name__ == '__main__':
# #     test()
#
#     def forward(self, x):
#         # 保持输入处理逻辑不变：单通道转三通道 + 尺寸调整
#         if x.shape[1] == 1:
#             x = x.repeat(1, 3, 1, 1)
#         x = self.resize(x)
#         return self.backbone(x)
#
#     # 以下接口保持与原代码完全一致，确保与ATP框架兼容
#     def change_bn(self, mode='grad', prior=0):
#         """用StandardLayerNormAdapter替换所有nn.LayerNorm"""
#         if mode != 'grad':
#             return
#
#         def _wrap_ln(module):
#             for name, child in module.named_children():
#                 if isinstance(child, nn.LayerNorm):
#                     setattr(module, name, StandardLayerNormAdapter(child))
#                 else:
#                     _wrap_ln(child)
#
#         _wrap_ln(self.backbone)
#
#     def set_running_stat_grads(self):
#         for m in self.backbone.modules():
#             if isinstance(m, StandardLayerNormAdapter):
#                 m.set_running_stat_grads()
#
#     def clip_bn_running_vars(self):
#         for m in self.backbone.modules():
#             if isinstance(m, StandardLayerNormAdapter):
#                 m.clip_running_var()
#
#     def freeze_bn_stats(self):
#         for m in self.backbone.modules():
#             if isinstance(m, StandardLayerNormAdapter):
#                 m.eval()
#
#     @property
#     def featurizer(self):
#         vit = self.backbone
#         return nn.Sequential(OrderedDict([
#             ('resize', self.resize),
#             ('conv_proj', vit.conv_proj),
#             ('encoder', vit.encoder),
#             ('encoder_norm', vit.encoder.ln),
#             ('flatten', nn.Flatten()),
#         ]))
#
#     @property
#     def classifier(self):
#         return self.backbone.heads
#
#     # 保留set_layers_to_adapt和surgical方法，与原逻辑一致
#     def set_layers_to_adapt(self, mode='all'):
#         print(mode)
#         self.backbone.train()
#
#         if mode in ['bn_all', 'bn_stat', 'bn_params']:
#             self.backbone.requires_grad_(False)
#             for m in self.backbone.modules():
#                 if isinstance(m, (StandardLayerNormAdapter)):
#                     if mode in ['bn_all', 'bn_params']:
#                         m.weight.requires_grad_(True)
#                         m.bias.requires_grad_(True)
#
#         elif mode == 'tent':
#             self.backbone.requires_grad_(False)
#             for m in self.backbone.modules():
#                 if isinstance(m, (StandardLayerNormAdapter)):
#                     m.weight.requires_grad_(True)
#                     m.bias.requires_grad_(True)
#
#         elif mode == 'last_layer':
#             self.backbone.requires_grad_(False)
#             self.backbone.heads.requires_grad_(True)
#
#         elif mode == 'last_bias':
#             self.backbone.requires_grad_(False)
#             self.backbone.heads.bias.requires_grad_(True)
#
#         elif mode == 'first_conv':
#             self.backbone.requires_grad_(False)
#             self.backbone.conv_proj.requires_grad_(True)
#
#         elif mode == 'encoder_layers':
#             self.backbone.requires_grad_(False)
#             self.backbone.encoder.requires_grad_(True)
#
#         elif mode == 'attention_layers':
#             self.backbone.requires_grad_(False)
#             for layer in self.backbone.encoder.layers:
#                 layer.self_attention.requires_grad_(True)
#
#         elif mode == 'mlp_layers':
#             self.backbone.requires_grad_(False)
#             for layer in self.backbone.encoder.layers:
#                 layer.mlp.requires_grad_(True)
#
#         elif mode == 'all':
#             pass
#
#         else:
#             raise NotImplementedError
#
#     def surgical(self, mode='all'):
#         self.backbone.requires_grad_(False)
#
#         if mode == 'encoder':
#             self.backbone.encoder.requires_grad_(True)
#
#         elif mode == 'last_layer':
#             self.backbone.heads.requires_grad_(True)
#
#         elif mode == 'patch_embed':
#             self.backbone.conv_proj.requires_grad_(True)
#
#         elif mode == 'attention':
#             for layer in self.backbone.encoder.layers:
#                 layer.self_attention.requires_grad_(True)
#
#         elif mode == 'mlp':
#             for layer in self.backbone.encoder.layers:
#                 layer.mlp.requires_grad_(True)
#
#         else:
#             raise NotImplementedError
import torch
import torch.nn as nn
from torchvision.transforms import Resize
from collections import OrderedDict
import timm
from .Model import Model
from .MyLayerNorm import StandardLayerNormAdapter


class ViT(Model):
    def __init__(self, shape_out):
        super(ViT, self).__init__()

        # 使用 timm 的 vit_small (ViT-S/16)，num_classes=0 表示输出特征而非分类结果
        self.backbone = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
        self.feature_dim = 384  # ViT-S/16的特征维度固定为384

        # 自定义分类头（输入维度必须匹配backbone输出的特征维度）
        self.head = nn.Linear(self.feature_dim, shape_out)

        # 图像尺寸调整
        self.resize = Resize((224, 224))
        self.drop_last = True

    def forward(self, x):
        # 单通道转三通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # 调整尺寸后提取特征，再通过分类头
        x = self.resize(x)
        features = self.backbone(x)  # backbone输出 (batch_size, 384) 的特征
        return self.head(features)

    # ========== LayerNorm 适配相关方法 ==========
    def change_bn(self, mode='grad', prior=0):
        if mode != 'grad':
            return

        def _wrap_ln(module):
            for name, child in module.named_children():
                if isinstance(child, nn.LayerNorm):
                    setattr(module, name, StandardLayerNormAdapter(child))
                else:
                    _wrap_ln(child)

        _wrap_ln(self.backbone)

    def set_running_stat_grads(self):
        for m in self.backbone.modules():
            if isinstance(m, StandardLayerNormAdapter):
                m.set_running_stat_grads()

    def clip_bn_running_vars(self):
        for m in self.backbone.modules():
            if isinstance(m, StandardLayerNormAdapter):
                m.clip_running_var()

    def freeze_bn_stats(self):
        for m in self.backbone.modules():
            if isinstance(m, StandardLayerNormAdapter):
                m.eval()

    # ========== 参数冻结控制 ==========
    def set_layers_to_adapt(self, mode='all'):
        self.backbone.train()

        if mode in ['bn_all', 'bn_stat', 'bn_params']:
            self.backbone.requires_grad_(False)
            for m in self.backbone.modules():
                if isinstance(m, StandardLayerNormAdapter):
                    if mode in ['bn_all', 'bn_params']:
                        m.meta_scale.requires_grad_(True)
                        m.meta_shift.requires_grad_(True)

        elif mode == 'tent':
            self.backbone.requires_grad_(False)
            for m in self.backbone.modules():
                if isinstance(m, StandardLayerNormAdapter):
                    m.meta_scale.requires_grad_(True)
                    m.meta_shift.requires_grad_(True)

        elif mode == 'last_layer':
            self.backbone.requires_grad_(False)
            self.head.requires_grad_(True)

        elif mode == 'last_bias':
            self.backbone.requires_grad_(False)
            self.head.bias.requires_grad_(True)

        elif mode == 'first_conv':
            self.backbone.requires_grad_(False)
            self.backbone.patch_embed.proj.requires_grad_(True)

        elif mode == 'encoder_layers':
            self.backbone.requires_grad_(False)
            self.backbone.blocks.requires_grad_(True)

        elif mode == 'attention_layers':
            self.backbone.requires_grad_(False)
            for block in self.backbone.blocks:
                block.attn.requires_grad_(True)

        elif mode == 'mlp_layers':
            self.backbone.requires_grad_(False)
            for block in self.backbone.blocks:
                block.mlp.requires_grad_(True)

        elif mode == 'all':
            pass

        else:
            raise NotImplementedError(f"Unsupported adapt mode: {mode}")

    # ========== 修复核心：适配Surgical算法的surgical方法 ==========
    def surgical(self, mode='all'):
        """
        适配Surgical算法的层选择方法，支持以下mode：
        - encoder: 整个编码器层
        - last_layer: 分类头
        - patch_embed: 补丁嵌入层（对应CNN的block1）
        - attention: 注意力层（对应CNN的block2/3）
        - mlp: MLP层（对应CNN的block4）
        """
        # 先冻结所有参数
        self.backbone.requires_grad_(False)
        self.head.requires_grad_(False)

        if mode == 'encoder':
            # 启用整个编码器层（对应CNN的多个block）
            self.backbone.encoder.requires_grad_(True) if hasattr(self.backbone, 'encoder') else self.backbone.blocks.requires_grad_(True)

        elif mode == 'last_layer':
            # 启用分类头（和原有逻辑一致）
            self.head.requires_grad_(True)

        elif mode == 'patch_embed':
            # 启用补丁嵌入层（对应CNN的block1）
            self.backbone.patch_embed.requires_grad_(True) if hasattr(self.backbone, 'patch_embed') else self.backbone.conv_proj.requires_grad_(True)

        elif mode == 'attention':
            # 启用注意力层（对应CNN的block2/3）
            if hasattr(self.backbone, 'blocks'):
                for block in self.backbone.blocks:
                    block.attn.requires_grad_(True)
            elif hasattr(self.backbone, 'encoder'):
                for layer in self.backbone.encoder.layers:
                    layer.self_attention.requires_grad_(True)

        elif mode == 'mlp':
            # 启用MLP层（对应CNN的block4）
            if hasattr(self.backbone, 'blocks'):
                for block in self.backbone.blocks:
                    block.mlp.requires_grad_(True)
            elif hasattr(self.backbone, 'encoder'):
                for layer in self.backbone.encoder.layers:
                    layer.mlp.requires_grad_(True)

        else:
            raise NotImplementedError(f"Unsupported surgical mode for ViT: {mode}")

    # ========== 特征提取接口 ==========
    @property
    def featurizer(self):
        # 定义特征提取器：返回ViT backbone的输出（384维特征）
        class Featurizer(nn.Module):
            def __init__(self, resize, backbone):
                super().__init__()
                self.resize = resize
                self.backbone = backbone

            def forward(self, x):
                # 单通道转三通道（和forward方法保持一致）
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)
                x = self.resize(x)
                return self.backbone(x)  # 输出 (batch_size, 384)

        return Featurizer(self.resize, self.backbone)

    @property
    def classifier(self):
        # 分类器就是自定义的head层
        return self.head

    # ========== 辅助方法：获取可训练参数 ==========
    @property
    def trainable_parameters(self):
        """兼容SurgicalClient中获取可训练参数的逻辑"""
        return [p for p in self.parameters() if p.requires_grad]
