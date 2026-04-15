import torch
import torch.nn as nn

class StandardLayerNormAdapter(nn.Module):
    """
    零侵入式外挂：标准 LayerNorm + 可学习 meta_scale / meta_shift
    """
    def __init__(self, original_ln: nn.LayerNorm):
        super().__init__()
        # 保存原始 LayerNorm（不动）
        self.ln = original_ln

        # 把可学习参数直接注册为 Parameter，并放到同一设备
        device = original_ln.weight.device
        self.meta_scale  = nn.Parameter(torch.ones_like(original_ln.weight, device=device))
        self.meta_shift  = nn.Parameter(torch.zeros_like(original_ln.bias,  device=device))

    def forward(self, x):
        y = self.ln(x)                       # 标准 LayerNorm
        # 广播到 (..., C)
        shape = [1]*(x.dim()-1) + [-1]
        return y * self.meta_scale.view(shape) + self.meta_shift.view(shape)

    # ATP 占位接口
    def set_running_stat_grads(self): pass
    def clip_running_var(self):       pass


# class MyLayerNorm(nn.Module):
#     def __init__(self, ln):
#         super(MyLayerNorm, self).__init__()
#
#         self.normalized_shape = ln.normalized_shape
#         self.eps = ln.eps
#
#         # 复制原始参数并确保在相同设备上
#         self.weight = nn.Parameter(ln.weight.detach().clone())
#         self.bias = nn.Parameter(ln.bias.detach().clone())
#
#         # 注册缓冲区并确保在相同设备上
#         self.register_buffer('running_mean', torch.zeros(ln.normalized_shape, device=ln.weight.device))
#         self.register_buffer('running_var', torch.ones(ln.normalized_shape, device=ln.weight.device))
#         self.num_batches_tracked = 0
#
#         self.snapshot_mean = None
#         self.snapshot_var = None
#
#         self.training = True
#
#     def forward(self, x):
#         # 确保所有张量在同一设备上
#         weight = self.weight.to(x.device)
#         bias = self.bias.to(x.device)
#
#         # 计算当前批次的统计量
#         if self.training:
#             with torch.no_grad():
#                 # 计算除最后一个维度外的所有维度的均值和方差
#                 self.snapshot_mean = x.mean(dim=tuple(range(x.dim() - 1)), keepdim=True)
#                 self.snapshot_var = x.var(dim=tuple(range(x.dim() - 1)), keepdim=True, unbiased=False)
#                 self.num_batches_tracked += 1
#
#         if not self.training:
#             # 使用运行统计量
#             running_mean = self.running_mean.to(x.device)
#             running_var = self.running_var.to(x.device)
#             mean = running_mean.view((1,) * (x.dim() - 1) + (-1,))
#             var = running_var.view((1,) * (x.dim() - 1) + (-1,))
#         else:
#             # 使用当前批次的统计量
#             mean = self.snapshot_mean
#             var = self.snapshot_var
#
#         # 应用归一化
#         x = (x - mean) / torch.sqrt(var + self.eps)
#
#         # 应用缩放和平移
#         weight = weight.view((1,) * (x.dim() - 1) + (-1,))
#         bias = bias.view((1,) * (x.dim() - 1) + (-1,))
#
#         return x * weight + bias
#
#     def set_running_stat_grads(self):
#         if self.snapshot_mean is not None and self.snapshot_var is not None:
#             with torch.no_grad():
#                 self.running_mean.grad = self.running_mean - self.snapshot_mean.squeeze().to(self.running_mean.device)
#                 self.running_var.grad = self.running_var - self.snapshot_var.squeeze().to(self.running_var.device)
#
#     def clip_running_var(self):
#         with torch.no_grad():
#             self.running_var.clamp_(min=0)
#
#
# class ModifiedLayerNorm(nn.Module):
#     """
#     LayerNorm with modified forward pass, similar to the BatchNorm version
#     """
#
#     def __init__(self, ln, prior):
#         super(ModifiedLayerNorm, self).__init__()
#
#         self.normalized_shape = ln.normalized_shape
#         self.eps = ln.eps
#
#         self.weight = ln.weight
#         self.bias = ln.bias
#
#         self.prior = prior
#
#         # LayerNorm doesn't have running statistics, so we add custom tracking
#         self.running_mean = nn.Parameter(torch.zeros(ln.normalized_shape), requires_grad=False)
#         self.running_var = nn.Parameter(torch.ones(ln.normalized_shape), requires_grad=False)
#
#         self.training = ln.training
#
#     def forward(self, input):
#         # Calculate current batch statistics
#         est_mean = input.mean(dim=tuple(range(1, input.dim())))
#         est_var = input.var(dim=tuple(range(1, input.dim())), unbiased=False)
#
#         # Update running statistics
#         if self.training:
#             momentum = 0.9
#             self.running_mean.data = (1 - momentum) * self.running_mean.data + momentum * est_mean.detach()
#             self.running_var.data = (1 - momentum) * self.running_var.data + momentum * est_var.detach()
#
#         # Blend running and current statistics based on prior
#         running_mean = self.prior * self.running_mean + (1 - self.prior) * est_mean
#         running_var = self.prior * self.running_var + (1 - self.prior) * est_var
#
#         # Apply normalization
#         normalized = (input - running_mean.view((-1,) + (1,) * (input.dim() - 1))) / torch.sqrt(
#             running_var.view((-1,) + (1,) * (input.dim() - 1)) + self.eps)
#         return normalized * self.weight.view((-1,) + (1,) * (input.dim() - 1)) + self.bias.view(
#             (-1,) + (1,) * (input.dim() - 1))


import torch
import torch.nn as nn
import torch.nn.functional as F


class MyLayerNorm(nn.LayerNorm):
    """自定义LayerNorm，支持运行时统计量梯度"""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, device=None, dtype=None):
        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)
        self.running_mean = torch.zeros(normalized_shape, device=device, dtype=dtype)
        self.running_var = torch.ones(normalized_shape, device=device, dtype=dtype)
        self.num_batches_tracked = torch.tensor(0, device=device)

    def forward(self, x):
        # 前向传播时更新运行时统计量
        if self.training:
            mean = x.mean([0] + list(range(2, x.dim())))
            var = x.var([0] + list(range(2, x.dim())), unbiased=False)

            # 更新运行时统计量
            self.running_mean = (self.running_mean * self.num_batches_tracked + mean) / (self.num_batches_tracked + 1)
            self.running_var = (self.running_var * self.num_batches_tracked + var) / (self.num_batches_tracked + 1)
            self.num_batches_tracked += 1

        return F.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )

    def set_running_stat_grads(self):
        """为运行时统计量启用梯度"""
        self.running_mean.requires_grad_(True)
        self.running_var.requires_grad_(True)

    def clip_running_var(self):
        """限制运行时方差的范围"""
        self.running_var.clamp_(min=1e-6)


class ModifiedLayerNorm(MyLayerNorm):
    """带先验的LayerNorm"""

    def __init__(self, layer_norm, prior=0.1):
        super().__init__(
            normalized_shape=layer_norm.normalized_shape,
            eps=layer_norm.eps,
            elementwise_affine=layer_norm.elementwise_affine,
            device=layer_norm.weight.device if layer_norm.elementwise_affine else None,
            dtype=layer_norm.weight.dtype if layer_norm.elementwise_affine else None
        )
        # 复制原有参数
        if layer_norm.elementwise_affine:
            self.weight.data = layer_norm.weight.data.clone()
            self.bias.data = layer_norm.bias.data.clone()
        self.prior = prior

    def forward(self, x):
        # 结合先验的归一化
        mean = x.mean([0] + list(range(2, x.dim())))
        var = x.var([0] + list(range(2, x.dim())), unbiased=False)

        # 融合先验
        mean = (1 - self.prior) * mean + self.prior * self.running_mean
        var = (1 - self.prior) * var + self.prior * self.running_var

        return F.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )