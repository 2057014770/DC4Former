from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair as to_2tuple

from mmseg.models.builder import BACKBONES
from mmengine.model import BaseModule
from mmengine.model.weight_init import constant_init, trunc_normal_init
from mmengine.runner import CheckpointLoader
from mmengine.logging import print_log


# ==========================================================
# Basic utilities
# ==========================================================
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


# ==========================================================
# Strong C4 spatial operators
# ==========================================================
class C4SymConv2d(nn.Conv2d):
    """C4 weight-symmetrized Conv2d for stride-1 spatial convolutions.

    This layer is used for CPE and IRB depthwise convolutions. It averages the
    four rotated kernels and performs a single convolution. Therefore it adds no
    learnable parameters and keeps the extra compute low for stride-1 operators.
    """

    def __init__(self, *args, group_average: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_average = group_average
        if self.stride != (1, 1) and self.stride != 1:
            raise ValueError(
                'C4SymConv2d is only valid for stride=1. '
                'Use C4FeatureSymConv2d for strided downsampling.'
            )

    def forward(self, x):
        if (not self.group_average) or self.kernel_size == (1, 1):
            return self._conv_forward(x, self.weight, self.bias)

        w = self.weight
        sym_weight = (
            w
            + torch.rot90(w, 1, dims=(2, 3))
            + torch.rot90(w, 2, dims=(2, 3))
            + torch.rot90(w, 3, dims=(2, 3))
        ) * 0.25
        return self._conv_forward(x, sym_weight, self.bias)


class C4FeatureSymConv2d(nn.Conv2d):
    """C4 feature-rotation Conv2d for strided/downsampling layers.

    For stride>1, rotating kernels is not equivalent to rotating features before
    downsampling. Therefore the downsampling layers use the stronger formulation:
        S(x) = 1/4 * sum_k R_-k Conv(R_k x).

    This is the only intentionally more expensive C4 convolution path. It is
    used only in Stem/PatchEmbed, which preserves strong 0/90/180/270 behavior
    without adding parameters.
    """

    def __init__(self, *args, group_average: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_average = group_average

    def forward(self, x):
        if (not self.group_average) or self.kernel_size == (1, 1):
            return self._conv_forward(x, self.weight, self.bias)

        y_sum = None
        for k in range(4):
            xr = torch.rot90(x, k, dims=(2, 3))
            yr = self._conv_forward(xr, self.weight, self.bias)
            yr = torch.rot90(yr, -k, dims=(2, 3))
            y_sum = yr if y_sum is None else y_sum + yr
        return y_sum * 0.25


def make_conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
                dilation=1, groups=1, bias=True, padding_mode='zeros',
                c4: bool = True, c4_downsample: bool = True):
    """Create a spatial convolution for the DC4Former backbone.

    - 1x1 conv: ordinary Conv2d.
    - k>1, stride=1: C4 weight symmetrization, no extra parameters, one conv.
    - k>1, stride>1: C4 feature-rotation downsampling when c4_downsample=True.
    """
    kernel_size_pair = to_2tuple(kernel_size)
    stride_pair = to_2tuple(stride)

    if c4 and kernel_size_pair != (1, 1):
        if stride_pair == (1, 1):
            return C4SymConv2d(
                in_channels, out_channels, kernel_size_pair, stride, padding,
                dilation, groups, bias, padding_mode
            )
        if c4_downsample:
            return C4FeatureSymConv2d(
                in_channels, out_channels, kernel_size_pair, stride, padding,
                dilation, groups, bias, padding_mode
            )

    return nn.Conv2d(
        in_channels, out_channels, kernel_size_pair, stride, padding,
        dilation, groups, bias, padding_mode
    )


# ==========================================================
# DC4Former components
# ==========================================================
class IRB(nn.Module):
    """DC4Former IRB with C4-symmetrized depthwise spatial convolution."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 ksize=3, act_layer=nn.GELU, extra_act=False, drop=0.,
                 c4_conv=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.act = act_layer()
        self.act1 = act_layer() if extra_act else nn.Identity()
        self.conv = make_conv2d(
            hidden_features, hidden_features, kernel_size=ksize,
            padding=ksize // 2, stride=1, groups=hidden_features,
            bias=True, c4=c4_conv
        )
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.conv(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x.reshape(B, C, -1).permute(0, 2, 1)


class PoolingAttention(nn.Module):
    """Strong-C4 pooling attention with constrained lightweight ESAF.

    This is the TGRS-oriented version:
    1) Four rot90 branches are still computed to preserve the Strong-C4 basis.
    2) Uniform mean fusion remains the dominant path.
    3) A lightweight spatial scorer can only add a bounded residual correction:
           W_i = (1 - alpha) / 4 + alpha * softmax(S_i / tau)
       where alpha is learnable but capped by ``fusion_alpha_max``.

    Compared with free ESAF, this avoids blind branch over-confidence under
    out-of-group angles while restoring some spatial/detail adaptivity over pure
    mean fusion. The scorer is zero-initialized, so the initial forward is
    exactly equivalent to C4 mean fusion.
    """

    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., pooled_sizes=(11, 8, 6, 4),
                 q_pooled_size=1, q_conv=False, c4_conv=True,
                 c4_attention=True, fusion_type='cl_esaf',
                 fusion_alpha_max=0.25, fusion_alpha_init=-2.0,
                 fusion_temp_init=2.0):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pooled_sizes = list(pooled_sizes)
        self.eps = 0.001
        self.norm = nn.LayerNorm(dim)
        self.q_pooled_size = q_pooled_size
        self.c4_attention = c4_attention

        # 'mean' is the strict C4 ablation. 'cl_esaf' is the proposed version.
        self.fusion_type = str(fusion_type).lower()
        self.fusion_alpha_max = float(fusion_alpha_max)
        if self.fusion_type not in ('mean', 'cl_esaf', 'constrained_esaf'):
            raise ValueError(f'Unsupported fusion_type: {fusion_type}')

        if self.fusion_type in ('cl_esaf', 'constrained_esaf'):
            hidden = max(4, min(dim // 8, 32))
            self.fusion_scorer = nn.Sequential(
                nn.Conv2d(dim, hidden, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Conv2d(hidden, 1, kernel_size=1, bias=True)
            )
            # Starts from uniform adaptive weights; therefore the whole module
            # starts exactly as mean fusion.
            nn.init.constant_(self.fusion_scorer[-1].weight, 0.0)
            nn.init.constant_(self.fusion_scorer[-1].bias, 0.0)
            self.fusion_log_temp = nn.Parameter(torch.tensor(float(fusion_temp_init)))
            self.fusion_alpha_logit = nn.Parameter(torch.tensor(float(fusion_alpha_init)))
        else:
            self.fusion_scorer = None
            self.register_parameter('fusion_log_temp', None)
            self.register_parameter('fusion_alpha_logit', None)

        if q_conv and self.q_pooled_size > 1:
            self.q_conv = make_conv2d(
                dim, dim, kernel_size=3, padding=1, stride=1,
                groups=dim, bias=True, c4=c4_conv
            )
            self.q_norm = nn.LayerNorm(dim)
        else:
            self.q_conv = None
            self.q_norm = None

    def _pool_size(self, base_size, H, W):
        H = float(H)
        W = float(W)
        if W >= H:
            return (max(1, int(round(base_size))),
                    max(1, int(round(W * float(base_size) / H + self.eps))))
        return (max(1, int(round(H * float(base_size) / W + self.eps))),
                max(1, int(round(base_size))))

    def _forward_one_orientation(self, x_2d, d_convs, q_pool_hw, kv_pool_hws):
        assert d_convs is not None, 'PoolingAttention requires stage-level d_convs.'
        B, C, H, W = x_2d.shape

        if self.q_pooled_size > 1:
            q = F.adaptive_avg_pool2d(x_2d, q_pool_hw)
            if self.q_conv is not None:
                q = q + self.q_conv(q)
                q_tokens = self.q_norm(q.flatten(2).transpose(1, 2))
            else:
                q_tokens = q.flatten(2).transpose(1, 2)
        else:
            if self.q_conv is not None:
                q = x_2d + self.q_conv(x_2d)
                q_tokens = self.q_norm(q.flatten(2).transpose(1, 2))
            else:
                q_tokens = x_2d.flatten(2).transpose(1, 2)
            q = x_2d

        Hq, Wq = q.shape[2:]
        q = self.q(q_tokens).reshape(B, -1, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3).contiguous()

        pools = []
        for pool_hw, d_conv in zip(kv_pool_hws, d_convs):
            pool = F.adaptive_avg_pool2d(x_2d, pool_hw)
            pool = pool + d_conv(pool)
            pools.append(pool.flatten(2).transpose(1, 2))

        pools = self.norm(torch.cat(pools, dim=1))
        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, -1, C)
        return out, Hq, Wq

    def _fuse_c4_branches(self, out_branches):
        """Fuse aligned C4 branches.

        out_branches: list of four tensors [B, C, Hq, Wq].
        """
        stacked = torch.stack(out_branches, dim=1)  # [B, 4, C, Hq, Wq]

        if self.fusion_type == 'mean':
            return stacked.mean(dim=1)

        score_list = [self.fusion_scorer(branch) for branch in out_branches]
        logits = torch.cat(score_list, dim=1)  # [B, 4, Hq, Wq]
        temperature = F.softplus(self.fusion_log_temp) + 1e-6
        adaptive_w = F.softmax(logits / temperature, dim=1)

        # Bounded adaptive correction. Even when the scorer becomes confident,
        # it cannot fully override the stable mean-fusion path.
        alpha = self.fusion_alpha_max * torch.sigmoid(self.fusion_alpha_logit)
        weights = (1.0 - alpha) * 0.25 + alpha * adaptive_w
        return (stacked * weights.unsqueeze(2)).sum(dim=1)

    def forward(self, x, H, W, d_convs=None):
        B, N, C = x.shape
        H, W = int(H), int(W)
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)

        base_q_pool = self._pool_size(self.q_pooled_size, H, W) if self.q_pooled_size > 1 else (H, W)
        base_kv_pools = [self._pool_size(s, H, W) for s in self.pooled_sizes]

        if not self.c4_attention:
            out, Hq, Wq = self._forward_one_orientation(x_2d, d_convs, base_q_pool, base_kv_pools)
            out = self.proj(out)
            out = self.proj_drop(out)
            if self.q_pooled_size > 1:
                out = out.transpose(1, 2).reshape(B, C, Hq, Wq)
                out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
                out = out.flatten(2).transpose(1, 2)
            return out

        out_branches = []
        for k in range(4):
            x_rot = torch.rot90(x_2d, k, dims=(2, 3))

            if k % 2 == 1:
                curr_q_pool = (base_q_pool[1], base_q_pool[0])
                curr_kv_pools = [(s[1], s[0]) for s in base_kv_pools]
            else:
                curr_q_pool = base_q_pool
                curr_kv_pools = base_kv_pools

            out_rot, Hq, Wq = self._forward_one_orientation(x_rot, d_convs, curr_q_pool, curr_kv_pools)
            out_rot = out_rot.transpose(1, 2).reshape(B, C, Hq, Wq)
            out_aligned = torch.rot90(out_rot, -k, dims=(2, 3))
            out_branches.append(out_aligned)

        out_low = self._fuse_c4_branches(out_branches)
        Hq_base, Wq_base = out_low.shape[2:]

        out = self.proj(out_low.flatten(2).transpose(1, 2))
        out = self.proj_drop(out)
        out = out.transpose(1, 2).reshape(B, C, Hq_base, Wq_base)

        if self.q_pooled_size > 1:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out.flatten(2).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, ls=False,
                 pooled_sizes=(12, 16, 20, 24), q_pooled_size=1,
                 q_conv=False, extra_act=False, use_prenorm=False,
                 c4_conv=True, c4_attention=True, fusion_type='cl_esaf',
                 fusion_alpha_max=0.25, fusion_alpha_init=-2.0,
                 fusion_temp_init=2.0):
        super().__init__()
        self.pre_norm = nn.BatchNorm2d(dim) if use_prenorm else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.attn = PoolingAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, pooled_sizes=pooled_sizes,
            q_pooled_size=q_pooled_size, q_conv=q_conv,
            c4_conv=c4_conv, c4_attention=c4_attention,
            fusion_type=fusion_type, fusion_alpha_max=fusion_alpha_max,
            fusion_alpha_init=fusion_alpha_init,
            fusion_temp_init=fusion_temp_init
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = IRB(
            in_features=dim, hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer, drop=drop, ksize=3,
            extra_act=extra_act, c4_conv=c4_conv
        )

        self.ls = ls
        if self.ls:
            layer_scale_init_value = 1e-6
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.cpe = make_conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                               groups=dim, bias=True, c4=c4_conv)

    def forward(self, x, H, W, d_convs=None):
        B, N, C = x.shape
        x_2d = x.permute(0, 2, 1).reshape(B, C, H, W)
        x_2d = self.pre_norm(x_2d)
        x_2d = self.cpe(x_2d) + x_2d
        x = x_2d.reshape(B, C, -1).permute(0, 2, 1)

        if self.ls:
            x = x + self.drop_path(self.layer_scale_1[None, None, :] * self.attn(self.norm1(x), H, W, d_convs=d_convs))
            x = x + self.drop_path(self.layer_scale_2[None, None, :] * self.mlp(self.norm2(x), H, W))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), H, W, d_convs=d_convs))
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class PatchEmbed(nn.Module):
    """Overlapped image-to-patch embedding with optional strong C4 downsampling."""

    def __init__(self, img_size=224, patch_size=16, kernel_size=3,
                 in_chans=3, embed_dim=768, overlap=True,
                 c4_conv=True, c4_downsample=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f'img_size {img_size} should be divided by patch_size {patch_size}.'
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        if not overlap:
            self.proj = make_conv2d(
                in_chans, embed_dim, kernel_size=patch_size,
                stride=patch_size, padding=0, c4=c4_conv,
                c4_downsample=c4_downsample
            )
        else:
            self.proj = make_conv2d(
                in_chans, embed_dim, kernel_size=kernel_size,
                stride=patch_size, padding=kernel_size // 2, c4=c4_conv,
                c4_downsample=c4_downsample
            )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, (H, W)


class Stem(nn.Module):
    """DC4Former stem with optional strong C4 downsampling."""

    def __init__(self, in_chans=3, out_chans=64, patch_size=4,
                 c4_conv=True, c4_downsample=True):
        super().__init__()
        self.conv1 = nn.Sequential(
            make_conv2d(in_chans, out_chans // 2, 3, 2, 1,
                        c4=c4_conv, c4_downsample=c4_downsample),
            nn.BatchNorm2d(out_chans // 2),
            nn.GELU(),
            make_conv2d(out_chans // 2, out_chans // 2, 3, 1, 1,
                        c4=c4_conv),
            nn.BatchNorm2d(out_chans // 2),
            nn.GELU(),
            make_conv2d(out_chans // 2, out_chans, 3, 2, 1,
                        c4=c4_conv, c4_downsample=c4_downsample),
            nn.BatchNorm2d(out_chans),
            nn.GELU(),
            make_conv2d(out_chans, out_chans, 3, 1, 1,
                        c4=c4_conv),
        )
        self.norm = nn.LayerNorm(out_chans, eps=1e-6)
        self.patch_size = to_2tuple(patch_size)

    def forward(self, x):
        x = self.conv1(x)
        _, _, H, W = x.shape
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, (H, W)


@BACKBONES.register_module(force=True)
class DC4Former(BaseModule):
    """DC4Former backbone with C4-consistent spatial processing.

    Based on the original lightweight backbone, this paper-oriented version keeps
    only three C4 components:
      1. C4 weight symmetrization for stride-1 spatial operators.
      2. C4 feature-rotation averaging for strided downsampling operators.
      3. C4 multi-view pooling attention with constrained lightweight ESAF.

    The attention fusion keeps mean aggregation as the dominant path and only
    allows a bounded spatially adaptive residual correction. It deliberately removes high-resolution fusion, adaptive orientation scorer,
    detail residuals, variance modulation, q-max pooling, and anti-alias switches
    to avoid engineering-style accumulation of tricks.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 num_classes=1000, embed_dims=(64, 128, 320, 512),
                 num_heads=(2, 4, 10, 16), mlp_ratios=(8, 8, 4, 4),
                 qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 q_pooled_sizes=(16, 16, 16, 16), depths=(3, 3, 12, 3),
                 q_conv=False, use_ls=True, extra_act=True, init_cfg=None,
                 c4_conv=True, c4_attention=True, c4_downsample=True,
                 fusion_type='cl_esaf', fusion_alpha_max=0.25,
                 fusion_alpha_init=-2.0, fusion_temp_init=2.0,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.depths = list(depths)
        self.embed_dims = list(embed_dims)
        self.embed_dim = embed_dims[-1]
        self.head = nn.Identity()
        self.c4_conv = c4_conv
        self.c4_attention = c4_attention
        self.c4_downsample = c4_downsample

        base_pooled_sizes = [11, 8, 6, 4]
        pooled_sizes = [
            [max(1, round(temp * q_pooled_sizes[i] / 16. + 0.001)) for temp in base_pooled_sizes]
            for i in range(4)
        ]

        self.patch_embed1 = Stem(
            in_chans=in_chans, out_chans=embed_dims[0], patch_size=4,
            c4_conv=c4_conv, c4_downsample=c4_downsample
        )
        self.patch_embed2 = PatchEmbed(
            img_size=img_size // 4, patch_size=2, in_chans=embed_dims[0],
            embed_dim=embed_dims[1], overlap=True, c4_conv=c4_conv,
            c4_downsample=c4_downsample
        )
        self.patch_embed3 = PatchEmbed(
            img_size=img_size // 8, patch_size=2, in_chans=embed_dims[1],
            embed_dim=embed_dims[2], overlap=True, c4_conv=c4_conv,
            c4_downsample=c4_downsample
        )
        self.patch_embed4 = PatchEmbed(
            img_size=img_size // 16, patch_size=2, in_chans=embed_dims[2],
            embed_dim=embed_dims[3], overlap=True, c4_conv=c4_conv,
            c4_downsample=c4_downsample
        )

        # Keep DC4Former's stage-level d_convs ordinary. They are already inside
        # the outer C4 attention branches, so nested C4 would add cost without a
        # clean paper contribution.
        self.d_convs1 = nn.ModuleList([
            nn.Conv2d(embed_dims[0], embed_dims[0], 3, 1, 1, groups=embed_dims[0])
            for _ in pooled_sizes[0]
        ])
        self.d_convs2 = nn.ModuleList([
            nn.Conv2d(embed_dims[1], embed_dims[1], 3, 1, 1, groups=embed_dims[1])
            for _ in pooled_sizes[1]
        ])
        self.d_convs3 = nn.ModuleList([
            nn.Conv2d(embed_dims[2], embed_dims[2], 3, 1, 1, groups=embed_dims[2])
            for _ in pooled_sizes[2]
        ])
        self.d_convs4 = nn.ModuleList([
            nn.Conv2d(embed_dims[3], embed_dims[3], 3, 1, 1, groups=embed_dims[3])
            for _ in pooled_sizes[3]
        ])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([
            Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[cur + i], ls=False,
                  norm_layer=norm_layer, pooled_sizes=pooled_sizes[0],
                  q_pooled_size=q_pooled_sizes[0], q_conv=q_conv,
                  extra_act=extra_act, c4_conv=c4_conv,
                  c4_attention=c4_attention, fusion_type=fusion_type,
                  fusion_alpha_max=fusion_alpha_max,
                  fusion_alpha_init=fusion_alpha_init,
                  fusion_temp_init=fusion_temp_init)
            for i in range(depths[0])
        ])

        cur += depths[0]
        self.block2 = nn.ModuleList([
            Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[cur + i], ls=False,
                  norm_layer=norm_layer, pooled_sizes=pooled_sizes[1],
                  q_pooled_size=q_pooled_sizes[1], q_conv=q_conv,
                  extra_act=extra_act, c4_conv=c4_conv,
                  c4_attention=c4_attention, fusion_type=fusion_type,
                  fusion_alpha_max=fusion_alpha_max,
                  fusion_alpha_init=fusion_alpha_init,
                  fusion_temp_init=fusion_temp_init)
            for i in range(depths[1])
        ])

        cur += depths[1]
        self.block3 = nn.ModuleList([
            Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[cur + i], ls=use_ls,
                  norm_layer=norm_layer, pooled_sizes=pooled_sizes[2],
                  q_pooled_size=q_pooled_sizes[2], q_conv=q_conv,
                  extra_act=extra_act,
                  use_prenorm=(i in [7, 15] if depths[2] > 20 else False),
                  c4_conv=c4_conv, c4_attention=c4_attention, fusion_type=fusion_type,
                  fusion_alpha_max=fusion_alpha_max,
                  fusion_alpha_init=fusion_alpha_init,
                  fusion_temp_init=fusion_temp_init)
            for i in range(depths[2])
        ])

        cur += depths[2]
        self.block4 = nn.ModuleList([
            Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[cur + i], ls=use_ls,
                  norm_layer=norm_layer, pooled_sizes=pooled_sizes[3],
                  q_pooled_size=q_pooled_sizes[3], q_conv=q_conv,
                  extra_act=extra_act, c4_conv=c4_conv,
                  c4_attention=c4_attention, fusion_type=fusion_type,
                  fusion_alpha_max=fusion_alpha_max,
                  fusion_alpha_init=fusion_alpha_init,
                  fusion_temp_init=fusion_temp_init)
            for i in range(depths[3])
        ])

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.apply(self._init_weights)

    def init_weights(self):
        if self.init_cfg is None:
            print_log(f'No pre-trained weights for {self.__class__.__name__}, training starts from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
        else:
            assert 'checkpoint' in self.init_cfg
            ckpt = CheckpointLoader.load_checkpoint(self.init_cfg['checkpoint'], logger=None, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v

            if len(state_dict) > 0 and list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print_log(f'Loaded checkpoint with {len(missing)} missing keys and {len(unexpected)} unexpected keys')
            if len(missing) > 0:
                print_log('First missing keys: ' + str(missing[:20]))
            if len(unexpected) > 0:
                print_log('First unexpected keys: ' + str(unexpected[:20]))

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            if hasattr(self.block1[i].drop_path, 'drop_prob'):
                self.block1[i].drop_path.drop_prob = dpr[cur + i]
        cur += self.depths[0]
        for i in range(self.depths[1]):
            if hasattr(self.block2[i].drop_path, 'drop_prob'):
                self.block2[i].drop_path.drop_prob = dpr[cur + i]
        cur += self.depths[1]
        for i in range(self.depths[2]):
            if hasattr(self.block3[i].drop_path, 'drop_prob'):
                self.block3[i].drop_path.drop_prob = dpr[cur + i]
        cur += self.depths[2]
        for i in range(self.depths[3]):
            if hasattr(self.block4[i].drop_path, 'drop_prob'):
                self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        x, (H, W) = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W, self.d_convs1)
        x_2d = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x_2d)

        x, (H, W) = self.patch_embed2(x_2d)
        for blk in self.block2:
            x = blk(x, H, W, self.d_convs2)
        x_2d = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x_2d)

        x, (H, W) = self.patch_embed3(x_2d)
        for blk in self.block3:
            x = blk(x, H, W, self.d_convs3)
        x_2d = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x_2d)

        x, (H, W) = self.patch_embed4(x_2d)
        for blk in self.block4:
            x = blk(x, H, W, self.d_convs4)
        x_2d = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x_2d)

        return outs

    def forward(self, x):
        return self.forward_features(x)

    def forward_for_fpn(self, x):
        return self.forward_features(x)


# Keep the legacy registry name available for existing experiment configs.
BACKBONES.register_module(name='LRFormer', module=DC4Former, force=True)
LRFormer = DC4Former


def _conv_filter(state_dict, patch_size=16):
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict
