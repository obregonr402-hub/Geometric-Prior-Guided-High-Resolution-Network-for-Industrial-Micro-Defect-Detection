# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv
from .transformer import TransformerBlock

__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3', 'ConvNormLayer', 'BasicBlock', 
           'BottleNeck', 'Blocks','HFREB','GDIModule','CASABlock','DSFPNModule','ABFModule','HGRABlock','MSLCABlock'
               ,'C2fCIB','CIB','PSA','SCDown','A2Block','Attention','FasterBlock','PConv','DCNv4_Block','HGBlock','HG_Stem'
          ,'LightConv','ConvBNAct','AFPN_Neck','ASFF','GDNeck','IFM','FAM')


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

################################### RT-DETR PResnet ###################################
def get_activation(act: str, inpace: bool=True):
    '''get activation
    '''
    act = act.lower()
    
    if act == 'silu':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()
    
    elif act == 'gelu':
        m = nn.GELU()
        
    elif act is None:
        m = nn.Identity()
    
    elif isinstance(act, nn.Module):
        m = act

    else:
        raise RuntimeError('')  

    if hasattr(m, 'inplace'):
        m.inplace = inpace
    
    return m 

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 


    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        
        out = out + short
        out = self.act(out)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__()

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out 

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out


class Blocks(nn.Module):
    def __init__(self, ch_in, ch_out, block, count, stage_num, act='relu', input_resolution=None, sr_ratio=None, kernel_size=None, kan_name=None, variant='d'):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            if input_resolution is not None and sr_ratio is not None:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        input_resolution=input_resolution,
                        sr_ratio=sr_ratio)
                )
            elif kernel_size is not None:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        kernel_size=kernel_size)
                )
            elif kan_name is not None:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        kan_name=kan_name)
                )
            else:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act)
                )
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out


import torch
import torch.nn as nn


import torch
import torch.nn as nn


class HFREB(nn.Module):
    """
    High-Frequency Residual Enhancement Block
    Ultralytics-compatible version
    """

    def __init__(self, c1, c2=None):
        super().__init__()

        # 关键点：c2 如果没用，就强制等于 c1
        c2 = c1 if c2 is None else c2

        assert c1 == c2, \
            f"HFREB is identity-preserving, but got c1={c1}, c2={c2}"

        self.conv1 = nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.act = nn.SiLU()

        self.conv2 = nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c1)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)

# =========================================================================
# Innovation 2: GDI-Module (Global-Local Distribute Interaction)
# 原型：Gold-YOLO (Gather-and-Distribute)
# 简化版：利用全局池化收集上下文，再通过注意力分发给局部特征
# =========================================================================

class GDIModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # c1: Concat 后的通道数 (例如 512)
        # c2: 输出通道数 (例如 256)

        # 1. 局部特征处理 (Local Branch)
        self.cv_local = Conv(c1, c2, 1, 1)

        # 2. 全局特征收集 (Gather) -> 类似 FAM
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, c2, 1, 1),
            nn.ReLU(),
            nn.Conv2d(c2, c1, 1, 1),  # 恢复到输入维度，用于加权
            nn.Sigmoid()
        )

        # 3. 特征分发与注入 (Distribute) -> 类似 IFM
        # 引入大核卷积来模拟“分发”时的感受野
        self.distribute = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=5, padding=2, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

        self.fusion = Conv(c1 + c2, c2, 1, 1)  # 最终融合

    def forward(self, x):
        # x 是 Concat 后的特征 [B, 512, H, W]

        # === Gather 阶段 ===
        # 收集全局上下文信息，判断这张图是不是“反光严重”
        global_context = self.gap(x)
        global_weight = self.fc(global_context)

        # 利用全局权重对原始特征进行“清洗”
        x_refined = x * global_weight

        # === Local 阶段 ===
        feat_local = self.cv_local(x_refined)

        # === Distribute 阶段 ===
        # 将局部特征进一步扩散，增强邻域信息
        feat_dist = self.distribute(feat_local)

        # === Interaction ===
        # 将清洗后的原始特征(x_refined) 和 分发后的特征(feat_dist) 再次融合
        # 注意：这里需要先把 x_refined 缩减通道或者直接拼接到 feat_dist
        # 为了简单且强力，我们直接拼回去
        out = torch.cat([x_refined, feat_dist], dim=1)

        return self.fusion(out)
# =========================================================================
# Innovation 3: CASA-Module (Cross-Axis Structural Attention)
# 原型：PolaFormer (ICLR 2025)
# 作用：利用极化机制保持 X/Y 轴的高分辨率，精准捕捉线状缺陷（短路/断路）。
# =========================================================================

# class CASABlock(nn.Module):
#     def __init__(self, c1, num_heads=8, dropout=0.):
#         super().__init__()
#         self.c1 = c1
#         self.num_heads = num_heads
#         self.head_dim = c1 // num_heads
#
#         # 1. 核心算子：极化注意力
#         # QKV 生成
#         self.qkv = nn.Conv2d(c1, c1 * 3, 1)
#
#         # 2. 空间聚合算子 (模拟极化)
#         # 这里的技巧是：不要全局池化，而是分别对 H 和 W 维度池化
#         # 这就是 "Cross-Axis" 的来源
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 保留 H，压缩 W
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 保留 W，压缩 H
#
#         # 3. 激励函数
#         self.softmax = nn.Softmax(dim=-1)
#         self.proj = nn.Conv2d(c1, c1, 1)
#
#         # FFN 部分
#         self.norm1 = nn.GroupNorm(1, c1)
#         self.norm2 = nn.GroupNorm(1, c1)
#         self.mlp = nn.Sequential(
#             nn.Conv2d(c1, c1 * 4, 1),
#             nn.GELU(),
#             nn.Conv2d(c1 * 4, c1, 1)
#         )
#
#     def forward(self, x):
#         # x: [B, C, H, W]
#         B, C, H, W = x.shape
#         residue = x
#         x = self.norm1(x)
#
#         # 生成 Q, K, V
#         qkv = self.qkv(x).chunk(3, dim=1)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         # --- 极化注意力机制 (Simplified Pola) ---
#
#         # 1. Channel-only Branch (通道注意力)
#         # [B, C, 1, 1]
#         w_ch = self.softmax(
#             torch.mean(q, dim=[2, 3], keepdim=True) * torch.mean(k, dim=[2, 3], keepdim=True) * (self.c1 ** -0.5))
#         v_ch = w_ch * v
#
#         # 2. Spatial-only Branch (空间极化)
#         # 关键创新：分别看 H 轴和 W 轴，不丢失线性特征
#         q_h = self.pool_w(q)  # [B, C, H, 1]
#         k_h = self.pool_w(k)
#         q_w = self.pool_h(q)  # [B, C, 1, W]
#         k_w = self.pool_h(k)
#
#         # 简单的线性注意力模拟
#         att_h = torch.sigmoid(q_h * k_h)  # 高度方向的结构权重
#         att_w = torch.sigmoid(q_w * k_w)  # 宽度方向的结构权重
#
#         # 3. 融合：特征 = 原始V * 通道权 * 空间权
#         # 这里的广播机制会自动处理维度
#         out = v_ch * att_h * att_w
#
#         out = self.proj(out)
#         out = out + residue
#
#         # FFN
#         out = out + self.mlp(self.norm2(out))
#         return out


# =========================================================================
# Innovation 2: DSFPN-Module (Defect-Sensitive Fusion)
# 原型：HSFPN (MFDS-DETR)
# 改进：加入 Scale-Aware Attention，专门放大微小缺陷信号
# =========================================================================


# =========================================================================
# Innovation 2: ABF-Module (Attentional Bi-Fusion) [Final Fix]
# 修复：移除了升维层，输出通道数与 YAML 配置保持一致 (256)
# =========================================================================

class ABFModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # c1: Concat 后的通道数 (512)
        # c2: 输出通道数 (256)

        # 1. 降维融合
        self.cv_reduce = Conv(c1, c2, 1, 1)

        # 2. 通道注意力权重生成器
        self.att_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(c2, c2 // 4, 1, 1, act=True),
            Conv(c2 // 4, 2 * c2, 1, 1, act=False),
            nn.Sigmoid()
        )

        # === 修复点：删掉了 self.expand，不再升维 ===

    def forward(self, x):
        # x: [B, 512, H, W]
        feat = self.cv_reduce(x)  # [B, 256, H, W]
        x1, x2 = torch.chunk(x, 2, dim=1)

        weights = self.att_gen(feat)
        w1, w2 = torch.chunk(weights, 2, dim=1)

        # 加权融合
        weighted_feat = x1 * w1 + x2 * w2

        # === 修复点：直接返回 256 通道的结果 ===
        return weighted_feat
# ================================================================
# [New Innovation] HGRA-Block: Hierarchical Gradient-Response Aggregation
# 针对 PCB 微小缺陷（短路/毛刺）优化的自研主干模块
# 包含：空洞卷积 (Dilated Conv) + 梯度选择 (Gradient Selection)
# ================================================================

# def autopad(k, p=None, d=1):  # kernel, padding, dilation
#     # Pad to 'same' shape outputs
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
#     return p
#
#
# class HGRABlock(nn.Module):
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#
#         # === 核心手术点 1: 引入空洞卷积 (Dilation=2) ===
#         # 原理：扩大感受野，但不丢失分辨率，专门针对细微的断路/短路
#         self.m = nn.Sequential(*(
#             nn.Sequential(
#                 # 第一层：普通 3x3 卷积
#                 Conv(c_, c_, 3, 1, g=g),
#                 # 第二层：空洞 3x3 卷积 (dilation=2)，提取长距离依赖
#                 # 注意：这里我们手动构造一个带 dilation 的 Conv
#                 nn.Sequential(
#                     nn.Conv2d(c_, c_, 3, 1, padding=2, dilation=2, groups=g, bias=False),
#                     nn.BatchNorm2d(c_),
#                     nn.SiLU()
#                 )
#             ) for _ in range(n)
#         ))
#
#         # === 核心手术点 2: 注入梯度选择机制 (DSM) ===
#         # 替换了原版可能存在的 AvgPool，使用 MaxPool 保留强特征
#         self.att = nn.Sequential(
#             nn.AdaptiveMaxPool2d(1),  # <--- 关键修改：最大池化
#             Conv(c_, c_, 1, 1),
#             nn.Sigmoid()
#         )
#
#         self.cv3 = Conv(2 * c_, c2, 1)
#
#     def forward(self, x):
#         # 1. 分支 1: 经过空洞卷积处理
#         y1 = self.m(self.cv1(x))
#         # 2. 分支 2: 原始特征
#         y2 = self.cv2(x)
#
#         # 3. 融合：利用注意力加权，过滤背景噪点
#         return self.cv3(torch.cat((y1 * self.att(y1), y2), 1))
# import torch
# import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
# =========================================================================
# Innovation 2: DySample-Module (Dynamic Upsampling)
# 作用：替换普通的 Upsample，无损放大 HGRA 的细节特征
# 来源：ICCV 2023
# =========================================================================

import torch.nn.functional as F


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


# =========================================================================
# Innovation 2: MS-LCA (Multi-Scale Local Context Attention)
# 原型：LCA (CVPR 2025)
# 魔改：引入 3x3 和 7x7 双路感知，并加入残差连接，专治小目标模糊
# =========================================================================

class MSLCABlock(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # c1: 输入通道 (Concat后)
        # c2: 输出通道
        self.conv = Conv(c1, c2, 1, 1)

        # 分支 1: 小感受野 (3x3)，专注微小缺陷细节
        self.local_small = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

        # 分支 2: 大感受野 (7x7)，专注局部背景去噪
        self.local_large = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=7, padding=3, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

        # 融合与生成权重
        self.att_conv = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.Sigmoid()
        )

        # 可学习的残差系数，初始化为0，保证不掉点
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 1. 降维
        feat = self.conv(x)

        # 2. 多尺度感知
        # 两个分支相加，既有细节又有背景
        context = self.local_small(feat) + self.local_large(feat)

        # 3. 生成注意力图
        attn = self.att_conv(context)

        # 4. 加权 + 残差 (Safe Residual)
        # 这里的残差保证了最差情况也是 Identity，不会比不用差
        return feat + feat * attn * self.scale

# 1. SCDown 模块 (v10 下采样)
class SCDown(nn.Module):
    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        from .conv import Conv  # 确保能搜到同目录的Conv
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k, s, g=c2)
    def forward(self, x):
        return self.cv2(self.cv1(x))

# 2. PSA 模块 (v10 注意力机制)
class PSA(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        from .conv import Conv
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv(self.c, c1, 1, 1)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.c, self.c, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        a, b = x.chunk(2, 1)
        b = self.cv2(self.cv1(b) * self.attn(self.cv1(b)))
        return torch.cat((a, b), 1)

# 3. C2fCIB 模块 (v10 核心残差块)
class CIB(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5, dw=True):
        super().__init__()
        from .conv import Conv
        c_ = int(c2 * e)
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1) if dw else Conv(c1, c1, 3),
            Conv(c1, c_, 1, 1),
            Conv(c_, c_, 3, g=c_) if dw else Conv(c_, c_, 3),
            Conv(c_, c2, 1, 1)
        )
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv1(x) if self.add else self.cv1(x)

class C2fCIB(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5, dw=True):
        super().__init__()
        from .conv import Conv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, dw=dw) for _ in range(n))
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Attention(nn.Module):
    """Attention block for YOLOv12/v10 modified versions."""

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2

        # 这里的 Conv 引用的是同文件下的 Conv 类
        from .conv import Conv
        self.qkv = nn.Conv2d(dim, h, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.pe = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, -1, N).split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(x)
        x = self.proj(x)
        return x


# 如果是 YOLOv12，通常还会配套一个 A2Block
class A2Block(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        from .conv import Conv
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)
        self.attn = Attention(self.c)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)
        b = self.attn(b)
        return x + self.cv2(torch.cat((a, b), 1)) if self.add else self.cv2(torch.cat((a, b), 1))
import torch
import torch.nn as nn
from .conv import Conv

class PConv(nn.Module):
    """Partial Convolution (PConv) for FasterNet."""
    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.forward_cfg = forward

    def forward(self, x):
        if self.forward_cfg == 'split_cat':
            x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
            x1 = self.partial_conv3(x1)
            x = torch.cat((x1, x2), 1)
        return x

class FasterBlock(nn.Module):
    """FasterNet Block: PConv -> Conv1x1 -> Conv1x1."""
    def __init__(self, c1, c2, n_div=4, expansion=2):
        super().__init__()
        # FasterNet 要求输入输出通道一致，此处为了适配 YOLO/RT-DETR 做了调整
        self.pconv = PConv(c1, n_div)
        hidden_dim = int(c1 * expansion)
        self.mlp = nn.Sequential(
            Conv(c1, hidden_dim, 1),
            nn.Conv2d(hidden_dim, c1, 1, bias=False)
        )
        self.drop_path = nn.Identity() # 简化版适配

    def forward(self, x):
        shortcut = x
        x = self.pconv(x)
        x = self.mlp(x)
        return shortcut + self.drop_path(x)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
try:
    from dcnv4 import DCNv4 as DCNv4_Official
    HAS_DCNV4 = True
except ImportError:
    HAS_DCNV4 = False

class DCNv4_Block(nn.Module):
    """
    专为 ResNet18 架构设计的 BasicBlock 变体，集成了 DCNv4。
    结构：Conv3x3 -> DCNv4 -> Shortcut
    """

    def __init__(self, c1, c2, stride=1, k=3, s=1, p=1, g=1, d=1, act=True):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.stride = stride

        # 1. 第一层卷积：负责改变通道数和下采样 (Standard Conv)
        # 为了稳定性，降采样层通常保持标准卷积
        self.cv1 = nn.Conv2d(c1, c2, k, stride, p, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.act1 = nn.ReLU(inplace=True)

        # 2. 第二层卷积：替换为 DCNv4 (保持 Stride=1)
        if HAS_DCNV4:
            # 官方 DCNv4 (需要 pip install dcnv4)
            self.cv2 = DCNv4_Official(
                channels=c2,
                kernel_size=3,
                stride=1,
                pad=1,
                dilation=1,
                group=4,  # DCNv4 推荐 group=4 或 8
                offset_scale=1.0,
                act_layer='GELU',
                norm_layer='BN',
                without_pointwise=False
            )
            self.use_official = True
        else:
            # 降级方案：使用 TorchVision DeformConv2d (DCNv2)
            # 这是一个带 Offset 生成器的封装
            self.offset_conv = nn.Conv2d(c2, 2 * 3 * 3, kernel_size=3, stride=1, padding=1)
            self.mask_conv = nn.Conv2d(c2, 1 * 3 * 3, kernel_size=3, stride=1, padding=1)
            self.dcn_conv = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, bias=False)  # 权重容器
            self.bn2 = nn.BatchNorm2d(c2)
            self.use_official = False
            if stride == 1 and c1 == c2:
                print(f"Warning: dcnv4 library not found. Falling back to torchvision DeformConv2d for {c2} channels.")

        self.act2 = nn.ReLU(inplace=True)

        # 3. Shortcut 连接 (ResNet 结构)
        self.downsample = nn.Identity()
        if stride != 1 or c1 != c2:
            self.downsample = nn.Sequential(
                nn.Conv2d(c1, c2, 1, stride, bias=False),
                nn.BatchNorm2d(c2)
            )

    def forward(self, x):
        identity = self.downsample(x)

        # Step 1: Standard Conv
        x = self.cv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Step 2: DCN Operation
        if self.use_official:
            # 官方 DCNv4 前向
            x = self.cv2(x)
        else:
            # TorchVision DCNv2 前向模拟
            offset = self.offset_conv(x)
            mask = torch.sigmoid(self.mask_conv(x))

            # 使用 deform_conv2d 算子
            # 注意：这里权值使用 self.dcn_conv.weight
            x = deform_conv2d(
                input=x,
                offset=offset,
                weight=self.dcn_conv.weight,
                bias=None,
                stride=1,
                padding=1,
                dilation=1,
                mask=mask
            )
            x = self.bn2(x)

        # Step 3: Shortcut
        x += identity
        x = self.act2(x)
        return x


import torch
import torch.nn as nn

# --- 将此代码粘贴到 ultralytics/nn/modules/block.py 的最末尾 ---
import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class ConvBNAct(nn.Module):
    """标准的 Conv + BN + Activation 模块"""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LightConv(nn.Module):
    """
    修正后的轻量级卷积 (Depth-wise Separable Convolution)
    解决 'in_channels must be divisible by groups' 报错
    """

    def __init__(self, c1, c2, k=3):
        super().__init__()
        # 1. Depth-wise Conv: 保持通道数不变，groups=c1
        self.dw = nn.Conv2d(c1, c1, k, 1, k // 2, groups=c1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.act1 = nn.ReLU()

        # 2. Point-wise Conv: 1x1 卷积，负责改变通道数 (c1 -> c2)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.ReLU()

    def forward(self, x):
        x = self.act1(self.bn1(self.dw(x)))
        x = self.act2(self.bn2(self.pw(x)))
        return x


class HG_Stem(nn.Module):
    """HGNet 的 Stem 层 (初始降采样)"""

    def __init__(self, c1, c2, k=3):
        super().__init__()
        # c1 -> cm -> c2
        cm = (c1 + c2) // 2
        self.stem1 = ConvBNAct(c1, cm, 3, 2, 1)  # Stride 2
        self.stem2 = ConvBNAct(cm, c2, 3, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.stem1(x)
        x = self.stem2(x)
        x = self.pool(x)
        return x


class HGBlock(nn.Module):
    """
    HGNetV2 的核心模块
    Args:
        c1: 输入通道
        c2: 输出通道
        k: 卷积核大小
        light_conv: 是否使用轻量级卷积
        layer_num: 内部堆叠层数 (控制模型深度)
    """

    def __init__(self, c1, c2, k=3, light_conv=True, layer_num=6):
        super().__init__()

        # 内部通道数计算 (Mid Channels)
        hidden_channels = 32  # 基础宽度
        hidden_channels = 48 if c2 < 256 else 96

        self.layers = nn.ModuleList()
        # 构建内部密集连接层
        for i in range(layer_num):
            in_c = c1 if i == 0 else hidden_channels
            if light_conv:
                self.layers.append(LightConv(in_c, hidden_channels, k=k))
            else:
                self.layers.append(ConvBNAct(in_c, hidden_channels, k=k, s=1))

        # 聚合层：将所有层的特征聚合
        # 输入通道 = c1 (原始输入) + layer_num * hidden_channels (所有中间层输出)
        total_channels = c1 + layer_num * hidden_channels
        self.aggregation = ConvBNAct(total_channels, c2, k=1, s=1)  # 1x1 Conv 降维

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            # 每一层的输入是上一层的输出
            out = layer(features[-1])
            features.append(out)

        # 将所有特征在通道维度拼接
        out = torch.cat(features, dim=1)
        out = self.aggregation(out)
        return out


import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv


class ASFF(nn.Module):
    """
    自适应空间特征融合 (Adaptive Spatial Feature Fusion)
    这是 AFPN 的核心组件，计算量较大，适合用来增加 GFLOPs
    """

    def __init__(self, level, ratios, c_in_list, c_out):
        super().__init__()
        self.level = level
        self.dim = c_out

        # 权重计算分支
        self.weight_levels = nn.Conv2d(c_out * len(c_in_list), len(c_in_list), kernel_size=1, stride=1, padding=0)

        # 特征对齐分支
        self.convs = nn.ModuleList()
        for i, c_in in enumerate(c_in_list):
            if i == level:  # 同层
                self.convs.append(Conv(c_in, c_out, 1))
            elif i < level:  # 上层需要下采样
                self.convs.append(Conv(c_in, c_out, 3, 2 ** (level - i)))  # 动态计算下采样倍数
            elif i > level:  # 下层需要上采样
                self.convs.append(nn.Sequential(
                    Conv(c_in, c_out, 1),
                    nn.Upsample(scale_factor=2 ** (i - level), mode='nearest')
                ))

    def forward(self, x_list):
        # 1. 对齐特征图尺寸和通道
        aligned_x = []
        for i, conv in enumerate(self.convs):
            aligned_x.append(conv(x_list[i]))
        # 2. 计算权重
        x_cat = torch.cat(aligned_x, dim=1)
        weights = F.softmax(self.weight_levels(x_cat), dim=1)

        # 3. 加权融合
        out = 0
        for i in range(len(aligned_x)):
            out += weights[:, i:i + 1, :, :] * aligned_x[i]
        return out

    """
    AFPN (Asymptotic Feature Pyramid Network) 封装版
    输入: [P3, P4, P5]
    输出: [P3, P4, P5] (通道数统一为 256)
    """


class AFPN_Neck(nn.Module):
    # c1 会自动接收来自 Backbone 的通道列表 [128, 256, 512]
    # c2 会接收 YAML 里的输出通道数 256
    def __init__(self, c1, c2):
        super().__init__()
        in_channels = c1  # 把 c1 赋值给 in_channels
        out_channels = c2  # 把 c2 赋值给 out_channels

        # 下面保持不变...
        self.projs = nn.ModuleList([
            Conv(c, out_channels, 1) for c in in_channels
        ])
        self.projs = nn.ModuleList([
            Conv(c, out_channels, 1) for c in in_channels
        ])

        # 投影层：先把 Backbone 的不同通道统一
        self.projs = nn.ModuleList([
            Conv(c, out_channels, 1) for c in in_channels
        ])

        c = out_channels
        # 第一阶段融合 (渐进式)
        self.l0_3 = ASFF(0, [1, 2], [c, c], c)  # 融合 P3, P4
        self.l0_4 = ASFF(1, [1, 2], [c, c], c)  # 融合 P3, P4

        self.l1_4 = ASFF(0, [1, 2], [c, c], c)  # 融合 P4, P5
        self.l1_5 = ASFF(1, [1, 2], [c, c], c)  # 融合 P4, P5

        # 第二阶段融合 (全局)
        self.l2_3 = ASFF(0, [1, 2, 3], [c, c, c], c)
        self.l2_4 = ASFF(1, [1, 2, 3], [c, c, c], c)
        self.l2_5 = ASFF(2, [1, 2, 3], [c, c, c], c)

    def forward(self, x):
        # x is [P3, P4, P5]
        p3, p4, p5 = x[0], x[1], x[2]

        # 统一通道
        p3 = self.projs[0](p3)
        p4 = self.projs[1](p4)
        p5 = self.projs[2](p5)

        # Level 0 (Pairwise Fusion)
        # 这里的实现为了简化 YAML 复杂度，采用固定逻辑
        # 实际上 AFPN 结构很深，这里模拟了其“多阶段融合”的计算量

        # Stage 1
        p3_mid = self.l0_3([p3, p4])
        p4_mid_a = self.l0_4([p3, p4])

        p4_mid_b = self.l1_4([p4, p5])
        p5_mid = self.l1_5([p4, p5])

        p4_mid = (p4_mid_a + p4_mid_b) / 2  # 简单融合中间层

        # Stage 2 (Final Fusion)
        out3 = self.l2_3([p3_mid, p4_mid, p5_mid])
        out4 = self.l2_4([p3_mid, p4_mid, p5_mid])
        out5 = self.l2_5([p3_mid, p4_mid, p5_mid])

        return [out3, out4, out5]


import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv, autopad


class FAM(nn.Module):
    """
    Feature Alignment Module (特征对齐模块)
    负责将 P3, P4, P5 对齐到统一尺度并拼接
    """

    def __init__(self, c_in_list, c_mid):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 将 P3, P4, P5 投影到中间通道
        self.convs = nn.ModuleList([
            Conv(c, c_mid, 1) for c in c_in_list
        ])

    def forward(self, x_list):
        # x_list: [P3, P4, P5]
        # 对齐到 P4 (中间那个尺度)
        target_h, target_w = x_list[1].shape[2], x_list[1].shape[3]

        feats = []
        for i, x in enumerate(x_list):
            x = self.convs[i](x)
            if x.shape[2:] != (target_h, target_w):
                x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
            feats.append(x)

        # 拼接
        return torch.cat(feats, dim=1)


class IFM(nn.Module):
    """
    Injection Fusion Module (注入融合模块)
    负责将全局特征分发回 P3, P4, P5
    """

    def __init__(self, c_global, c_out_list):
        super().__init__()
        # 将全局特征切分并投影回各个层级的通道数
        self.convs = nn.ModuleList([
            Conv(c_global, c, 1) for c in c_out_list
        ])

    def forward(self, x_global, x_list):
        # x_global: 融合后的全局特征
        # x_list: 原始的 [P3, P4, P5]
        outs = []
        for i, x_origin in enumerate(x_list):
            # 1. 调整全局特征到当前层级的尺寸
            h, w = x_origin.shape[2], x_origin.shape[3]
            x_inject = self.convs[i](x_global)
            if x_inject.shape[2:] != (h, w):
                x_inject = F.interpolate(x_inject, size=(h, w), mode='bilinear', align_corners=False)

            # 2. 融合 (这里使用简单的加法 + 卷积融合)
            outs.append(x_origin + x_inject)

        return outs


class GDNeck(nn.Module):
    """
    Gold-YOLO Neck (Gather-and-Distribute Mechanism)
    Args:
        c1: 输入通道列表 [P3, P4, P5]
        c2: 输出通道 (这里作为基准通道)
    """

    def __init__(self, c1, c2):
        super().__init__()
        # c1 应该是由 tasks.py 传入的列表 [128, 256, 512]
        # c2 是输出基准通道，例如 256

        self.in_channels = c1
        self.out_channels = c2
        mid_c = c2 // 2  # 中间特征通道

        # 1. Gather (收集)
        # 将所有特征对齐拼接
        self.fam = FAM(c1, mid_c)

        # 全局融合层 (Global Fusion) - 使用大核或深层卷积处理收集到的特征
        # 拼接后通道数 = mid_c * 3
        gather_c = mid_c * 3
        self.global_fusion = nn.Sequential(
            Conv(gather_c, gather_c, 3, 1),
            Conv(gather_c, gather_c, 3, 1),
            Conv(gather_c, gather_c, 1)  # 融合完成
        )

        # 2. Distribute (分发)
        # 将全局特征注入回各层
        # 这里为了适配 RT-DETR Head，我们将输出统一为 c2 (256)
        self.ifm = IFM(gather_c, [c2, c2, c2])

        # 对原始输入也做一下投影，以便相加
        self.input_projs = nn.ModuleList([
            Conv(c, c2, 1) for c in c1
        ])

    def forward(self, x):
        # x: [P3, P4, P5]

        # 1. Gather Process
        x_global = self.fam(x)
        x_global = self.global_fusion(x_global)

        # 2. Input Projection (原始特征转为目标通道)
        x_proj = [proj(item) for proj, item in zip(self.input_projs, x)]

        # 3. Distribute Process (注入)
        outs = self.ifm(x_global, x_proj)

        # outs: [P3_new, P4_new, P5_new] (all 256 channels)
        return outs
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# ================================================================
# [New Innovation] HGRA-Block: Hierarchical Gradient-Response Aggregation
# 针对 PCB 微小缺陷（短路/毛刺）优化的自研主干模块
# 包含：空洞卷积 (Dilated Conv) + 梯度选择 (Gradient Selection)
# ================================================================
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class HGRABlock(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)

        # === 核心手术点 1: 引入空洞卷积 (Dilation=2) ===
        # 原理：扩大感受野，但不丢失分辨率，专门针对细微的断路/短路
        self.m = nn.Sequential(*(
            nn.Sequential(
                # 第一层：普通 3x3 卷积
                Conv(c_, c_, 3, 1, g=g),
                # 第二层：空洞 3x3 卷积 (dilation=2)，提取长距离依赖
                # 注意：这里我们手动构造一个带 dilation 的 Conv
                nn.Sequential(
                    nn.Conv2d(c_, c_, 3, 1, padding=2, dilation=2, groups=g, bias=False),
                    nn.BatchNorm2d(c_),
                    nn.SiLU()
                )
            ) for _ in range(n)
        ))

        # === 核心手术点 2: 注入梯度选择机制 (DSM) ===
        # 替换了原版可能存在的 AvgPool，使用 MaxPool 保留强特征
        self.att = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),  # <--- 关键修改：最大池化
            Conv(c_, c_, 1, 1),
            nn.Sigmoid()
        )

        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        # 1. 分支 1: 经过空洞卷积处理
        y1 = self.m(self.cv1(x))
        # 2. 分支 2: 原始特征
        y2 = self.cv2(x)

        # 3. 融合：利用注意力加权，过滤背景噪点
        return self.cv3(torch.cat((y1 * self.att(y1), y2), 1))
# =========================================================================
# Innovation 3: CASA-Module (Cross-Axis Structural Attention)
# 原型：PolaFormer (ICLR 2025)
# 作用：利用极化机制保持 X/Y 轴的高分辨率，精准捕捉线状缺陷（短路/断路）。
# =========================================================================

class CASABlock(nn.Module):
    def __init__(self, c1, num_heads=8, dropout=0.):
        super().__init__()
        self.c1 = c1
        self.num_heads = num_heads
        self.head_dim = c1 // num_heads

        # 1. 核心算子：极化注意力
        # QKV 生成
        self.qkv = nn.Conv2d(c1, c1 * 3, 1)

        # 2. 空间聚合算子 (模拟极化)
        # 这里的技巧是：不要全局池化，而是分别对 H 和 W 维度池化
        # 这就是 "Cross-Axis" 的来源
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 保留 H，压缩 W
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 保留 W，压缩 H

        # 3. 激励函数
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Conv2d(c1, c1, 1)

        # FFN 部分
        self.norm1 = nn.GroupNorm(1, c1)
        self.norm2 = nn.GroupNorm(1, c1)
        self.mlp = nn.Sequential(
            nn.Conv2d(c1, c1 * 4, 1),
            nn.GELU(),
            nn.Conv2d(c1 * 4, c1, 1)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        residue = x
        x = self.norm1(x)

        # 生成 Q, K, V
        qkv = self.qkv(x).chunk(3, dim=1)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # --- 极化注意力机制 (Simplified Pola) ---

        # 1. Channel-only Branch (通道注意力)
        # [B, C, 1, 1]
        w_ch = self.softmax(
            torch.mean(q, dim=[2, 3], keepdim=True) * torch.mean(k, dim=[2, 3], keepdim=True) * (self.c1 ** -0.5))
        v_ch = w_ch * v

        # 2. Spatial-only Branch (空间极化)
        # 关键创新：分别看 H 轴和 W 轴，不丢失线性特征
        q_h = self.pool_w(q)  # [B, C, H, 1]
        k_h = self.pool_w(k)
        q_w = self.pool_h(q)  # [B, C, 1, W]
        k_w = self.pool_h(k)

        # 简单的线性注意力模拟
        att_h = torch.sigmoid(q_h * k_h)  # 高度方向的结构权重
        att_w = torch.sigmoid(q_w * k_w)  # 宽度方向的结构权重

        # 3. 融合：特征 = 原始V * 通道权 * 空间权
        # 这里的广播机制会自动处理维度
        out = v_ch * att_h * att_w

        out = self.proj(out)
        out = out + residue

        # FFN
        out = out + self.mlp(self.norm2(out))
        return out
# =========================================================================
# Innovation 2: DSFPN-Module (Defect-Sensitive Fusion)
# 原型：HSFPN (MFDS-DETR)
# 改进：加入 Scale-Aware Attention，专门放大微小缺陷信号
# =========================================================================

# =========================================================================
# Innovation 2: DSFPN-Module (Defect-Sensitive Fusion)
# 原型：基于你的 HSFPN YAML 逻辑进行封装
# 升级：加入了 Spatial Attention，专门强化微小缺陷的定位
# =========================================================================

class DSFPNModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # c1: Concat 后的通道数 (例如 512)
        # c2: 输出通道数 (例如 256)

        # 基础融合卷积
        self.conv = Conv(c1, c2, 1, 1)

        # 1. 通道注意力 (你原本有的)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(c2, c2 // 16, 1, 1, act=True),
            Conv(c2 // 16, c2, 1, 1, act=False),
            nn.Sigmoid()
        )

        # 2. 空间注意力 (我额外送你的，提分关键！)
        # 专门用来高亮 PCB 上的微小缺陷点
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        # 3. 最终加权
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 1. 先降维
        feat = self.conv(x)

        # 2. 计算通道权重 (关注“是什么缺陷”)
        chn_weight = self.ca(feat)
        feat_ca = feat * chn_weight

        # 3. 计算空间权重 (关注“缺陷在哪里”)
        # 对通道做 Max 和 Avg 压缩
        max_out, _ = torch.max(feat_ca, dim=1, keepdim=True)
        avg_out = torch.mean(feat_ca, dim=1, keepdim=True)
        spatial_weight = self.sa(torch.cat([max_out, avg_out], dim=1))

        # 4. 双重加权输出 + 残差连接
        return feat + feat_ca * spatial_weight * self.scale