# -*- coding: utf-8 -*-
# Source from 
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# Modifications and relu override by 
# Felipe Torres Figueroa, felipe.torres@lis-lab.fr

# Torch Imports
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet

# In package imports
from models.transformer import CLS_Attention, Light_Shared

# Package imports
from einops import rearrange, repeat
from typing import Type, Any, Callable, Union, List, Optional
import pdb

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

# ========================================================================
dict_pretrained = {'resnet18': resnet.resnet18,
                   'resnet34': resnet.resnet34,
                   'resnet50': resnet.resnet50,
                   'resnet101': resnet.resnet101,
                   'resnet152': resnet.resnet152,
                   'resnext50_32x4d': resnet.resnext50_32x4d,
                   'resnext101_32x8d': resnet.resnext101_32x8d,
                   'wide_resnet50_2': resnet.wide_resnet50_2,
                   'wide_resnet101_2': resnet.wide_resnet101_2}

# ========================================================================
def conv3x3(in_planes: int, out_planes: int, stride: int = 1,
            groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)

# ========================================================================
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)

# ========================================================================
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None, groups: int = 1,
                 base_width: int = 64, dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None
                 ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and'\
                             'base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in"\
                                      " BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the 
        # input when stride != 1
        self.override = False
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
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

# ========================================================================
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 
    # convolution(self.conv2) while original implementation places the 
    # stride at the first 1x1 convolution(self.conv1) according to "Deep
    # residual learning for image recognition" 
    # https://arxiv.org/abs/1512.03385 
    # This variant is also known as ResNet V1.5 and improves accuracy
    # according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None, groups: int = 1,
                 base_width: int = 64, dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None
                 ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the
        # input when stride != 1
        self.override = False
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = self.relu(out)

        return out

# ========================================================================
class ResNet(nn.Module):

    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int], num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 groups: int = 1, width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 fixed_dim: bool = False, project: bool = False,
                 dropout: float = 0., heads: int = 1,
                 shared: bool = False
                 ) -> None:

        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        self.project = True
        self.num_classes = num_classes
        self.shared = shared

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None"
                             " or a 3-element tuple, got {}".format(\
                             replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, padding=3,
                               stride=2, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.fc = nn.Linear(512*block.expansion, num_classes)

        # Transformer stream parameters
        # Self Attention type
        self._increasing_layers(block, heads, dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual
        # block behaves like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]],
                    planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.groups,
                            self.base_width, previous_dilation,
                            norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_increasing(self, depth: int=512, dim_head: int=256,
                         heads: int=1, dropout: float=0.,
                         project: bool=True):
        if self.shared:
            layer = Light_Shared(dim_head=dim_head, heads=heads,
                                 project_out=self.project,
                                 dropout=dropout)
        else:
            layer = CLS_Attention(depth, heads=heads, dim_head=dim_head,
                                  project_out=project,
                                  dropout=dropout)

        return layer

    def _increasing_layers(self, block, heads, dropout):
        # CLS Token
        ones_cls = torch.ones((1, 1, 64))

        self.cls_token = nn.Parameter(torch.normal(mean=ones_cls,
                                                   std=ones_cls),
                                                   requires_grad=True)

        if not self.shared:            
            self.base = self._make_increasing(depth=block.expansion*64,
                                              dim_head=64, heads=heads,
                                              dropout=dropout)

        self.l1 = self._make_increasing(depth=block.expansion*128,
                                        dim_head=block.expansion*64,
                                        heads=heads,
                                        dropout=dropout)

        self.l2 = self._make_increasing(depth=block.expansion*256,
                                        dim_head=block.expansion*128,
                                        heads=heads,
                                        dropout=dropout)

        self.l3 = self._make_increasing(depth=block.expansion*512,
                                        dim_head=block.expansion*256,
                                        heads=heads,
                                        dropout=dropout)

        self.l4 = self._make_increasing(depth=block.expansion*512,
                                        dim_head=block.expansion*512,
                                        heads=heads,
                                        dropout=0)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        b, n, _, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        cls_tokens = cls_tokens.unsqueeze(1)
        
        x0 = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if not self.shared:
            attn, cls_tokens = self.base(x, cls_tokens)

        x = self.layer1(x)
        attn, cls_tokens = self.l1(x, cls_tokens)

        x = self.layer2(x)
        attn, cls_tokens = self.l2(x, cls_tokens)

        x = self.layer3(x)
        attn, cls_tokens = self.l3(x, cls_tokens)
    
        x = self.layer4(x)
        attn, cls_tokens = self.l4(x, cls_tokens)
        cls_tokens = torch.flatten(cls_tokens, 1)
        
        x = self.fc(cls_tokens)
        return x


    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(arch: str, block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int], pretrained: bool, progress: bool,
            **kwargs: Any) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = dict_pretrained[arch](pretrained=True).state_dict()
        state_keys = state_dict.keys()
        model_state = model.state_dict()
        for key in model_state:
            if key in state_keys:
                if model_state[key].shape == state_dict[key].shape:
                    model_state[key] = state_dict[key]
        model.load_state_dict(model_state)
    return model


def resnet18(pretrained: bool = False, progress: bool = True,
            **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to
        stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained,
                   progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True,
             **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to
        stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True,
             **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to
        stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True,
              **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to
        stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained,
                   progress, **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True,
              project: bool = False,
              **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to
        stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained,
                   progress, project,  **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True,
                    project: bool = False,
                    **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks"
    <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download
        to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, project,  **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True,
                     project: bool = False,
                     **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks"
    <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download
        to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, project, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True,
                    project: bool = False,
                    **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download
        to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, project, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True,
                     project: bool = False,
                     **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download
        to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, project, **kwargs)


# ========================================================================
dict_clsv1 ={'resnet18_clsv1': resnet18,
             'resnet34_clsv1': resnet34,
             'resnet50_clsv1': resnet50,
             'resnet101_clsv1': resnet101,
             'resnet152_clsv1': resnet152,
             'resnext50_32x4d_clsv1': resnet.resnext50_32x4d,
             'resnext101_32x8d_clsv1': resnext101_32x8d,
             'wide_resnet50_2_clsv1': wide_resnet50_2,
             'wide_resnet101_2_clsv1': wide_resnet101_2}
