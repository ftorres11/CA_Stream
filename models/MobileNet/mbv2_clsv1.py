# -*- coding: utf-8 -*-
# Source from
# https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
# Modifications and simplification by:
# Felipe Torres Figueroa, felipe.torres@lis-lab.fr

# Torch Imports
import torch
from torch import nn, Tensor

from torchvision.models import mobilenetv2
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.models._utils import (_make_divisible,
                                       _ovewrite_named_param,
                                       handle_legacy_interface)

# In Package imports
from models.transformer import CLS_Attention

# Package Imports
from functools import partial
from typing import Any, Callable, List, Optional
from einops import rearrange, repeat
import pdb

# necessary for backwards compatibility
class InvertedResidual(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
            dropout (float): The droupout probability

        """
        super().__init__()
        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )


        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        # CLS-token
        ones_cls = torch.ones((1, 1, input_channel))
        self.cls_token = nn.Parameter(torch.normal(mean=ones_cls,
                                                   std=ones_cls),
                                                   requires_grad=True)

        features: List[nn.Module] = [
            Conv2dNormActivation(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)
        ]
        ca_blocks: List[nn.Module] = []
        ca_blocks.append(CLS_Attention(\
                         inverted_residual_setting[0][1], heads=1,
                         dim_head=input_channel, project_out=True))
        
        self.ca_placement=[]
        self.ca_placement.append(0)
        placements = 0
        # building inverted residual blocks
        for idf, cnf in enumerate(inverted_residual_setting):
            t, c, n, s = cnf 
            output_channel = _make_divisible(c * width_mult, round_nearest)
            placements+=n
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
            self.ca_placement.append(placements)
            if idf+1 < len(inverted_residual_setting):
                ca_blocks.append(CLS_Attention(\
                                 inverted_residual_setting[idf+1][1], heads=1,
                                 dim_head=output_channel, project_out=True))

        # building last several layers
        features.append(
            Conv2dNormActivation(
                input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6
            )
        )
        placements += 1
        self.ca_placement.append(placements)
        # Last Cross Attention Layers
        ca_blocks.append(CLS_Attention(self.last_channel, heads=1,
                         dim_head=input_channel, project_out=True))
        # Before GAP
        ca_blocks.append(CLS_Attention(self.last_channel, heads=1,
                         dim_head=self.last_channel, project_out=True))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        
        # CLS-Stream already initialized
        self.cls_stream = nn.Sequential(*ca_blocks)


    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        b, _, _, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)
        counter = 0

        # Modified forward to account to counter intuitive wrapping of layers
        for idf, feature in enumerate(self.features):
            x = feature(x)
            if idf == self.ca_placement[counter]:
                attn, cls_tokens = self.cls_stream[counter](x, cls_tokens)
                counter = counter+1

        # Cannot use "squeeze" as batch-size can be 1
        cls_tokens  = torch.flatten(cls_tokens, 1)
        cls_tokens = self.classifier(cls_tokens)
        return cls_tokens

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def mobilenet_v2_clsv1(
    *, weights: Optional[mobilenetv2.MobileNet_V2_Weights] = None,
    progress: bool = True, **kwargs: Any) -> MobileNetV2:
    """MobileNetV2 architecture from the `MobileNetV2: Inverted Residuals and Linear
    Bottlenecks <https://arxiv.org/abs/1801.04381>`_ paper.

    Args:
        weights (:class:`~torchvision.models.MobileNet_V2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.MobileNet_V2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.mobilenetv2.MobileNetV2``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.MobileNet_V2_Weights
        :members:
    """
    model = MobileNetV2(**kwargs)

    return model
