# -*- coding: utf-8 -*-
# Source from:
# https://pytorch.org/vision/main/_modules/torchvision/models/convnext.html 
# Modifications and simplification by:
# Felipe Torres Figueroa, felipe.torres@lis-lab.fr

# Torch Imports
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchvision.models import convnext
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.ops.misc import Conv2dNormActivation, Permute
from torchvision.models._utils import _ovewrite_named_param

# In-package imports
from models.transformer import CLS_Attention

# Package-Imports
from functools import partial
from einops import repeat
from typing import Any, Callable, List, Optional, Sequence


# ========================================================================
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight,
                         self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class CNBlock(nn.Module):
    def __init__(self, dim, layer_scale: float,
                 stochastic_depth_prob: float,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            norm_layer(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


class CNBlockConfig:
    # Stores information listed at Section 3 of the ConvNeXt paper
    def __init__(self, input_channels: int, out_channels: Optional[int],
                 num_layers: int, ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)


class ConvNeXt(nn.Module):
    def __init__(self, block_setting: List[CNBlockConfig],
                 stochastic_depth_prob: float = 0.0,
                 layer_scale: float = 1e-6, num_classes: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 **kwargs: Any, ) -> None:
        super().__init__()

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and \
                  all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)
        
        layers: List[nn.Module] = []
        ca_blocks: List[nn.Module] = []

        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(3, firstconv_output_channels,
                                 kernel_size=4, stride=4, padding=0,
                                 norm_layer=norm_layer,
                                 activation_layer=None, bias=True,)
        )
        # CLS-Token
        ones_cls = torch.ones((1, 1, firstconv_output_channels))
        self.cls_token = nn.Parameter(torch.normal(mean=ones_cls,
                                                   std=ones_cls),
                                                   requires_grad=True)
        # First Block
        ca_blocks.append(CLS_Attention(block_setting[0].out_channels,
                                       heads=1, dim_head=firstconv_output_channels,
                                       project_out=True))

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        self.ca_placement = []
        self.ca_placement.append(0)
        placement = 0
        for idx, cnf in enumerate(block_setting):
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            placement+=1
            if cnf.out_channels is not None:
                # Downsampling
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                    )
                )
                placement+=1
            if idx+1 < len(block_setting):
                cfg = block_setting[idx+1]
                try:
                    ca_blocks.append(CLS_Attention(cfg.out_channels, heads=1,
                                                   dim_head=cnf.out_channels,
                                                   project_out=True))
                    self.ca_placement.append(placement)
                except TypeError:
                    pass

        self.features = nn.Sequential(*layers)
        lastblock = block_setting[-1]
        lastconv_output_channels = (
            lastblock.out_channels if lastblock.out_channels is not None else lastblock.input_channels
        )
        self.ca_placement.append(placement)

        # Before the classifier
        ca_blocks.append(CLS_Attention(lastconv_output_channels, heads=1,
                                       dim_head=lastconv_output_channels,
                                       project_out=True))

        self.cls_stream = nn.Sequential(*ca_blocks)
        self.classifier = nn.Sequential(
            norm_layer(lastconv_output_channels), nn.Flatten(1),
                       nn.Linear(lastconv_output_channels, num_classes)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)
        counter = 0
        for idf, feature in enumerate(self.features):
            x = feature(x)
            if idf == self.ca_placement[counter]:
                attn, cls_tokens = self.cls_stream[counter](x, cls_tokens)
                counter = counter+1
        cls_tokens = cls_tokens.view(b, cls_tokens.shape[1], 1, 1)
        x = self.classifier(cls_tokens)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _convnext(block_setting: List[CNBlockConfig], stochastic_depth_prob: float,
              weights: None, progress: bool, **kwargs: Any, ) -> ConvNeXt:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ConvNeXt(block_setting, stochastic_depth_prob=stochastic_depth_prob, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def convnext_tiny(*, weights: Optional[convnext.ConvNeXt_Tiny_Weights]=None,
                  progress: bool=True, **kwargs: Any) -> ConvNeXt:
    """ConvNeXt Tiny model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        weights (:class:`~torchvision.models.convnext.ConvNeXt_Tiny_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Tiny_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ConvNeXt_Tiny_Weights
        :members:
    """
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)


def convnext_small(*, weights: Optional[convnext.ConvNeXt_Small_Weights]=None,
                   progress: bool = True, **kwargs: Any) -> ConvNeXt:
    """ConvNeXt Small model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        weights (:class:`~torchvision.models.convnext.ConvNeXt_Small_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Small_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ConvNeXt_Small_Weights
        :members:
    """
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 27),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.4)
    return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)

def convnext_base(*, weights: Optional[convnext.ConvNeXt_Base_Weights]=None,
                  progress: bool = True, **kwargs: Any) -> ConvNeXt:
    """ConvNeXt Base model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        weights (:class:`~torchvision.models.convnext.ConvNeXt_Base_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Base_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ConvNeXt_Base_Weights
        :members:
    """
    block_setting = [
        CNBlockConfig(128, 256, 3),
        CNBlockConfig(256, 512, 3),
        CNBlockConfig(512, 1024, 27),
        CNBlockConfig(1024, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)

def convnext_large(*,
                   weights: Optional[convnext.ConvNeXt_Large_Weights]=None,
                   progress: bool=True, **kwargs: Any)-> ConvNeXt:
    """ConvNeXt Large model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        weights (:class:`~torchvision.models.convnext.ConvNeXt_Large_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Large_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ConvNeXt_Large_Weights
        :members:
    """
    block_setting = [
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 3),
        CNBlockConfig(768, 1536, 27),
        CNBlockConfig(1536, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)

# ConvNext models, off the shelf
dict_convnext_stream={'convnext_tiny_stream': convnext_tiny,
                      'convnext_small_stream': convnext_small,
                      'convnext_base_stream': convnext_base,
                      'convnext_large_stream': convnext_large}
