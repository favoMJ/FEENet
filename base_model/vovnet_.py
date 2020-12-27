""" Adapted from the original implementation. """
#FPS: 135.62
import collections
import dataclasses
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclasses.dataclass
class VoVNetParams:
    stem_out: int
    stage_conv_ch: List[int]  # Channel depth of
    stage_out_ch: List[int]  # The channel depth of the concatenated output
    layer_per_block: int
    block_per_stage: List[int]
    dw: bool

'''

VoVNet19_slim_eSE = {
    'stem': [64, 64, 128],
    'stage_conv_ch': [64, 80, 96, 112],
    'stage_out_ch': [112, 256, 384, 512],
    'layer_per_block': 3,
    'block_per_stage': [1, 1, 1, 1],
    'eSE' : True,
    "dw" : False
}
'''
_STAGE_SPECS = {
    "vovnet-19-slim-dw": VoVNetParams(
        64, [64, 80, 96, 112], [112, 256, 384, 512], 3, [1, 1, 1, 1], True
    ),
    "vovnet-19-dw": VoVNetParams(
        64, [128, 160, 192, 224], [256, 512, 768, 1024], 3, [1, 1, 1, 1], True
    ),
    "vovnet-19-slim": VoVNetParams(
        64, [64, 80, 96, 112], [112, 256, 384, 512], 3, [1, 1, 1, 1], False
    ),
    "vovnet-19": VoVNetParams(
        128, [128, 160, 192, 224], [256, 512, 768, 1024], 3, [1, 1, 1, 1], False
    ),
    "vovnet-39": VoVNetParams(
        128, [128, 160, 192, 224], [256, 512, 768, 1024], 5, [1, 1, 2, 2], False
    ),
    "vovnet-57": VoVNetParams(
        128, [128, 160, 192, 224], [256, 512, 768, 1024], 5, [1, 1, 4, 3], False
    ),
    "vovnet-99": VoVNetParams(
        128, [128, 160, 192, 224], [256, 512, 768, 1024], 5, [1, 3, 9, 3], False
    ),
}

_BN_MOMENTUM = 1e-1
_BN_EPS = 1e-5


def dw_conv(
        in_channels: int, out_channels: int, stride: int = 1
) -> List[torch.nn.Module]:
    """ Depthwise separable pointwise linear convolution. """
    return [
        torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            groups=in_channels,
            bias=False,
        ),
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
        torch.nn.BatchNorm2d(out_channels, eps=_BN_EPS, momentum=_BN_MOMENTUM),
        torch.nn.ReLU(inplace=True),
    ]


def conv(
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        kernel_size: int = 3,
        padding: int = 1,
) -> List[torch.nn.Module]:
    """ 3x3 convolution with padding."""
    return [
        torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
        torch.nn.BatchNorm2d(out_channels, eps=_BN_EPS, momentum=_BN_MOMENTUM),
        torch.nn.ReLU(inplace=True),
    ]


def pointwise(in_channels: int, out_channels: int) -> List[torch.nn.Module]:
    """ Pointwise convolution."""
    return [
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
        torch.nn.BatchNorm2d(out_channels, eps=_BN_EPS, momentum=_BN_MOMENTUM),
        torch.nn.ReLU(inplace=True),
    ]


# As seen here: https://arxiv.org/pdf/1910.03151v4.pdf. Can outperform ESE with far fewer
# paramters.
class ESA(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)
        # BCHW -> BHCW
        y = y.permute(0, 2, 1, 3)
        y = self.conv(y)

        # Change the dimensions back to BCHW
        y = y.permute(0, 2, 1, 3)
        y = torch.sigmoid_(y)
        return x * y.expand_as(x)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups  # groups是分的组数

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class _OSA(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            stage_channels: int,
            concat_channels: int,
            layer_per_block: int,
            use_depthwise: bool = False,
    ) -> None:

        super().__init__()

        aggregated = in_channels + layer_per_block * stage_channels
        self.isReduced = in_channels != stage_channels

        self.identity = in_channels == concat_channels

        self.layers = torch.nn.ModuleList()
        self.use_depthwise = use_depthwise


        self.conv0 = torch.nn.Sequential(*conv(in_channels, stage_channels))
        in_channels = stage_channels
        self.conv1 = nn.Conv2d(in_channels, stage_channels,kernel_size=1,stride=1,padding=0)
        self.conv2 = nn.Conv2d(in_channels, stage_channels,kernel_size=1,stride=1,padding=0)
        # feature aggregation

        self.conv11 = torch.nn.Sequential(*conv(in_channels, stage_channels,groups=stage_channels))
        self.conv22 = torch.nn.Sequential(*conv(in_channels, stage_channels,groups=stage_channels))

        self.concat = torch.nn.Sequential(*pointwise(aggregated, concat_channels))
        self.esa = ESA(concat_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.identity:
            identity_feat = x

        output = [x]
        x0 = self.conv0(x)

        x1 = self.conv1(x0)
        x2 = self.conv2(x1)

        output.append(x0)
        t = x0 + x1
        x1 = self.conv11(t)
        output.append(x1)

        t = x1 + x2
        x2 = self.conv22(t)
        output.append(x2)

        x = torch.cat(output, dim=1)
        x = channel_shuffle(x)
        xt = self.concat(x)
        xt = self.esa(xt)
        if self.identity:
            xt += identity_feat
        return xt


class _Maxpool(torch.nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        stage_channel = in_channels//2
        self.conv = torch.nn.Sequential(*conv(in_channels, stage_channel,kernel_size=1,padding=0))#试一下3*3 和 1*1
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(stage_channel,stage_channel,kernel_size=3,stride=2,padding=1)

    def forward(self,x):
        x = self.conv(x)
        x1 = self.maxpool(x)
        x2 = self.conv3(x)
        return torch.cat([x1,x2],dim=1)

class _OSA_stage(torch.nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            stage_channels: int,
            concat_channels: int,
            block_per_stage: int,
            layer_per_block: int,
            stage_num: int,
            use_depthwise: bool = False,
    ) -> None:

        super().__init__()

        self.add_module('maxpool',_Maxpool(in_channels))

        for idx in range(block_per_stage):
            # Add the OSA modules. If this is the first block in the stage, use the
            # proper in in channels, but the rest of the rest of the OSA layers will use
            # the concatenation channel depth outputted from the previous layer.
            self.add_module(
                f"OSA{stage_num}_{idx + 1}",
                _OSA(
                    in_channels if idx == 0 else concat_channels,
                    stage_channels,
                    concat_channels,
                    layer_per_block,
                    use_depthwise=use_depthwise,
                ),
            )



class VoVNet(torch.nn.Sequential):
    def __init__(
            self, model_name: str, num_classes: int = 10, input_channels: int = 3
    ) -> None:

        super().__init__()
        assert model_name in _STAGE_SPECS, f"{model_name} not supported."

        stem_ch = _STAGE_SPECS[model_name].stem_out
        config_stage_ch = _STAGE_SPECS[model_name].stage_conv_ch
        config_concat_ch = _STAGE_SPECS[model_name].stage_out_ch
        block_per_stage = _STAGE_SPECS[model_name].block_per_stage
        layer_per_block = _STAGE_SPECS[model_name].layer_per_block
        conv_type = dw_conv if _STAGE_SPECS[model_name].dw else conv

        # Construct the stem.
        stem = conv(input_channels, 32, stride=2)
        stem += conv_type(32, 32)
        stem += conv_type(32, stem_ch)
        '''
        "vovnet-19-slim": VoVNetParams(
        128, [64, 80, 96, 112], [112, 256, 384, 512], 3, [1, 1, 1, 1], False),
        '''
        # The original implementation uses a stride=2 on the conv below, but in this
        # implementation we'll just pool at every OSA stage, unlike the original
        # which doesn't pool at the first OSA stage.
        # stem += conv_type(64, stem_ch)

        self.model = torch.nn.Sequential()
        self.osa_0 = torch.nn.Sequential()
        self.osa_1 = torch.nn.Sequential()
        self.osa_2 = torch.nn.Sequential()
        self.osa_3 = torch.nn.Sequential()
        self.model.add_module("stem", torch.nn.Sequential(*stem))
        self._out_feature_channels = [stem_ch]

        # Organize the outputs of each OSA stage. This is the concatentated channel
        # depth of each sub block's layer's outputs.
        in_ch_list = [stem_ch] + config_concat_ch[:-1]

        # Add the OSA modules. Typically 4 modules.
        self.osa_0.add_module(f"OSA_{(0 + 2)}",_OSA_stage(
                    in_ch_list[0],
                    config_stage_ch[0],
                    config_concat_ch[0],
                    block_per_stage[0],
                    layer_per_block,
                    0 + 2,
                    _STAGE_SPECS[model_name].dw,
                ),
        )

        self._out_feature_channels.append(config_concat_ch[0])
        self.osa_1.add_module(f"OSA_{(1 + 2)}",_OSA_stage(
                    in_ch_list[1],
                    config_stage_ch[1],
                    config_concat_ch[1],
                    block_per_stage[1],
                    layer_per_block,
                    1 + 2,
                    _STAGE_SPECS[model_name].dw,
                ),
        )
        self._out_feature_channels.append(config_concat_ch[1])
        self.osa_2.add_module(f"OSA_{(2 + 2)}",_OSA_stage(
                    in_ch_list[2],
                    config_stage_ch[2],
                    config_concat_ch[2],
                    block_per_stage[2],
                    layer_per_block,
                    1 + 2,
                    _STAGE_SPECS[model_name].dw,
                ),
        )
        self._out_feature_channels.append(config_concat_ch[2])
        self.osa_3.add_module(f"OSA_{(3 + 2)}",_OSA_stage(
                    in_ch_list[3],
                    config_stage_ch[3],
                    config_concat_ch[3],
                    block_per_stage[3],
                    layer_per_block,
                    1 + 2,
                    _STAGE_SPECS[model_name].dw,
                ),
        )
        self._out_feature_channels.append(config_concat_ch[3])
        from base_model.blurpool import BlurPool

        self.bulr0 = BlurPool(64, filt_size=3, stride=1)
        self.bulr1 = BlurPool(112, filt_size=3, stride=1)
        self.bulr2 = BlurPool(256, filt_size=2, stride=1)
        self.bulr3 = BlurPool(384, filt_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        x = self.model(x)
        # print(x.shape)
        x = x + self.bulr0(x)

        x = self.osa_0( x )
        res.append(x)

        x = x + self.bulr1(x)
        x = self.osa_1(x)
        res.append(x)

        x = x + self.bulr2(x)
        x = self.osa_2(x)
        res.append(x)

        x = x + self.bulr3(x)
        x = self.osa_3(x)
        res.append(x)
        return res
