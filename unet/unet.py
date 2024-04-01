"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F
# from simsiam.builder import DepthwiseBottleneck


class UNet(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        zero_init_residual: bool = False
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image
        # print('input size:', output.size())
        # print('Encoder: ')
        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            # print(output.size())
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)
        # print(output.size())

        # print('Decoder:')
        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)
        # print(output.size())

        return output


class UNetEncoder(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            num_classes: int = 2048,
            chans: int = 32,
            num_pool_layers: int = 4,
            drop_prob: float = 0.0,
            zero_init_residual: bool = False
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.num_classes = num_classes
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)
        self.output_chans = ch * 2
        # original
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(ch * 2, num_classes)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, image: torch.Tensor):
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image
        # print('input size:', output.size())
        # print('Encoder: ')
        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            # print(output.size())
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        stack.append(output)
        # print(output.size())


        # original
        # output = self.avgpool(output)
        # output = torch.flatten(output, 1)
        # output = self.fc(output)
        # print('output size:', output.size())
    
        # return output


        return stack  # v1: output all hidden features


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_chans),
            nn.GroupNorm(out_chans // 16, out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_chans),
            nn.GroupNorm(out_chans // 16, out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            # nn.BatchNorm2d(out_chans),
            nn.GroupNorm(out_chans // 16, out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)


class UNetforSimMIM(UNet):
    def __init__(self,
                 in_chans: int,
                 out_chans: int,
                 chans: int = 32,
                 num_pool_layers: int = 4,
                 drop_prob: float = 0.0,
                 zero_init_residual: bool = False
                 ):
        super(UNetforSimMIM, self).__init__(in_chans, out_chans, chans, num_pool_layers, drop_prob, zero_init_residual)
        self.proj = nn.Conv2d(in_chans, chans, kernel_size=1)
        self.mask_token = nn.Parameter(torch.zeros(1, chans, 1, 1))
        self.down_sample_layers[0] = ConvBlock(chans, chans, drop_prob)

    def forward(self, image: torch.Tensor, mask=None) -> torch.Tensor:
        image = self.proj(image)
        if mask is not None:
            B, C, H, W = image.shape
            mask_tokens = self.mask_token.expand(B, -1, H, W)
            w = mask.unsqueeze(1).type_as(mask_tokens)

            image = image * (1. - w) + mask_tokens * w

        output = super(UNetforSimMIM, self).forward(image)
        return output

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mask_token'}


class UNetV2(UNet):
    def __init__(self,
                 in_chans: int,
                 out_chans: int,
                 chans: int = 32,
                 num_pool_layers: int = 4,
                 drop_prob: float = 0.0,
                 zero_init_residual: bool = False
                 ):
        super(UNetV2, self).__init__(in_chans, out_chans, chans, num_pool_layers, drop_prob, zero_init_residual)
        self.proj = nn.Conv2d(in_chans, chans, kernel_size=1)
        self.down_sample_layers[0] = ConvBlock(chans, chans, drop_prob)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self.proj(image)

        output = super(UNetV2, self).forward(image)
        return output


class JointUNetMIM(UNet):
    def __init__(self,
                 in_chans: int,
                 out_chans: int,
                 chans: int = 32,
                 num_pool_layers: int = 4,
                 drop_prob: float = 0.0,
                 zero_init_residual: bool = False
                 ):
        super(JointUNetMIM, self).__init__(in_chans, out_chans, chans, num_pool_layers, drop_prob, zero_init_residual)
        self.proj = nn.Conv2d(in_chans, chans, kernel_size=1)
        self.mask_token = nn.Parameter(torch.zeros(1, chans, 1, 1))
        self.down_sample_layers[0] = ConvBlock(chans, chans, drop_prob)
        # v1
        self.up_conv[-1] = nn.Identity()
        self.main_block = nn.Sequential(
                ConvBlock(chans * 2, chans, drop_prob),
                nn.Conv2d(chans, self.out_chans, kernel_size=1, stride=1),
            )
        self.pred_block = nn.Sequential(
                ConvBlock(chans * 2, chans, drop_prob),
                nn.Conv2d(chans, self.out_chans, kernel_size=1, stride=1),
            )

        # v2
        # self.up_conv[-1] = ConvBlock(chans * 2, chans, drop_prob)
        # self.main_block = nn.Conv2d(chans, self.out_chans, kernel_size=1, stride=1)
        # self.pred_block = nn.Conv2d(chans, self.out_chans, kernel_size=1, stride=1)

    def forward(self, image: torch.Tensor, mask=None, only_recon=False, feature_detach=False, n2s=False):
        image = self.proj(image)
        B, C, H, W = image.shape

        if mask is not None:
            if n2s:
                B //= 2
                image, interpolate_image = torch.split(image, [B, B], dim=0)
            mask_tokens = self.mask_token.expand(B, -1, H, W)
            w = mask.unsqueeze(1).type_as(mask_tokens)
            masked_image = image * (1. - w) + mask_tokens * w
            if only_recon:
                inputs = masked_image
            elif n2s:
                inputs = torch.cat([interpolate_image, masked_image], dim=0)
            else:
                inputs = torch.cat([image, masked_image], dim=0)

        else:
            inputs = image

        features = super(JointUNetMIM, self).forward(inputs)

        if mask is not None:
            if only_recon:
                output = self.pred_block(features)
                return output
            feature_main, feature_pred = torch.split(features, [B, B], dim=0)
            if feature_detach:
                feature_main = feature_main.detach()
            output_main = self.main_block(feature_main)
            output_pred = self.pred_block(feature_pred)

            return output_main, output_pred

        else:
            output = self.main_block(features)
            return output

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mask_token'}


