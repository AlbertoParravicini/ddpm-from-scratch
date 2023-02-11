from typing import Sequence

import torch
import torch.nn as nn
from einops import rearrange
from torchtyping import TensorType

from ddpm_from_scratch.models.unet_simple_with_timestep import TimestepEmbedding
from ddpm_from_scratch.utils import C1, C2, H1, H2, W1, W2, B, C, H, W, expand_to_dims


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        up: bool = False,
        down: bool = False,
    ) -> None:
        super().__init__()
        # Following the following implementation of ResBlock, we apply GroupNorm followed by non-linearity,
        # and finally a 2D convolution. This sequence is referred to as `ConvBlock`.
        # Link: https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/unet.py#L182
        self.convblock_1 = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        )
        # Same as the first ConvBlock. In this case, the number of layers is not changed.
        self.convblock_2 = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        )
        self.timestep_embedding = TimestepEmbedding(out_channels)
        # Specify if this ResBlock will leave the spatial resolution unchanged,
        # or if we downscale/upscale the resolution.
        # Downscaling is done with a 3x3 convolution, since it empirically works better than `nn.MaxPool2d(2)`.
        # Upscaling is done with bilinear upsampling, since transposed convolutions give checkerboard artifacts
        # and it's not trivial to ensure that we obtain the same resolution as the corresponding downscaling blocks,
        # to concatenate values in the UNet.
        # We also define the skip transform, which combines a 1x1 Conv2d to map C1 to C2
        # with a spatial transformation that maps H1xW1 to H2xW2 and is identical to `self.rescale`.
        self.rescale: nn.Module
        self.skip_transform_spatial: nn.Module
        if up:
            self.rescale = nn.UpsamplingBilinear2d(scale_factor=2)
            self.skip_transform_spatial = nn.UpsamplingBilinear2d(scale_factor=2)
        elif down:
            self.rescale = nn.Conv2d(
                in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=2
            )
            self.skip_transform_spatial = nn.Conv2d(
                in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=2
            )
        else:
            self.rescale = nn.Identity()
            self.skip_transform_spatial = nn.Identity()
        self.skip_transform = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=1),
            self.skip_transform_spatial,
        )

    def forward(
        self, t: TensorType["B", "int"], x: TensorType["B", "C1", "H1", "W1", "float"]
    ) -> TensorType["B", "C2", "H2", "W2", "float"]:
        h = self.convblock_1(x)  # First ConvBlock, from C1 to C2.
        t = expand_to_dims(self.timestep_embedding(t), x)  # Replicate time embedding to H1 and W1.
        h = h + t  # Add timestep embedding.
        h = self.rescale(h)  # From H1xW1 to H2xW2.
        h = self.convblock_2(h)  # An additional ConvBlock.
        x = h + self.skip_transform(x)  # The skip connection maps C1xH1xW1 to C2xH2xW2.
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.attention = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)

    def forward(
        self, t: TensorType["B", "int"], x: TensorType["B", "C", "H", "W", "float"]
    ) -> TensorType["B", "C", "H", "W", "float"]:
        # Group together the spatial dimensions, and perform self attention by aggregating channels
        h, w = x.shape[-2:]
        x = self.norm(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        # Compute self-attention, using the same tensor for key, query, value.
        # Discard the attention matrix, using `need_weights=False` and taking the first output.
        x = self.attention(x, x, x, need_weights=False)[0]
        # Restore the original spatial structure
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
        channel_multipliers: Sequence[int] = (1, 2, 3),
    ) -> None:
        """
        A UNet model with time-conditioning.
        It has a few downsample 2D 3x3 convolutional layers with stride 2, followed by
        upsample 2D transposed 3x3 convolution layers that take as input both the previous layer's output
        and the output of the corresponding downsample layer.

        :param in_channels: input channels of the first layer. 1 for grayscale images, 3 for RGB images
        :param hidden_channels: base number of channels of hidden layers. The number of channels in layer `i`
            is obtained as `hidden_channels * channel_multipliers[i]`.
        :param channel_multipliers: a sequence of integers that specifies the number of down/upsample layers.
            The number of channels in layer `i` is given by `hidden_channels * channel_multipliers[i]`.
        """
        super().__init__()
        self._channels = [hidden_channels * c for c in channel_multipliers]
        # Initial layer, to project the number of dimensions to the number of hidden channels,
        self.initial_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=hidden_channels * channel_multipliers[0], kernel_size=3, padding=1
        )
        # Downsample layers. At each layer, we halve the resolution,
        # and increase the channel count by the specified factor.
        self.downsample_layers = nn.ModuleList(
            ResBlock(in_channels=self._channels[i], out_channels=self._channels[i + 1], down=True)
            for i in range(len(self._channels) - 1)
        )
        # Middle layer. Self-attention.
        self.middle_layers = nn.ModuleList(
            [
                ResBlock(in_channels=self._channels[-1], out_channels=self._channels[-1]),
                MultiheadAttention(self._channels[-1]),
                ResBlock(in_channels=self._channels[-1], out_channels=self._channels[-1]),
            ]
        )
        # Upsample layers. At each layer, we double the resolution
        # and decrease the channel count by the specified factor.
        # The number of input channels is twice as the ones in the downsample layers,
        # since upsample layers have skip connections that take as input both the output of the previous layer,
        # and the output of the downsample layer at the same index.
        self.upsample_layers = nn.ModuleList(
            ResBlock(in_channels=self._channels[i + 1] * 2, out_channels=self._channels[i], up=True)
            for i in range(len(self._channels) - 1)[::-1]
        )
        # Final layer, return the original amount of channels.
        self.final_conv = nn.Conv2d(
            in_channels=hidden_channels * channel_multipliers[0], out_channels=in_channels, kernel_size=3, padding=1
        )

    def forward(
        self, t: TensorType["B", "int"], x: TensorType["B", "C", "H", "W", "float"]
    ) -> TensorType["B", "C", "H", "W", "float"]:
        # Store the output of each layer.
        xs = []
        # Initial convolution layer.
        x = self.initial_conv(x)
        # Downsample layers.
        for layer in self.downsample_layers:
            # Compute each downsample layer.
            x = layer(t, x)
            xs.append(x)
        # Middle layers.
        for layer in self.middle_layers:
            x = layer(t, x)
        # Upsample layers.
        for layer in self.upsample_layers:
            # Concatenate each input with the output of the corresponding downsample layer, on the channel dimension.
            y = xs.pop()
            x = torch.cat([x, y], dim=1)
            x = layer(t, x)
        # Final convolution layer.
        x = self.final_conv(x)
        return x
