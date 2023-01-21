from typing import Sequence

import torch
import torch.nn as nn
from torchtyping import TensorType

from ddpm_from_scratch.utils import B, C, H, W


class UNetSimple(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
        channel_multipliers: Sequence[int] = (1, 2, 3),
    ) -> None:
        """
        A simple UNet model, without time-conditioning.
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
        self._channels = [in_channels] + [hidden_channels * c for c in channel_multipliers]
        # Downsample layers. At each layer, we halve the resolution,
        # and increase the channel count by the specified factor.
        self.downsample_layers = nn.ModuleDict(
            {
                f"down_{i}": nn.Conv2d(
                    in_channels=self._channels[i],
                    out_channels=self._channels[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=0
                    if self._channels[i] % 2 == 0
                    else 1,  # We need compatible dimensions with upsample layers.
                )
                for i in range(len(self._channels) - 1)
            }
        )
        # Upsample layers. At each layer, we double the resolution
        # and decrease the channel count by the specified factor.
        # The number of input channels is twice as the ones in the downsample layers,
        # since upsample layers have skip connections that take as input both the output of the previous layer,
        # and the output of the downsample layer at the same index.
        self.upsample_layers = nn.ModuleDict(
            {
                f"up_{i}": nn.ConvTranspose2d(
                    in_channels=self._channels[i + 1] * 2,
                    out_channels=self._channels[i],
                    kernel_size=3,
                    stride=2,
                    padding=0 if self._channels[i] % 2 == 0 else 1,
                    output_padding=1,
                )
                for i in range(len(self._channels) - 1)[::-1]
            }
        )

    def forward(
        self, t: TensorType["B", "int"], x: TensorType["B", "C", "H", "W", "float"]
    ) -> TensorType["B", "C", "H", "W", "float"]:
        # Store the output of each layer
        xs = []
        # Downsample pass
        for layer in self.downsample_layers.values():
            # Compute each downsample layer
            x = layer(x)
            x = nn.functional.relu(x)
            xs.append(x)
        # Upsample pass
        for layer in self.upsample_layers.values():
            # Concatenate each input with the output of the corresponding downsample layer, on the channel dimension
            x = torch.cat([x, xs.pop()], dim=1)
            x = layer(x)
            x = nn.functional.relu(x)
        return x
