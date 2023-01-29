from typing import Sequence

import torch
import torch.nn as nn
from torchtyping import TensorType

from ddpm_from_scratch.models.spiral_denoising_model import SinusoidalEncoding
from ddpm_from_scratch.models.unet_simple import UNetSimple
from ddpm_from_scratch.utils import B, C, H, W, expand_to_dims


class TimestepEmbedding(nn.Module):
    def __init__(self, output_channels: int, hidden_channels: int = 16):
        super().__init__()
        self.timestep_embedding = nn.Sequential(
            SinusoidalEncoding(hidden_channels, maximum_length=1024),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, output_channels),
        )

    def forward(self, t: TensorType["B", "int"]) -> TensorType["B", "C"]:
        return self.timestep_embedding(t)


class UNetSimpleWithTimestep(UNetSimple):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
        channel_multipliers: Sequence[int] = (1, 2, 3),
    ) -> None:
        """
        A simple UNet model, with time-conditioning.
        It has a few downsample 2D 3x3 convolutional layers with stride 2, followed by
        upsample 2D transposed 3x3 convolution layers that take as input both the previous layer's output
        and the output of the corresponding downsample layer.
        The timestep is encoded with sinusoidal encoding, and added to the output of each layer,
        broadcasted on the spatial coordinates.

        :param in_channels: input channels of the first layer. 1 for grayscale images, 3 for RGB images
        :param hidden_channels: base number of channels of hidden layers. The number of channels in layer `i`
            is obtained as `hidden_channels * channel_multipliers[i]`.
        :param channel_multipliers: a sequence of integers that specifies the number of down/upsample layers.
            The number of channels in layer `i` is given by `hidden_channels * channel_multipliers[i]`.
        """
        super().__init__(
            in_channels=in_channels, hidden_channels=hidden_channels, channel_multipliers=channel_multipliers
        )
        # Add a sinusoidal timestep encoding for each layer,
        # with the same size as the number of output channels of that layer
        self.downsample_timesteps = nn.ModuleDict(
            {f"timestep_down_{i}": TimestepEmbedding(self._channels[i + 1]) for i in range(len(self._channels) - 1)}
        )
        self.upsample_timesteps = nn.ModuleDict(
            {f"timestep_up_{i}": TimestepEmbedding(self._channels[i]) for i in range(len(self._channels) - 1)[::-1]}
        )

    def forward(
        self, t: TensorType["B", "int"], x: TensorType["B", "C", "H", "W", "float"]
    ) -> TensorType["B", "C", "H", "W", "float"]:
        # Store the output of each layer
        xs = []
        # Downsample pass
        for layer, emb in zip(self.downsample_layers.values(), self.downsample_timesteps.values()):
            # Compute each downsample layer
            x = layer(x)
            # Add the timestep to the layer output
            x = x + expand_to_dims(emb(t), x)  # Replicate time embedding to H and W
            x = nn.functional.relu(x)
            xs.append(x)
        # Upsample pass
        for layer, emb in zip(self.upsample_layers.values(), self.upsample_timesteps.values()):
            # Concatenate each input with the output of the corresponding downsample layer, on the channel dimension
            x = torch.cat([x, xs.pop()], dim=1)
            x = layer(x)
            # Add the timestep to the layer output
            x = x + expand_to_dims(emb(t), x)  # Replicate time embedding to H and W
            x = nn.functional.relu(x)
        return x
