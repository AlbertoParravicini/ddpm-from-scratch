from typing import Sequence, cast, Optional

import torch
import torch.nn as nn
from jaxtyping import Float, Integer

from ddpm_from_scratch.utils import expand_to_dims
from ddpm_from_scratch.models.spiral_denoising_model import SinusoidalEncoding


class EmbeddingProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 16):
        """
        A learnable block that projects an embedding (e.g. a timestep embedding) into the desired size.
        The embedding is passed through a non-linear transformation,
        and the output of the block has size `out_channels`.
        """
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, e: Integer[torch.Tensor, " b"]) -> Float[torch.Tensor, "b c"]:
        return cast(torch.Tensor, self.embedding(e))


class UNetWithConditioning(nn.Module):
    def __init__(
        self,
        classes: int,
        in_channels: int = 1,
        hidden_channels: int = 16,
        channel_multipliers: Sequence[int] = (1, 2, 3),
    ) -> None:
        """
        A simple UNet model, with time-conditioning and class-conditioning.
        It has a few downsample 2D 3x3 convolutional layers with stride 2, followed by
        upsample 2D transposed 3x3 convolution layers that take as input both the previous layer's output
        and the output of the corresponding downsample layer.
        The timestep is encoded with sinusoidal encoding, and added to the output of each layer,
        projected to the layer number of channels and broadcasted on the spatial coordinates.
        The class conditioning is a learnt embedding, also added to the output of each layer,
        projected to the layer number of channels and broadcasted on the spatial coordinates.

        :param classes: number of classes for the class conditioning. The model learns an embedding for each class,
            plus an embedding for `no class`, used for unconditional generation.
        :param in_channels: input channels of the first layer. 1 for grayscale images, 3 for RGB images
        :param hidden_channels: base number of channels of hidden layers. The number of channels in layer `i`
            is obtained as `hidden_channels * channel_multipliers[i]`.
        :param channel_multipliers: a sequence of integers that specifies the number of down/upsample layers.
            The number of channels in layer `i` is given by `hidden_channels * channel_multipliers[i]`.
        """
        super().__init__()
        self._channels = [in_channels] + [hidden_channels * c for c in channel_multipliers]
        # Add a sinusoidal timestep encoding at the start, and a projection layer to every layer
        # that is usd to add the embedding. It has the same size as the number of output channels of that layer.
        # There is also a learnable embedding for the class, with `classes + 1` embeddings
        # (the extra one is for unconditional generation).
        self.classes = classes
        self.time_embedding = SinusoidalEncoding(hidden_channels, maximum_length=1024)
        self.class_embedding = nn.Embedding(num_embeddings=classes + 1, embedding_dim=hidden_channels)
        self.downsample_timesteps = nn.ModuleDict(
            {
                f"timestep_down_{i}": EmbeddingProjection(hidden_channels, self._channels[i + 1])
                for i in range(len(self._channels) - 1)
            }
        )
        self.upsample_timesteps = nn.ModuleDict(
            {
                f"timestep_up_{i}": EmbeddingProjection(hidden_channels, self._channels[i])
                for i in range(len(self._channels) - 1)[::-1]
            }
        )
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
        self,
        t: Integer[torch.Tensor, " b"],
        x: Float[torch.Tensor, "b c h w"],
        c: Optional[Integer[torch.Tensor, " b"]] = None,
    ) -> Float[torch.Tensor, "b c h w"]:
        # Timestep embedding
        t_emb = self.time_embedding(t)
        # Class embedding
        c_emb = (
            self.class_embedding(c)
            if c is not None
            else self.class_embedding(torch.tensor(self.classes, device=x.device))
        )
        # Combine time and class embeddings
        e = t_emb + c_emb
        # Store the output of each layer
        xs = []
        # Downsample pass
        for layer, emb in zip(self.downsample_layers.values(), self.downsample_timesteps.values()):
            # Compute each downsample layer
            x = layer(x)
            # Add the timestep to the layer output
            x = x + expand_to_dims(emb(e), x)  # Replicate time embedding to H and W
            x = nn.functional.relu(x)
            xs.append(x)
        # Upsample pass
        for i, (layer, emb) in enumerate(zip(self.upsample_layers.values(), self.upsample_timesteps.values())):
            # Concatenate each input with the output of the corresponding downsample layer, on the channel dimension
            x = torch.cat([x, xs.pop()], dim=1)
            x = layer(x)
            # Add the timestep to the layer output
            x = x + expand_to_dims(emb(e), x)  # Replicate time embedding to H and W
            x = nn.functional.relu(x) if i < len(self.upsample_layers) - 1 else x  # Don't apply ReLU to the last layer
        return x
