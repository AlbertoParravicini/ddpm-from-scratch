from math import log

import torch
import torch.nn as nn
from torchtyping import TensorType

from ddpm_from_scratch.utils import B, N


class SinusoidalEncoding(nn.Module):
    def __init__(self, size, maximum_length: int = 5000) -> None:
        """
        Sinusoidal Positional Encoding introduced by Vaswani et al. [1].
        Use a fixed trigonometric encoding for the position of an element in a sequence,

        [1] Vaswani et al. (https://arxiv.org/abs/1706.03762)

        Implementation inspired by `nncore`,
        link: https://github.com/yeliudev/nncore/blob/main/nncore/nn/blocks/transformer.py

        :param size: Size of each positional encoding vector.
        :param maximum_length: The maximum length of the input sequence.
        """
        super().__init__()

        self._size = size
        self._maximum_length = maximum_length
        self._max_period = 10000.0

        # Trigonometric encoding following [1].
        pos = torch.arange(self._maximum_length).unsqueeze(1)
        div = torch.exp(torch.arange(0, self._size, 2) * -log(self._max_period) / self._size)
        pe = torch.zeros(self._maximum_length, self._size)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, t: TensorType["B", "int"]) -> TensorType["B", "N", "float"]:
        # Extract the encodings specified by the input timesteps
        return self.pe[t]


class SpiralDenoisingModel(nn.Module):
    """
    Simple feed-forward network used to denoise a 2D spiral with Gaussian noise applied to it.
    The function is used as denoiser for the backward process of a DDPM.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dense_1 = nn.Linear(2, 32)
        self.embedding_1 = SinusoidalEncoding(32, maximum_length=1000)
        self.dense_2 = nn.Linear(32, 32)
        self.embedding_2 = SinusoidalEncoding(32, maximum_length=1000)
        self.dense_3 = nn.Linear(32, 32)
        self.embedding_3 = SinusoidalEncoding(32, maximum_length=1000)
        self.dense_4 = nn.Linear(32, 2)

    def forward(
        self, t: TensorType["B", "int"], x: TensorType["B", "N", "2", "float"]
    ) -> TensorType["B", "N", "2", "float"]:
        # If x is given without batch size, we remove it at the end
        x_shape_len = len(x.shape)
        # Sum the positional embedding at every layer.
        # We unsqueeze it on the second dimension so that it can be broadcasted over the output the dense layer.
        # This assumes that the batch dimension is present.
        x = self.dense_1(x) + self.embedding_1(t).unsqueeze(1)
        x = nn.functional.silu(x)
        x = self.dense_2(x) + self.embedding_2(t).unsqueeze(1)
        x = nn.functional.silu(x)
        x = self.dense_3(x) + self.embedding_3(t).unsqueeze(1)
        x = self.dense_4(x)
        # Remove batch size if it was not present
        if len(x.shape) > x_shape_len:
            x = x.squeeze(dim=0)
        return x
