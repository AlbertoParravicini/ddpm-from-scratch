from typing import Sequence, TypeVar, Union

import torch
from torchtyping import TensorType

COOL_GREEN = "#57bb8a"

B = TypeVar("B")  # Batch size
C = TypeVar("C")  # Number of color channels
W = TypeVar("W")  # Width
H = TypeVar("H")  # Height
N = TypeVar("N")  # Generic size
M = TypeVar("M")  # Generic size
T = TypeVar("T")  # Timesteps

Timestep = Union[int, Sequence[int], TensorType["B", "int"]]


def univariate_gaussian_sample(n: int, μ: float = 0, σ: float = 1) -> TensorType["N", "float"]:
    return μ + σ * torch.randn(n)


def make_spiral(
    n: int, start: float = 1.5 * torch.pi, end: float = 4.5 * torch.pi, normalize: bool = False
) -> TensorType[2, "N"]:
    """
    Create a spiral with the specified number of samples. The starting point has angle `start`,
    defined starting from 0 * np.pi with radius `start`, while the ending angle and radius defined by `end`,
    obtained as the specified number of rotations with radius increased by π for each rotation.

    :param n: number of samples
    :param start: angle of the starting point, with radius `start`
    :param end: angle of the ending point, with radius equal to `(end - start) / π`
    :param normalize: if True, normalize the output so that it's contained in [0, 1], horizontally and vertically
    :return: array of shape `[n, 2]` of x and y coordinates of the spiral
    """
    t_min = start
    t_max = end

    t = torch.rand(n) * (t_max - t_min) + t_min

    x = t * torch.cos(t)
    y = t * torch.sin(t)

    if normalize:
        x_min, x_max = torch.min(x), torch.max(x)
        x = (x - x_min) / (x_max - x_min)
        y_min, y_max = torch.min(y), torch.max(y)
        y = (y - y_min) / (y_max - y_min)
        x = 2 * x - 1
        y = 2 * y - 1
    return torch.stack([x, y]).T


def linear_beta_schedule(
    num_timesteps: int = 1000, β_start: float = 0.00085, β_end: float = 0.012, num_train_timesteps: int = 1000
) -> TensorType["T"]:
    """
    Create a variance schedule (`beta schedule`) with linearly spaced values from a starting value
    to an ending value. Default values are the ones commonly used in LDM/Stable Diffusion.

    :param num_timesteps: number of values in the generated schedule.
    :param β_start: starting value of the beta schedule, at timestep 0
    :param β_end: ending value of the beta schedule, at timestep T
    :param num_train_timesteps: reference value for the number timesteps. In DDPM, a large value (like 1000).
        The values of `beta` are multiplied by `num_train_timesteps` / `num_timesteps`.
        Using `num_timesteps < num_train_timesteps` allows to have a schedule with "shape" identical
        to using `num_timesteps == num_train_timesteps`, but defined over fewer timesteps.
    :return: the generated beta schedule
    """
    scale = num_train_timesteps / num_timesteps
    β_start *= scale
    β_end *= scale
    return torch.linspace(β_start, β_end, num_timesteps)


def scaled_linear_beta_schedule(
    num_timesteps: int = 1000, β_start: float = 0.00085, β_end: float = 0.012, num_train_timesteps: int = 1000
) -> TensorType["T"]:
    """
    Create a variance schedule (`beta schedule`) with linearly spaced values from a starting value
    to an ending value. The schedule is scaled by using the square root of the provided β values,
    and the overall schedule is squared. This scaling results in a smoother curve with
    noise variance that becomes lower earlier in the generation process, instead
    of becoming small only in the latest steps.
    Default values are the ones commonly used in LDM/Stable Diffusion.

    :param num_timesteps: number of values in the generated schedule.
    :param β_start: starting value of the beta schedule, at timestep 0
    :param β_end: ending value of the beta schedule, at timestep T
    :param num_train_timesteps: reference value for the number timesteps. In DDPM, a large value (like 1000).
        The values of `beta` are multiplied by `num_train_timesteps` / `num_timesteps`.
        Using `num_timesteps < num_train_timesteps` allows to have a schedule with "shape" identical
        to using `num_timesteps == num_train_timesteps`, but defined over fewer timesteps.
    :return: the generated beta schedule
    """
    return linear_beta_schedule(num_timesteps, β_start**0.5, β_end**0.5, num_train_timesteps) ** 2


def expand_to_dims(x: torch.Tensor, y: torch.Tensor):
    """
    Expand the shape of `x` to match the number of dimensions of `y`, by adding
    size-1 dimensions to `x`. For example, if `x` has shape `[4]` and `y` has shape `[4, 2, 3]`,
    the expanded `x` has shape `[4, 1, 1]`.
    This is useful to broadcast an array of values (`x`) over all the other dimensions of `y`.

    :param x: a tensor to expand
    :param y: tensor whose number of dimensions must be matched
    :return: the expanded input tensor
    """
    return x[(...,) + (None,) * (len(y.shape) - len(x.shape))]
