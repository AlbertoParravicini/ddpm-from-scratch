from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, cast

import torch
from jaxtyping import Float, Integer

COOL_GREEN = "#57bb8a"


Timestep = Union[int, list[int], Integer[torch.Tensor, "*#b #t"]]


def univariate_gaussian_sample(n: int, μ: float = 0, σ: float = 1) -> Float[torch.Tensor, " n"]:
    return μ + σ * torch.randn(n)


def make_spiral(
    n: int,
    start: float = 1.5 * torch.pi,
    end: float = 4.5 * torch.pi,
    normalize: bool = False,
) -> Float[torch.Tensor, "n 2"]:
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


@dataclass
class BetaSchedule(ABC):
    """
    Create a variance schedule (`beta schedule`) with values from a starting value
    to an ending value. Default values are the ones commonly used in LDM/Stable Diffusion.
    """

    num_train_timesteps: int = 1000
    """
    Reference value for the number timesteps. In DDPM, a large value (like 1000).
    The values of `beta` are multiplied by `num_train_timesteps` / `num_timesteps`.
    Using `num_timesteps < num_train_timesteps` allows to have a schedule with "shape" identical
    to using `num_timesteps == num_train_timesteps`, but defined over fewer timesteps.
    """

    @abstractmethod
    def betas(self, num_timesteps: int = 1000) -> Float[torch.Tensor, " t"]:
        """
        Generate a beta schedule with the specified number of steps.
        """
        pass


@dataclass
class LinearBetaSchedule(BetaSchedule):
    """
    Create a variance schedule (`beta schedule`) with linearly spaced values from a starting value
    to an ending value.
    Default values are taken from Stable Diffusion.
    The original DDPM paper used β_start=0.0001 and β_end=0.02.
    """

    β_start: float = 0.00085
    """
    Starting value of the beta schedule, at timestep 0
    """
    β_end: float = 0.012
    """
    Ending value of the beta schedule, at timestep T
    """

    def betas(self, num_timesteps: int = 1000) -> Float[torch.Tensor, " t"]:
        scale = self.num_train_timesteps / num_timesteps
        β_start = self.β_start * scale
        β_end = self.β_end * scale
        return torch.linspace(β_start, β_end, num_timesteps)


@dataclass
class ScaledLinearBetaSchedule(BetaSchedule):
    """
    Create a variance schedule (`beta schedule`) with linearly spaced values from a starting value
    to an ending value. The schedule is scaled by using the square root of the provided β values,
    and the overall schedule is squared. This scaling results in a smoother curve with
    noise variance that becomes lower earlier in the generation process, instead
    of becoming small only in the latest steps.
    Default values are the ones commonly used in LDM/Stable Diffusion.
    """

    β_start: float = 0.00085
    """
    Starting value of the beta schedule, at timestep 0
    """
    β_end: float = 0.012
    """
    Ending value of the beta schedule, at timestep T
    """

    def betas(self, num_timesteps: int = 1000) -> Float[torch.Tensor, " t"]:
        β_start = self.β_start**0.5
        β_end = self.β_end**0.5
        scale = (self.num_train_timesteps / num_timesteps) ** 0.5
        β_start *= scale
        β_end *= scale
        return cast(torch.Tensor, torch.linspace(β_start, β_end, num_timesteps) ** 2)


@dataclass
class CosineBetaSchedule(BetaSchedule):
    """
    Create a variance schedule (`beta schedule`) with a cosine progression.
    Nichol et al. (https://arxiv.org/pdf/2102.09672.pdf) found that this schedule distributes
    noise more evenly over the time range, instead of having a sharp reduction as in a liner schedule.
    """

    s: float = 0.008
    """
    Smoothing applied to the cosine schedule. The schedule follows a perfect cosine for `s = 0`,
    while for large `s` it will decrease faster to 0.
    """

    def betas(self, num_timesteps: int = 1000) -> Float[torch.Tensor, " t"]:
        t = torch.arange(0, num_timesteps + 1)
        # α_hat are defined so that α is 1 at timestep 0, 0 at timestep `num_timestep`,
        # and the progression follows a cosine curve, with a smoothing controlled by `s`.
        # Each x_t is a Gaussian with mean α_hat_t.
        α_hat = torch.cos(((t / num_timesteps + self.s) / (1 + self.s)) * (torch.pi / 2)) ** 2
        α_hat = α_hat / α_hat[0]  # Ensure that α_hat[0] is 1
        β = 1 - α_hat[1:] / α_hat[:-1]  # α = α_hat[1:] / α_hat[:-1], β = 1 - α
        β = torch.clamp(β, 0, 0.999)
        return cast(torch.Tensor, β)


def expand_to_dims(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Expand the shape of `x` to match the number of dimensions of `y`, by adding
    size-1 dimensions to `x`. For example, if `x` has shape `[4]` and `y` has shape `[4, 2, 3]`,
    the expanded `x` has shape `[4, 1, 1]`.
    This is useful to broadcast a vector of values (`x`) over all the other dimensions of `y`.
    We use this function to add 1D conditioning embeddings over the 2D structure of an image in latent space.

    :param x: a tensor to expand
    :param y: tensor whose number of dimensions must be matched
    :return: the expanded input tensor
    """
    return x[(...,) + (None,) * (len(y.shape) - len(x.shape))]


def gaussian_frechet_distance(
    μ_1: Float[torch.Tensor, " n"],
    Σ_1: Float[torch.Tensor, "n n"],
    μ_2: Float[torch.Tensor, " n"],
    Σ_2: Float[torch.Tensor, "n n"],
) -> Float[torch.Tensor, "1"]:
    """
    Frechet Inception Distance (FID) between two multivariate Gaussian distributions.
    It measures how different the two distributions are: intuitively, it looks for the difference of means,
    and for how different the "rotations" of the covariance matrices are.
    It is computed as `|μ_1 - μ_2|**2 + trace(Σ_1 + Σ_2 - 2*sqrt(Σ_1 * Σ_2))`.

    :return: the FID of the two distributions.
    """
    diff = μ_1 - μ_2
    return torch.sum(diff * diff) + torch.trace(Σ_1 + Σ_2 - 2 * torch.sqrt(Σ_1 * Σ_2))
