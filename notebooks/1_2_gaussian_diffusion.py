import os
from pathlib import Path
from typing import Callable, Optional

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from segretini_matplottini.utils.colors import PALETTE_1
from segretini_matplottini.utils.plot_utils import reset_plot_style, save_plot

PLOT_DIR = Path(__file__).parent.parent / "plots"


def make_spiral(
    n: int, start: float = 1.5 * np.pi, end: float = 4.5 * np.pi, normalize: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a spiral with the specified number of samples. The starting point has angle `start`,
    defined starting from 0 * np.pi with radius `start`, while the ending angle and radius defined by `end`,
    obtained as the specified number of rotations with radius increased by π for each rotation.

    :param n: number of samples
    :param start: angle of the starting point, with radius `start`
    :param end: angle of the ending point, with radius equal to `(end - start) / π`
    :param normalize: if True, normalize the output so that it's contained in [0, 1], horizontally and vertically
    :return: arrays of x and y coordinates of the spiral
    """
    t_min = start
    t_max = end

    t = np.random.rand(n) * (t_max - t_min) + t_min

    x = t * np.cos(t)
    y = t * np.sin(t)

    if normalize:
        x_min, x_max = np.min(x), np.max(x)
        x = (x - x_min) / (x_max - x_min)
        y_min, y_max = np.min(y), np.max(y)
        y = (y - y_min) / (y_max - y_min)
        x = 2 * x - 1
        y = 2 * y - 1
    return x, y


def linear_beta_schedule(
    num_timesteps: int = 1000, β_start: float = 0.00085, β_end: float = 0.012, num_train_timesteps: int = 1000
) -> np.ndarray:
    """
    Create a variance schedule (`beta schedule`) with linearly spaced values from a starting value
    to an ending value. Default values are the ones commonly used in LDM/Stable Diffusion.

    :param num_timesteps: number of values in the generated schedule
    :param β_start: starting value of the beta schedule, at timestep 0
    :param β_end: ending value of the beta schedule, at timestep T
    :param num_train_timesteps: reference value for the number timesteps. In DDPM, a large value (like 1000).
        The values of `beta` are multiplied by `num_train_timesteps` / `num_timesteps`,
        to allow for a faster sampling process
    :return: the generated beta schedule
    """
    scale = num_train_timesteps / num_timesteps
    β_start *= scale
    β_end *= scale
    return np.linspace(β_start, β_end, num_timesteps)


class GaussianDiffusion:
    def __init__(
        self, num_timesteps: int, betas: np.ndarray, denoise_function: Callable[[int, np.ndarray], np.ndarray]
    ) -> None:
        assert num_timesteps == len(betas), "the number of timesteps must be the same as the number of betas"

        # The number of timesteps is used as a way to scale the original betas
        # to the amount used at inference time. It should be a large number, e.g. 1000 in DDPM,
        # to obtain a smooth sampling process.
        self.num_timesteps = num_timesteps

        ############################################
        # Coefficients used by the forward process #
        ############################################

        self.betas = betas
        self.alphas = 1 - betas
        self.alpha_cumprods = np.cumprod(self.alphas)
        self.alpha_cumprods_prevs = np.concatenate([np.array([1.0]), self.alpha_cumprods[:-1]])
        self.sqrt_alpha_cumprods = np.sqrt(self.alpha_cumprods)
        self.sqrt_one_minus_alpha = np.sqrt(1 - self.alpha_cumprods)

        #########################################################
        # Coefficients used by the backward process / posterior #
        #########################################################

        # 1 / sqrt(α_hat_t), used to estimate x_hat_0
        self.sqrt_reciprocal_alpha_cumprods = 1 / np.sqrt(self.alpha_cumprods)
        # sqrt(1 / α_hat_t - 1), used to estimate x_hat_0
        self.sqrt_reciprocal_alpha_cumprods_minus_one = np.sqrt(1 / self.alpha_cumprods - 1)
        # "beta_hat_t", β_t * (1 - α_hat_t-1) /  (1 - α_hat_t), variance of q(x_t-1 | x_t, x_0)
        self.posterior_variance = self.betas * (1 - self.alpha_cumprods_prevs) / (1 - self.alpha_cumprods)
        # Used to avoid exponentials. Clipping done for numerical stability
        self.log_clipped_posterior_variance = np.log(np.maximum(self.posterior_variance, 1e-20))
        # "alpha_hat_t", mean of the backward process, as linear combination of x_t and x_0.
        self.posterior_mean_x_0_coeff = self.betas * np.sqrt(self.alpha_cumprods_prevs) / (1 - self.alpha_cumprods)
        self.posterior_mean_x_t_coeff = (
            (1 - self.alpha_cumprods_prevs) * np.sqrt(self.alphas) / (1 - self.alpha_cumprods)
        )

        ################################
        # Function learnt by the model #
        ################################

        self.denoise_function = denoise_function

    def forward_sample(
        self, t: int, x_start: np.ndarray, noise: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute `q(x_i | x_i-t)`, as a sample of a Gaussian process with equation
        ```
        sqrt_alpha_cumprods[t] * x_start + sqrt_one_minus_alpha[t] * noise
        ```

        :param t: current timestep, as integer. It must be `[0, self.num_timesteps]`
        :param x_start: value of `x_i-t`, the value on which the forward process q is conditioned
        :param noise: noise added to the forward process. If None, sample from a standard Gaussian
        :return: the sampled value of `q(x_i | x_i-t)`, and the added noise
        """
        if noise is None:
            noise = np.random.randn(*x_start.shape)
        return self.sqrt_alpha_cumprods[t] * x_start + self.sqrt_one_minus_alpha[t] * noise, noise

    def _predict_x_0_and_noise(self, t: int, x_start: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute a sample of the backward process `q(x_0 | x_t)`, by denoising `x_t` using a model,
        and return both the sample and the predicted noise.
        This is computed as `x_hat_0 = 1/sqrt(α_hat_t) * x_t  - sqrt(1 - α_hat_t) / sqrt(α_hat_t)
        """
        noise = self.denoise_function(t, x_start)
        coeff_x_t = self.sqrt_reciprocal_alpha_cumprods[t]
        coeff_noise = self.sqrt_reciprocal_alpha_cumprods_minus_one[t]
        x_hat_0 = coeff_x_t * x_start - coeff_noise * noise / coeff_x_t
        return x_hat_0, noise

    def _posterior_mean_variance(self, t: int, x_start: np.ndarray, x_t: np.ndarray) -> tuple[float, float]:
        """
        Obtain the mean and variance of q(x_t-1 | x_t, x_0)
        """
        posterior_mean_x_0_coeff = self.posterior_mean_x_0_coeff[t]
        posterior_mean_x_t_coeff = self.posterior_mean_x_t_coeff[t]
        posterior_mean = posterior_mean_x_0_coeff * x_start + posterior_mean_x_t_coeff * x_t
        posterior_variance = self.posterior_variance[t]
        return posterior_mean, posterior_variance

    def backward_sample(
        self,
        t: int,
        x_start: np.ndarray,
        clip_predicted_x_0: bool = True,
        add_noise: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Obtain a sample of the backward process `q(x_t-1 | x_t, x_0)`,
        by predicting `x_0` using a denoising model, and then taking a step of the backward process.

        :param t: timestep of `x_t`
        :param x_start: value of `x_t`
        :param clip_predicted_x_0: if True, clip the predicted value of `x_0` in `[-1, 1]`
            This is meaningful only for denoising the spiral! We mights other values for images
        :param add_noise: if True, add noise, scaled by the posterior variance, to the predicted sample of `x_t-1`.
            If False, the backward sample is deterministic. It should be False for `t = 0`, True otherwise (in DDPM)
        :return: the sample of `x_t-1`, and the predicted `x_0`
        """
        # Predict x_0 using the model
        x_hat_0, _ = self._predict_x_0_and_noise(t, x_start)
        if clip_predicted_x_0:
            x_hat_0 = np.clip(x_hat_0, -1, 1)
        # Obtain the posterior mean and variance, and obtain a sample of q(x_t-1 | x_t, x_0)
        posterior_mean, posterior_variance = self._posterior_mean_variance(t, x_start=x_hat_0, x_t=x_start)
        x_t_minus_one = np.array(posterior_mean)
        # Add noise to the sample, instead of taking a deterministic step
        if add_noise:
            noise: np.ndarray = np.random.randn(*x_start.shape)
            x_t_minus_one += np.sqrt(posterior_variance) * noise
        return x_t_minus_one, x_hat_0


#%%
if __name__ == "__main__":

    # Setup
    np.random.seed(seed=42)
    reset_plot_style(xtick_major_pad=4, ytick_major_pad=4, border_width=1.5, label_pad=4)
    PLOT_DIR.mkdir(exist_ok=True, parents=True)

    # Create a spiral, and add noise to it. Create a β-schedule identical to before
    x, y = make_spiral(1000, normalize=True)
    num_timesteps = 1000
    βs = linear_beta_schedule(num_timesteps, 8e-6, 9e-5)
    X = np.array([x, y]).T  # Turn the spiral into a matrix, this is timestep 0

    # Create the diffusion forward process.
    # We need to provide a denoising function. Here we just return random noise, since we are not training a model yet
    gaussian_diffusion = GaussianDiffusion(num_timesteps, βs, denoise_function=lambda t, x: np.random.randn(*x.shape))

    # Now we work in continous time and with arbitrary timesteps.
    # Given the continous timestep in [0, 1], we index the closest
    # discrete timestep, and compute X_t directly from X_0
    discrete_steps = 5
    samples = [X]
    fig, ax = plt.subplots(ncols=discrete_steps, figsize=(6 * discrete_steps, 6))
    for i, t in enumerate(np.linspace(0, 1, discrete_steps)):
        timestep = min(int(t * num_timesteps), num_timesteps - 1)
        sample, _ = gaussian_diffusion.forward_sample(timestep, X)
        samples += [sample]
        # Plot the spiral
        ax[i].scatter(sample[:, 0], sample[:, 1], color=PALETTE_1[-2], alpha=0.8, edgecolor="#2f2f2f", lw=0.5)
        ax[i].set_title(f"A spiral becoming noise, step {t}")
        ax[i].set_xlim((-1, 1))  # enforce axes limits as we add noise
        ax[i].set_ylim((-1, 1))
    save_plot(PLOT_DIR, "1_2_gaussian_diffusion_forward.png", create_date_dir=False, bbox_inches="tight")

    # Try removing noise from the spiral. We don't have a model, so our "model" will just return random noise
    X_noisy = samples[-1]
    with imageio.get_writer(PLOT_DIR / "1_2_gaussian_diffusion_backward.gif", mode="I") as writer:  # Create a GIF!
        for i, t in enumerate(np.linspace(1, 0, num_timesteps)):
            # Get timestep, in the range [0, num_timesteps)
            timestep = min(int(t * num_timesteps), num_timesteps - 1)
            # Inference, predict the next step given the current one
            X_noisy, X_0 = gaussian_diffusion.backward_sample(timestep, X_noisy, add_noise=t != 0)
            # Plot the current spiral, every few steps
            if timestep % (num_timesteps // 20) == 0 or timestep == num_timesteps - 1:
                plt.figure(figsize=(6, 6))
                plt.scatter(X_noisy[:, 0], X_noisy[:, 1], color=PALETTE_1[-2], alpha=0.8, edgecolor="#2f2f2f", lw=0.5)
                plt.title("Noise becoming a spiral? " + r"$q(x_{t - 1} | x_t, \hat{x}_0), t=$" + f"{timestep}")
                plt.xlim((-1, 1))
                plt.ylim((-1, 1))
                # Create a temporary file to assemble the GIF
                filename = f"1_2_gaussian_diffusion_backward_{timestep}.jpeg"
                save_plot(PLOT_DIR, filename, create_date_dir=False)
                image = imageio.imread(PLOT_DIR / filename)
                writer.append_data(image)
                os.remove(PLOT_DIR / filename)
                plt.close()
