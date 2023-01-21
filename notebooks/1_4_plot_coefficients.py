import os
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from segretini_matplottini.utils.plot_utils import reset_plot_style, save_plot
from torchtyping import TensorType
from tqdm import tqdm

from ddpm_from_scratch.ddpm import DDPM
from ddpm_from_scratch.models.spiral_denoising_model import \
    SpiralDenoisingModel
from ddpm_from_scratch.utils import (T, linear_beta_schedule,
                                     scaled_linear_beta_schedule)

PLOT_DIR = Path(__file__).parent.parent / "plots"
DATA_DIR = Path(__file__).parent.parent / "data"


def plot_forward_coefficients(betas: TensorType["T"], ddpm: DDPM, title: str, filename: str):
    reset_plot_style(xtick_major_pad=4, ytick_major_pad=4, border_width=1.5, label_pad=4, grid_linewidth=0.4)
    num_plots = 4
    num_timesteps = ddpm.num_timesteps
    assert len(betas) == num_timesteps
    _, ax = plt.subplots(nrows=num_plots, figsize=(8, 2.5 * num_plots), gridspec_kw=dict(hspace=0.3, top=0.95))
    ax[0].plot(np.arange(num_timesteps), betas, lw=1, label=r"$\beta_t$")
    ax[1].plot(np.arange(num_timesteps), ddpm.alphas, lw=1, label=r"$\alpha_t = 1 - \beta_t$")
    ax[2].plot(np.arange(num_timesteps), ddpm.alpha_cumprods, lw=1, label=r"$\bar{\alpha}_t$")
    ax[2].plot(np.arange(num_timesteps), ddpm.sqrt_alpha_cumprods, lw=1, label=r"$\sqrt{\bar{\alpha}_t}$")
    ax[3].plot(np.arange(num_timesteps), ddpm.sqrt_one_minus_alpha**2, lw=1, label=r"$1 - \bar{\alpha}_t$")
    ax[3].plot(np.arange(num_timesteps), ddpm.sqrt_alpha_cumprods, lw=1, label=r"$\sqrt{\bar{\alpha}_t}$")
    ax[3].annotate(
        r"$x_t\, \sim\, \mathcal{N}(\sqrt{\bar{\alpha}_t}x_0,\, (1 - \bar{\alpha}_t)\mathbf{I})$",
        xy=(0.02, 0.85),
        xycoords="axes fraction",
    )
    for _ax in ax:
        _ax.set_xlim(0, num_timesteps)
        _ax.legend(loc="upper right")
        _ax.grid(axis="x")
    plt.suptitle(title)
    save_plot(PLOT_DIR, filename, create_date_dir=False, bbox_inches="tight")


def plot_posterior_coefficients(betas: TensorType["T"], ddpm: DDPM, title: str, filename: str):
    reset_plot_style(xtick_major_pad=4, ytick_major_pad=4, border_width=1.5, label_pad=4, grid_linewidth=0.4)
    num_plots = 3
    num_timesteps = ddpm.num_timesteps
    assert len(betas) == num_timesteps
    _, ax = plt.subplots(nrows=num_plots, figsize=(8, 2.5 * num_plots), gridspec_kw=dict(hspace=0.3, top=0.93))
    ax[0].plot(np.arange(num_timesteps), ddpm.sqrt_one_minus_alpha**2, lw=1, label=r"$1 - \bar{\alpha}_t$")
    ax[0].plot(np.arange(num_timesteps), ddpm.sqrt_alpha_cumprods, lw=1, label=r"$\sqrt{\bar{\alpha}_t}$")
    ax[0].annotate(
        r"$x_t\, \sim\, \mathcal{N}(\sqrt{\bar{\alpha}_t}x_0,\, (1 - \bar{\alpha}_t)\mathbf{I})$",
        xy=(0.02, 0.85),
        xycoords="axes fraction",
    )
    ax[1].plot(np.arange(num_timesteps), ddpm.posterior_mean_x_0_coeff, lw=1, label=r"$\bar{\mu}_t,\, x_0$")
    ax[1].plot(np.arange(num_timesteps), ddpm.posterior_mean_x_t_coeff, lw=1, label=r"$\bar{\mu}_t,\, x_t$")
    ax[1].annotate(
        r"$\bar{\mu}_t = \frac{(1 - \bar{\alpha}_{t - 1}) \sqrt{\alpha_t}}{1 - \bar{\alpha}_t}x_t + \frac{\beta_t \sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_t}x_0$",
        xy=(0.02, 0.85),
        xycoords="axes fraction",
    )
    ax[2].plot(np.arange(num_timesteps), ddpm.posterior_variance, lw=1, label=r"$\bar{\beta}_t$")
    ax[2].annotate(
        r"$\bar{\beta}_t = \beta_t \frac{1 - \bar{\alpha}_{t - 1}}{1 - \bar{\alpha}_t}$",
        xy=(0.02, 0.85),
        xycoords="axes fraction",
    )
    for _ax in ax:
        _ax.set_xlim(0, num_timesteps)
        _ax.legend(loc="upper right")
        _ax.grid(axis="x")
    plt.suptitle(title)
    save_plot(PLOT_DIR, filename, create_date_dir=False, bbox_inches="tight")


#%%
if __name__ == "__main__":

    # Plot the coefficients of a beta schedule, using the default DDPM values
    num_timesteps = 1000
    betas = linear_beta_schedule(num_timesteps)
    model = SpiralDenoisingModel()
    ddpm = DDPM(num_timesteps, betas, model)

    # Plot the beta schedule, alphas, and the coefficients of the forward process q(x_t | x_0),
    # which defines x_t ~ N(sqrt(cumprod_alpha_t) * x_0, (1 - cumprod_alpha_t) * I).
    # We can see how the mean that defines the distribution of x_t goes to zero as the timestep increases,
    # while variance goes from to 1.
    plot_forward_coefficients(
        betas,
        ddpm,
        title="Linear " + r"$\beta_t$" + " schedule, " + r"$t=$" + f"{num_timesteps}",
        filename="1_4_linear_betas.png",
    )
    # When looking at the posterior process coefficients, i.e. q(x_t-1 | x_t, x_0) ~ N(mu_hat, beta_hat),
    # we see how the variance increases linearly over time,
    # while x_t is used to estimate x_t-1 except for the last few steps, when the weight of the estimate of x_0
    # becomes predominant, since the prediction becomes more confident.
    plot_posterior_coefficients(
        betas,
        ddpm,
        title="Linear "
        + r"$\beta_t$"
        + " schedule, "
        + r"$t=$"
        + f"{num_timesteps}, "
        + r"$q(x_{t-1} | x_t, x_0) \sim \mathcal{N}(\bar{\mu}_t,\, \bar{\beta}_t)$",
        filename="1_4_posterior_coefficients_linear_beta.png",
    )

    #%% Do the same thing, but now we use just 100 timesteps.
    # The shape of the curves is the same as using 1000 steps!
    num_timesteps = 100
    betas = linear_beta_schedule(num_timesteps)
    model = SpiralDenoisingModel()
    ddpm = DDPM(num_timesteps, betas, model)

    # Plot again the beta schedule, alphas, and the coefficients of the forward process q(x_t | x_0).
    plot_forward_coefficients(
        betas,
        ddpm,
        title="Linear " + r"$\beta_t$" + " schedule, " + r"$t=$" + f"{num_timesteps}",
        filename="1_4_linear_betas_100_steps.png",
    )
    # Here the coefficients of bar_mu are not quite the same. Proportionally,
    # the estimate of x_0 becomes relevant earlier in the denoising process.
    plot_posterior_coefficients(
        betas,
        ddpm,
        title="Linear "
        + r"$\beta_t$"
        + " schedule, "
        + r"$t=$"
        + f"{num_timesteps}, "
        + r"$q(x_{t-1} | x_t, x_0) \sim \mathcal{N}(\bar{\mu}_t,\, \bar{\beta}_t)$",
        filename="1_4_posterior_coefficients_linear_beta_100_steps.png",
    )

    #%% LDM and Stable Diffusion use a different beta schedule that decreases noise in a smoother way.
    num_timesteps = 1000
    betas = scaled_linear_beta_schedule(num_timesteps)
    model = SpiralDenoisingModel()
    ddpm = DDPM(num_timesteps, betas, model)

    # Plot again the beta schedule, alphas, and the coefficients of the forward process q(x_t | x_0).
    plot_forward_coefficients(
        betas,
        ddpm,
        title="Scaled linear " + r"$\beta_t$" + " schedule, " + r"$t=$" + f"{num_timesteps}",
        filename="1_4_scaled_linear_betas.png",
    )
    plot_posterior_coefficients(
        betas,
        ddpm,
        title="Scaled linear "
        + r"$\beta_t$"
        + " schedule, "
        + r"$t=$"
        + f"{num_timesteps}, "
        + r"$q(x_{t-1} | x_t, x_0) \sim \mathcal{N}(\bar{\mu}_t,\, \bar{\beta}_t)$",
        filename="1_4_posterior_coefficients_scaled_linear_betas.png",
    )

    #%% Why do we use those coefficients? Let's see what happens if we try different maximum betas.
    # Basically, they use the largest beta that results in unitary variance at step T.
    # Any smaller beta would still provide unitary variance, but the noise schedule would be less smooth.
    num_timesteps = 1000
    max_beta_range = np.geomspace(0.0001, 0.25, 50)
    with imageio.get_writer(PLOT_DIR / "1_4_scaled_linear_betas_range.gif", mode="I") as writer:  # Create a GIF!
        for i, b in tqdm(enumerate(max_beta_range)):
            betas = scaled_linear_beta_schedule(num_timesteps, β_end=b)
            model = SpiralDenoisingModel()
            ddpm = DDPM(num_timesteps, betas, model)
            plot_name = f"1_4_scaled_linear_betas_range_{i}.png"
            plot_forward_coefficients(
                betas,
                ddpm,
                title="Scaled linear "
                + r"$\beta_t$"
                + " schedule, "
                + r"$t=$"
                + f"{num_timesteps}, "
                + r"$\beta_T=$"
                + f"{b:.4f}",
                filename=plot_name,
            )
            image = imageio.imread(PLOT_DIR / plot_name)
            writer.append_data(image)
            os.remove(PLOT_DIR / plot_name)
            plt.close()
    #%% In this case, we see how the posterior variance changes greatly,
    # from being irrelevant to becoming extremely large in the timesteps closest to T.
    # We also see how the coefficients of bar_mu are not really affected by the choice of beta!
    # But if β_end is too high and num_timesteps is too small, we observe numerical instability,
    # with bar_mu coefficients collapsing to zero instead of remaining approximately complementary.
    num_timesteps = 1000
    with imageio.get_writer(
        PLOT_DIR / "1_4_posterior_coefficients_scaled_linear_betas_range.gif", mode="I"
    ) as writer:  # Create a GIF!
        for i, b in tqdm(enumerate(max_beta_range)):
            betas = linear_beta_schedule(num_timesteps, β_end=b)
            model = SpiralDenoisingModel()
            ddpm = DDPM(num_timesteps, betas, model)
            plot_name = f"1_4_posterior_coefficients_scaled_linear_betas_range_{i}.png"
            plot_posterior_coefficients(
                betas,
                ddpm,
                title="Scaled linear "
                + r"$\beta_t$"
                + " schedule, "
                + r"$t=$"
                + f"{num_timesteps}, "
                + r"$q(x_{t-1} | x_t, x_0) \sim \mathcal{N}(\bar{\mu}_t,\, \bar{\beta}_t),\ \beta_T=$"
                + f"{b:.4f}",
                filename=plot_name,
            )
            image = imageio.imread(PLOT_DIR / plot_name)
            writer.append_data(image)
            os.remove(PLOT_DIR / plot_name)
            plt.close()
