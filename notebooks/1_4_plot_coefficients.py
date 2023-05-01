import os
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from segretini_matplottini.utils.plot_utils import reset_plot_style, save_plot
from tqdm import tqdm

from ddpm_from_scratch.models import SpiralDenoisingModel
from ddpm_from_scratch.samplers import DDPM
from ddpm_from_scratch.utils import BetaSchedule, LinearBetaSchedule, ScaledLinearBetaSchedule

PLOT_DIR = Path(__file__).parent.parent / "plots"
DATA_DIR = Path(__file__).parent.parent / "data"


def plot_forward_coefficients(
    ddpm: DDPM,
    title: str,
    filename: str,
    β_start: float = 1e-6,
    β_end: float = 0.015,
) -> None:
    reset_plot_style(
        xtick_major_pad=4,
        ytick_major_pad=4,
        border_width=1.5,
        label_pad=4,
        grid_linewidth=0.4,
    )
    num_plots = 4
    num_timesteps = ddpm.num_timesteps
    betas = ddpm.betas
    assert len(betas) == num_timesteps
    _, ax = plt.subplots(
        nrows=num_plots,
        figsize=(8, 2.5 * num_plots),
        gridspec_kw=dict(hspace=0.3, top=0.95),
    )
    ax[0].plot(np.arange(num_timesteps), betas, lw=1, label=r"$\beta_t$")
    ax[1].plot(
        np.arange(num_timesteps),
        ddpm.alphas,
        lw=1,
        label=r"$\alpha_t = 1 - \beta_t$",
    )
    ax[2].plot(
        np.arange(num_timesteps),
        ddpm.alpha_cumprods,
        lw=1,
        label=r"$\bar{\alpha}_t$",
    )
    ax[2].plot(
        np.arange(num_timesteps),
        ddpm.alpha_cumprods**0.5,
        lw=1,
        label=r"$\sqrt{\bar{\alpha}_t}$",
    )
    ax[3].plot(
        np.arange(num_timesteps),
        1 - ddpm.alpha_cumprods,
        lw=1,
        label=r"$1 - \bar{\alpha}_t$",
    )
    ax[3].plot(
        np.arange(num_timesteps),
        ddpm.alpha_cumprods**0.5,
        lw=1,
        label=r"$\sqrt{\bar{\alpha}_t}$",
    )
    ax[3].annotate(
        r"$x_t\, \sim\, \mathcal{N}(\sqrt{\bar{\alpha}_t}x_0,\, (1 - \bar{\alpha}_t)\mathbf{I})$",
        xy=(0.02, 0.85),
        xycoords="axes fraction",
    )
    ax[3].annotate(
        r"$\beta_T = $" + f"{betas[-1]:.4f}",
        xy=(0.86, 0.05),
        xycoords="axes fraction",
    )
    for _ax in ax:
        _ax.set_xlim(0, num_timesteps)
        _ax.legend(loc="upper right")
        _ax.grid(axis="x")
        # Enforce consistent formatting for the y-axis, so we can create smooth GIFs.
        _ax.yaxis.set_major_locator(plt.LinearLocator(numticks=5))
        _ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.4f}")
    ax[0].set_ylim(β_start, β_end)
    ax[1].set_ylim(1 - β_end, 1 - β_start)
    ax[2].set_ylim(0, 1)
    ax[3].set_ylim(0, 1)
    plt.suptitle(title)
    save_plot(PLOT_DIR, filename, create_date_dir=False, bbox_inches="tight")


def plot_posterior_coefficients(ddpm: DDPM, title: str, filename: str, β_end: float = 0.015) -> None:
    reset_plot_style(
        xtick_major_pad=4,
        ytick_major_pad=4,
        border_width=1.5,
        label_pad=4,
        grid_linewidth=0.4,
    )
    num_plots = 3
    num_timesteps = ddpm.num_timesteps
    betas = ddpm.betas
    assert len(betas) == num_timesteps
    _, ax = plt.subplots(
        nrows=num_plots,
        figsize=(8, 2.5 * num_plots),
        gridspec_kw=dict(hspace=0.3, top=0.93),
    )
    ax[0].plot(
        np.arange(num_timesteps),
        1 - ddpm.alpha_cumprods,
        lw=1,
        label=r"$1 - \bar{\alpha}_t$",
    )
    ax[0].plot(
        np.arange(num_timesteps),
        ddpm.alpha_cumprods**0.5,
        lw=1,
        label=r"$\sqrt{\bar{\alpha}_t}$",
    )
    ax[0].annotate(
        r"$x_t\, \sim\, \mathcal{N}(\sqrt{\bar{\alpha}_t}x_0,\, (1 - \bar{\alpha}_t)\mathbf{I})$",
        xy=(0.02, 0.85),
        xycoords="axes fraction",
    )
    ax[0].annotate(
        r"$\beta_T = $" + f"{betas[-1]:.4f}",
        xy=(0.86, 0.05),
        xycoords="axes fraction",
    )
    posterior_mean_x_0_coeff = ddpm.betas * torch.sqrt(ddpm.alpha_cumprods_prevs) / (1 - ddpm.alpha_cumprods)
    posterior_mean_x_t_coeff = (1 - ddpm.alpha_cumprods_prevs) * torch.sqrt(ddpm.alphas) / (1 - ddpm.alpha_cumprods)
    posterior_variance = ddpm.betas * (1 - ddpm.alpha_cumprods_prevs) / (1 - ddpm.alpha_cumprods)
    ax[1].plot(
        np.arange(num_timesteps),
        posterior_mean_x_0_coeff,
        lw=1,
        label=r"$\bar{\mu}_t,\, x_0$",
    )
    ax[1].plot(
        np.arange(num_timesteps),
        posterior_mean_x_t_coeff,
        lw=1,
        label=r"$\bar{\mu}_t,\, x_t$",
    )
    ax[1].annotate(
        r"$\bar{\mu}_t = \frac{(1 - \bar{\alpha}_{t - 1}) \sqrt{\alpha_t}}{1 - \bar{\alpha}_t}x_t + \frac{\beta_t \sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_t}x_0$",  # noqa: E501
        xy=(0.02, 0.85),
        xycoords="axes fraction",
    )
    ax[2].plot(
        np.arange(num_timesteps),
        posterior_variance,
        lw=1,
        label=r"$\bar{\beta}_t$",
    )
    ax[2].annotate(
        r"$\bar{\beta}_t = \beta_t \frac{1 - \bar{\alpha}_{t - 1}}{1 - \bar{\alpha}_t}$",
        xy=(0.02, 0.85),
        xycoords="axes fraction",
    )
    for _ax in ax:
        _ax.set_xlim(0, num_timesteps)
        _ax.legend(loc="upper right")
        _ax.grid(axis="x")
        # Enforce consistent formatting for the y-axis, so we can create smooth GIFs.
        _ax.yaxis.set_major_locator(plt.LinearLocator(numticks=5))
        _ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.4f}")
    ax[0].set_ylim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[2].set_ylim(0, β_end)
    plt.suptitle(title)
    save_plot(PLOT_DIR, filename, create_date_dir=False, bbox_inches="tight")


#%%
if __name__ == "__main__":

    # Plot the coefficients of a beta schedule, using the default DDPM values
    num_timesteps = 1000
    betas: BetaSchedule = LinearBetaSchedule()
    model = SpiralDenoisingModel()
    device = torch.device("cpu")
    ddpm = DDPM(betas, model, device=device, num_timesteps=num_timesteps)

    # Plot the beta schedule, alphas, and the coefficients of the forward process q(x_t | x_0),
    # which defines x_t ~ N(sqrt(cumprod_alpha_t) * x_0, (1 - cumprod_alpha_t) * I).
    # We can see how the mean that defines the distribution of x_t goes to zero as the timestep increases,
    # while variance goes from to 1.
    plot_forward_coefficients(
        ddpm,
        title="Linear " + r"$\beta_t$" + " schedule, " + r"$t=$" + f"{num_timesteps}",
        filename="1_4_linear_betas.png",
        β_start=float(ddpm.betas[0] * 0.9),
        β_end=float(ddpm.betas[-1] * 1.1),
    )
    # When looking at the posterior process coefficients, i.e. q(x_t-1 | x_t, x_0) ~ N(mu_hat, beta_hat),
    # we see how the variance increases linearly over time,
    # while x_t is used to estimate x_t-1 except for the last few steps, when the weight of the estimate of x_0
    # becomes predominant, since the prediction becomes more confident.
    plot_posterior_coefficients(
        ddpm,
        title="Linear "
        + r"$\beta_t$"
        + " schedule, "
        + r"$t=$"
        + f"{num_timesteps}, "
        + r"$q(x_{t-1} | x_t, x_0) \sim \mathcal{N}(\bar{\mu}_t,\, \bar{\beta}_t)$",
        filename="1_4_posterior_coefficients_linear_beta.png",
        β_end=float(ddpm.betas[-1] * 1.1),
    )

    #%% Do the same thing, but now we use just 100 timesteps.
    # The shape of the curves is the same as using 1000 steps!
    num_timesteps = 100
    betas = LinearBetaSchedule(num_train_timesteps=1000)
    model = SpiralDenoisingModel()
    ddpm = DDPM(betas, model, device=device, num_timesteps=num_timesteps)

    # Plot again the beta schedule, alphas, and the coefficients of the forward process q(x_t | x_0).
    plot_forward_coefficients(
        ddpm,
        title="Linear " + r"$\beta_t$" + " schedule, " + r"$t=$" + f"{num_timesteps}",
        filename="1_4_linear_betas_100_steps.png",
        β_start=float(ddpm.betas[0] * 0.9),
        β_end=float(ddpm.betas[-1] * 1.1),
    )
    # Here the coefficients of bar_mu are not quite the same. Proportionally,
    # the estimate of x_0 becomes relevant earlier in the denoising process.
    plot_posterior_coefficients(
        ddpm,
        title="Linear "
        + r"$\beta_t$"
        + " schedule, "
        + r"$t=$"
        + f"{num_timesteps}, "
        + r"$q(x_{t-1} | x_t, x_0) \sim \mathcal{N}(\bar{\mu}_t,\, \bar{\beta}_t)$",
        filename="1_4_posterior_coefficients_linear_beta_100_steps.png",
        β_end=float(ddpm.betas[-1] * 1.1),
    )

    #%% LDM and Stable Diffusion use a different beta schedule that decreases noise in a smoother way.
    num_timesteps = 1000
    betas = ScaledLinearBetaSchedule()
    model = SpiralDenoisingModel()
    ddpm = DDPM(betas, model, device=device, num_timesteps=num_timesteps)

    # Plot again the beta schedule, alphas, and the coefficients of the forward process q(x_t | x_0).
    plot_forward_coefficients(
        ddpm,
        title="Scaled linear " + r"$\beta_t$" + " schedule, " + r"$t=$" + f"{num_timesteps}",
        filename="1_4_scaled_linear_betas.png",
        β_start=float(ddpm.betas[0] * 0.9),
        β_end=float(ddpm.betas[-1] * 1.1),
    )
    plot_posterior_coefficients(
        ddpm,
        title="Scaled linear "
        + r"$\beta_t$"
        + " schedule, "
        + r"$t=$"
        + f"{num_timesteps}, "
        + r"$q(x_{t-1} | x_t, x_0) \sim \mathcal{N}(\bar{\mu}_t,\, \bar{\beta}_t)$",
        filename="1_4_posterior_coefficients_scaled_linear_betas.png",
        β_end=float(ddpm.betas[-1] * 1.1),
    )

    #%% Why do we use those coefficients? Let's see what happens if we try different maximum betas.
    # Basically, they use the largest beta that results in unitary variance at step T.
    # Any smaller beta would still provide unitary variance, but the noise schedule would be less smooth.
    num_timesteps = 1000
    max_beta_range = np.geomspace(0.0001, 0.25, 50)
    with imageio.get_writer(PLOT_DIR / "1_4_scaled_linear_betas_range.gif", mode="I") as writer:  # Create a GIF!
        for i, b in tqdm(enumerate(max_beta_range)):
            betas = ScaledLinearBetaSchedule(num_train_timesteps=num_timesteps, β_end=b)
            model = SpiralDenoisingModel()
            ddpm = DDPM(betas, model, device=device, num_timesteps=num_timesteps)
            plot_name = f"1_4_scaled_linear_betas_range_{i}.png"
            plot_forward_coefficients(
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
            writer.append_data(image)  # type: ignore
            os.remove(PLOT_DIR / plot_name)
            plt.close()
    #%% In this case, we see how the posterior variance changes greatly,
    # from being irrelevant to becoming extremely large in the timesteps closest to T.
    # We also see how the coefficients of bar_mu are not really affected by the choice of beta!
    # But if β_end is too high and num_timesteps is too small, we observe numerical instability,
    # with bar_mu coefficients collapsing to zero instead of remaining approximately complementary.
    num_timesteps = 1000
    with imageio.get_writer(
        PLOT_DIR / "1_4_posterior_coefficients_scaled_linear_betas_range.gif",
        mode="I",
    ) as writer:  # Create a GIF!
        for i, b in tqdm(enumerate(max_beta_range)):
            betas = LinearBetaSchedule(num_train_timesteps=num_timesteps, β_end=b)
            model = SpiralDenoisingModel()
            ddpm = DDPM(betas, model, device=device, num_timesteps=num_timesteps)
            plot_name = f"1_4_posterior_coefficients_scaled_linear_betas_range_{i}.png"
            plot_posterior_coefficients(
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
            writer.append_data(image)  # type: ignore
            os.remove(PLOT_DIR / plot_name)
            plt.close()
