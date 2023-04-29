import os
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import seaborn as sns
from segretini_matplottini.utils.colors import PALETTE_1
from segretini_matplottini.utils.plot_utils import reset_plot_style, save_plot

PLOT_DIR = Path(__file__).parent.parent / "plots"


def univariate_gaussian_sample(n: int, μ: float = 0, σ: float = 1) -> np.ndarray:
    return μ + σ * np.random.randn(n)


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


#%%
if __name__ == "__main__":

    # Setup
    np.random.seed(seed=42)
    reset_plot_style(xtick_major_pad=4, ytick_major_pad=4, border_width=1.5, label_pad=4)
    PLOT_DIR.mkdir(exist_ok=True, parents=True)

    # Sample data according to a univariate Gaussian
    # with chosen mean, variance
    x = univariate_gaussian_sample(n=1000, μ=1.5, σ=0.7)

    # Plot the histogram of the Gaussian samples.
    plt.figure(figsize=(6, 6))
    ax = sns.histplot(x, stat="density", kde=True, label="Samples", color=PALETTE_1[0])
    # Plot a real Gaussian distribution, using the theoretical PDF
    x_min, x_max = ax.get_xlim()
    x_pdf = np.linspace(x_min, x_max, 100)
    y_pdf = st.norm.pdf(x_pdf, loc=1.5, scale=0.7)
    ax.plot(x_pdf, y_pdf, PALETTE_1[-2], lw=2, label="Theoretical")
    # Add a legend to the plot
    ax.legend()
    plt.title("Gaussian samples vs. real distribution")
    # Store the result
    save_plot(PLOT_DIR, "1_1_gaussian_dist.png", create_date_dir=False)

    #%% Draw a spiral
    x, y = make_spiral(1000, normalize=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, color=PALETTE_1[-2], alpha=0.8, edgecolor="#2f2f2f", lw=0.5)
    plt.title("A spiral")
    save_plot(PLOT_DIR, "1_1_spiral.png", create_date_dir=False)

    # %% Generate beta schedule.
    # Note to the reader: the coefficient of β_end is the same as in
    # "Denoising diffusion probabilistic models from first principles".
    # However, this coefficient does not ensure that the q(x_T) has unitary variance and zero mean.
    # The author don't say why: I imagine it's because we are doing a denoising task, not a generation task.
    # It is not reasonable to fully denoise an image if we start from pure Gaussian noise.
    # So, we ensure that the maximum amount of noise being added is still something we can recover from.
    # In later notebooks we switch to a generative task, and use coefficients
    # that guarantee unitary variance and zero mean.
    num_timesteps = 40
    βs = linear_beta_schedule(num_timesteps, 8e-6, 9e-5)

    # The forward process, which adds noise to the input, is defined as:
    # q(x_t | x_(t - 1)) = N(sqrt(1 - β_t) * x_(t - 1), sqrt(β_t) * I), equation (1)

    # Add noise to the spiral, and store intermediate results
    X = np.array([x, y]).T  # Turn the spiral into a matrix, this is timestep 0
    Xt = [X]  # Save intermediate results
    with imageio.get_writer(PLOT_DIR / "1_1_spiral_noise.gif", mode="I") as writer:  # Create a GIF!
        for t in range(num_timesteps):
            # Forward process, equation (1)
            μ = X * np.sqrt(1 - βs[t])
            noise = np.random.randn(X.shape[0], 2)
            σ = np.sqrt(βs[t])
            X = μ + σ * noise
            Xt += [X]
            # Plot the spiral
            plt.figure(figsize=(6, 6))
            plt.scatter(X[:, 0], X[:, 1], color=PALETTE_1[-2], alpha=0.8, edgecolor="#2f2f2f", lw=0.5)
            plt.title(f"A spiral becoming noise, step {t}")
            plt.xlim((-1, 1))  # enforce axes limits as we add noise
            plt.ylim((-1, 1))
            # Store a temporary image to create the GIF, then delete it
            filename = f"1_1_spiral_noise_{t}.png"
            save_plot(PLOT_DIR, filename, create_date_dir=False)
            image = imageio.imread(PLOT_DIR / filename)
            writer.append_data(image)  # type: ignore
            os.remove(PLOT_DIR / filename)
            plt.close()
