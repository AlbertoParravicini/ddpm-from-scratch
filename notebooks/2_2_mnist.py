import os
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from imageio.core.format import Format
from segretini_matplottini.utils.plot_utils import reset_plot_style, save_plot
from torch.utils.data import DataLoader
from torchtyping import TensorType
from torchvision.datasets.mnist import MNIST
from torchvision.utils import make_grid
from tqdm import tqdm

from ddpm_from_scratch.ddpm import DDPM
from ddpm_from_scratch.engines.mnist import (MnistInferenceGifCallback,
                                             get_one_element_per_digit,
                                             inference, load_mnist, train)
from ddpm_from_scratch.models.unet_simple_with_timestep import \
    UNetSimpleWithTimestep
from ddpm_from_scratch.utils import linear_beta_schedule

PLOT_DIR = Path(__file__).parent.parent / "plots"
DATA_DIR = Path(__file__).parent.parent / "data"

#%%
if __name__ == "__main__":
    # Setup. This is identical to `2_1_mnist.py`, but we train a simple UNet with timestep conditioning.
    np.random.seed(seed=42)
    reset_plot_style(xtick_major_pad=4, ytick_major_pad=4, border_width=1.5, label_pad=4)
    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    # Define the denoising model.
    model = UNetSimpleWithTimestep()
    print(model)

    # Define the optimizer.
    optimizer = torch.optim.Adam(model.parameters())

    # Create the diffusion process.
    num_timesteps = 1000
    betas = linear_beta_schedule(num_timesteps, 8e-6, 9e-5)
    ddpm = DDPM(num_timesteps, betas, model)

    # Load the MNIST dataset.
    mnist, dataloader = load_mnist(DATA_DIR, batch_size=4)

    #%% Train the model, in the same way as before.
    losses = train(dataloader=dataloader, sampler=ddpm, optimizer=optimizer, epochs=3)

    #%% Plot the loss function
    plt.figure(figsize=(6, 6))
    plt.plot(np.arange(len(losses)), losses, lw=0.2)
    plt.plot(np.arange(len(losses)), pd.Series(losses).rolling(100).mean(), lw=1, zorder=2)
    plt.xlim(0, len(losses))
    plt.ylim(0.4, 1.6)
    save_plot(PLOT_DIR, "2_2_loss_function.png", create_date_dir=False)

    #%% Do inference, denoising one sample digit for each category (0, 1, 2, ...)
    x = get_one_element_per_digit(mnist)
    # Add noise to the digits.
    x_noisy, _ = ddpm.forward_sample(num_timesteps - 1, x)
    # Do inference, and store results into the GIF, using the callback.
    inference(
        x=x_noisy,
        sampler=ddpm,
        callback=MnistInferenceGifCallback(filename=PLOT_DIR / "2_2_inference.gif"),
        call_callback_every_n_steps=50,
    )