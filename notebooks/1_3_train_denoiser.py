from pathlib import Path
from telnetlib import GA

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from segretini_matplottini.utils.colors import PALETTE_1
from segretini_matplottini.utils.plot_utils import reset_plot_style, save_plot

from ddpm_from_scratch.gaussian_diffusion import GaussianDiffusion
from ddpm_from_scratch.models.spiral_denoising_model import (
    SinusoidalEncoding, SpiralDenoisingModel)
from ddpm_from_scratch.utils import linear_beta_schedule, make_spiral

PLOT_DIR = Path(__file__).parent.parent / "plots"


#%%
if __name__ == "__main__":
    # Setup
    np.random.seed(seed=42)
    reset_plot_style(xtick_major_pad=4, ytick_major_pad=4, border_width=1.5, label_pad=4)
    PLOT_DIR.mkdir(exist_ok=True, parents=True)

    # Create a Sinusoidal encoding, and plot encodings, to check that they have the expected shape
    sinus = SinusoidalEncoding(32, 50)
    sns.heatmap(sinus.pe.T.numpy())
    save_plot(PLOT_DIR, "1_sinusoidal_encodings.png", create_date_dir=False)

    # Define the model
    model = SpiralDenoisingModel()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Create the diffusion process. It is the same as `notebooks/1_2_gaussian_diffusion.py`, but rewritten in Pytorch.
    # Check out `ddpm_from_scratch/gaussian_diffusion.py` for the Pytorch implementation.
    num_timesteps = 1000
    betas = linear_beta_schedule(num_timesteps, 8e-6, 9e-5)
    gaussian_diffusion = GaussianDiffusion(num_timesteps, betas, model)

    # Create a spiral, and add noise using the new distribution
    X = make_spiral(1000, normalize=1)

    # Train the model
    num_training_steps = 10000
    batch_size = 4
    losses = []
    # Replicate the spiral to obtain the desired batch size
    X_train = X.repeat([batch_size, 1, 1])
    # Training loop
    for i in range(num_training_steps):
        # Zero gradients at every step
        optimizer.zero_grad()
        # Take a random timestep
        t = np.random.randint(num_timesteps, size=batch_size)
        # Add some noise to the spiral (this is done without gradient!)
        with torch.no_grad():
            X_noisy, noise = gaussian_diffusion.forward_sample(t, X_train)
        # Predict the noise
        _, predicted_noise = gaussian_diffusion._predict_x_0_and_noise(t, X_noisy)
        # Compute loss, as L2 of real and predicted noise
        loss = torch.mean((noise - predicted_noise) ** 2)
        # Backward step
        loss.backward()
        optimizer.step()
        print(f"epoch: {i}, loss: {loss.item():.4f}")
        losses += [loss.item()]

    #%% Plot the loss function
    plt.figure(figsize=(6, 6))
    plt.plot(list(range(num_training_steps)), losses, lw=0.2)
    plt.plot(list(range(num_training_steps)), pd.Series(losses).rolling(100).mean(), lw=1, zorder=2)
    plt.xlim(0, num_training_steps)
    save_plot(PLOT_DIR, "1_3_loss_function.png", create_date_dir=False)
