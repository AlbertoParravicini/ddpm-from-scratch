from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import torch
from segretini_matplottini.utils.colors import PALETTE_1
from segretini_matplottini.utils.plot_utils import reset_plot_style, save_plot
import imageio.v2 as imageio
from ddpm_from_scratch.ddpm import DDPM
from ddpm_from_scratch.models.spiral_denoising_model import (
    SinusoidalEncoding, SpiralDenoisingModel)
from ddpm_from_scratch.utils import linear_beta_schedule, make_spiral, COOL_GREEN
from tqdm import tqdm

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
    ddpm = DDPM(num_timesteps, betas, model)

    # Create a spiral, and add noise using the new distribution
    X = make_spiral(1000, normalize=True)

    #%% Train the model
    num_training_steps = 10000
    batch_size = 8
    losses = []
    # Replicate the spiral to obtain the desired batch size
    X_train = X.repeat([batch_size, 1, 1])
    # Training loop
    progress_bar = tqdm(range(num_training_steps), desc="training")
    for i in progress_bar:
        # Zero gradients at every step
        optimizer.zero_grad()
        # Take a random timestep
        t = np.random.randint(num_timesteps, size=batch_size)
        # Add some noise to the spiral (this is done without gradient!)
        with torch.no_grad():
            X_noisy, noise = ddpm.forward_sample(t, X_train)
        # Predict the noise
        _, predicted_noise = ddpm.predict_x_0_and_noise(t, X_noisy)
        # Compute loss, as L2 of real and predicted noise
        loss = torch.mean((noise - predicted_noise) ** 2)
        # Backward step
        loss.backward()
        optimizer.step()
        losses += [loss.item()]
        progress_bar.set_postfix({"epoch": i, "loss": loss.item()})

    #%% Plot the loss function
    plt.figure(figsize=(6, 6))
    plt.plot(list(range(num_training_steps)), losses, lw=0.2)
    plt.plot(list(range(num_training_steps)), pd.Series(losses).rolling(100).mean(), lw=1, zorder=2)
    plt.xlim(0, num_training_steps)
    save_plot(PLOT_DIR, "1_3_loss_function.png", create_date_dir=False)

    #%% Do inference, starting from a noisy spiral
    X = make_spiral(1000, normalize=True)
    X_noisy, _ = ddpm.forward_sample(num_timesteps - 1, X)
    
    X_curr = X_noisy
    with imageio.get_writer(PLOT_DIR / "1_3_inference.gif", mode="I") as writer:  # Create a GIF!
        for i, t in tqdm(enumerate(np.linspace(1, 0, num_timesteps)), desc="inference"):
            # Get timestep, in the range [0, num_timesteps)
            timestep = min(int(t * num_timesteps), num_timesteps - 1)
            # Inference
            with torch.no_grad():
                X_curr, X_0 = ddpm.backward_sample(timestep, X_curr, add_noise=False)
            # Plot the denoised spiral, every few steps
            if timestep % (num_timesteps // 20) == 0 or timestep == num_timesteps - 1:
                fig, ax = plt.subplots(ncols=2, figsize=(6 * 2, 6))
                ax[0].scatter(X_curr[:, 0], X_curr[:, 1], color=PALETTE_1[-2], alpha=0.8, edgecolor="#2f2f2f", lw=0.5)
                ax[0].set_title("Noise becoming a spiral, " + r"$q(x_{t - 1} | x_t, \hat{x}_0), t=$" + f"{timestep}")
                ax[0].set_xlim((-1, 1))
                ax[0].set_ylim((-1, 1))
                ax[1].scatter(X_0[:, 0], X_0[:, 1], color=COOL_GREEN, alpha=0.8, edgecolor="#2f2f2f", lw=0.5)
                ax[1].set_title("Prediction of " + r"$\hat{x}_0, t=$" + f"{timestep}")
                ax[1].set_xlim((-1, 1))
                ax[1].set_ylim((-1, 1))
                filename = f"1_3_inference_{timestep}.jpeg"
                save_plot(PLOT_DIR, filename, create_date_dir=False)
                image = imageio.imread(PLOT_DIR / filename)
                writer.append_data(image)
                os.remove(PLOT_DIR / filename)
                plt.close()
