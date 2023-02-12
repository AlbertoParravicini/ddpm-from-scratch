import os
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from segretini_matplottini.utils.colors import PALETTE_1
from segretini_matplottini.utils.plot_utils import reset_plot_style, save_plot
from tqdm import tqdm

from ddpm_from_scratch.ddpm import DDPM
from ddpm_from_scratch.models.spiral_denoising_model import SinusoidalEncoding, SpiralDenoisingModel
from ddpm_from_scratch.utils import COOL_GREEN, linear_beta_schedule, make_spiral

PLOT_DIR = Path(__file__).parent.parent / "plots"


#%%
if __name__ == "__main__":
    # Setup.
    np.random.seed(seed=42)
    reset_plot_style(xtick_major_pad=4, ytick_major_pad=4, border_width=1.5, label_pad=4)
    PLOT_DIR.mkdir(exist_ok=True, parents=True)

    # Create a Sinusoidal encoding, and plot encodings, to check that they have the expected shape.
    sinus = SinusoidalEncoding(32, 50)
    sns.heatmap(sinus.pe.T.numpy())
    save_plot(PLOT_DIR, "1_3_sinusoidal_encodings.png", create_date_dir=False)

    # Define the denoising model.
    model = SpiralDenoisingModel()

    # Define the optimizer.
    optimizer = torch.optim.Adam(model.parameters())

    # Create the diffusion process. It is the same as `notebooks/1_2_gaussian_diffusion.py`, but rewritten in Pytorch.
    # Check out `ddpm_from_scratch/ddpm.py` for the Pytorch implementation.
    num_timesteps = 1000
    betas = linear_beta_schedule(num_timesteps, 8e-6, 9e-5)
    ddpm = DDPM(betas, model)

    # Create a spiral, and add noise using the new distribution.
    X = make_spiral(1000, normalize=True)

    #%% Train the model.
    num_training_steps = 20000
    batch_size = 1
    losses = []
    # Replicate the spiral to obtain the desired batch size.
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
        x0_pred, predicted_noise = ddpm.predict_x_0_and_noise(t, X_noisy)
        # Compute loss, as L2 of real and predicted noise
        loss = torch.mean((noise - predicted_noise) ** 2)
        # Backward step
        loss.backward()
        optimizer.step()
        losses += [loss.item()]
        progress_bar.set_postfix({"loss": loss.item()})

    #%% Plot the loss function
    plt.figure(figsize=(6, 6))
    plt.plot(list(range(num_training_steps)), losses, lw=0.2)
    plt.plot(list(range(num_training_steps)), pd.Series(losses).rolling(100).mean(), lw=1, zorder=2)
    plt.xlim(0, num_training_steps)
    save_plot(PLOT_DIR, "1_3_loss_function.png", create_date_dir=False)

    #%% Do inference, starting from a noisy spiral
    X = make_spiral(1000, normalize=True)  # Create the spiral
    # Create a DDPM that does inference in fewer steps, but still uses the trained model
    inference_steps = 200
    ddpm_inference = DDPM(linear_beta_schedule(inference_steps), model)
    X_noisy, _ = ddpm_inference.forward_sample(inference_steps - 1, X)  # Add noise
    with imageio.get_writer(PLOT_DIR / "1_3_inference.gif", mode="I") as writer:  # Create a GIF!
        steps = np.linspace(1, 0, inference_steps)
        for i, t in tqdm(enumerate(steps), desc="inference", total=len(steps)):
            # Get timestep, in the range [0, num_timesteps)
            timestep = min(int(t * inference_steps), inference_steps - 1)
            # Inference, predict the next step given the current one
            with torch.no_grad():
                X_noisy, X_0 = ddpm_inference.backward_sample(timestep, X_noisy, add_noise=t != 0)
            # Plot the denoised spiral, every few steps
            if timestep % 5 == 0 or timestep == inference_steps - 1:
                fig, ax = plt.subplots(ncols=2, figsize=(6 * 2, 6))
                ax[0].scatter(
                    X_noisy[:, 0], X_noisy[:, 1], color=PALETTE_1[-2], alpha=0.8, edgecolor="#2f2f2f", lw=0.5
                )
                ax[0].set_title("Noise becoming a spiral, " + r"$q(x_{t - 1} | x_t, \hat{x}_0), t=$" + f"{timestep}")
                ax[0].set_xlim((-1, 1))
                ax[0].set_ylim((-1, 1))
                ax[1].scatter(X_0[:, 0], X_0[:, 1], color=COOL_GREEN, alpha=0.8, edgecolor="#2f2f2f", lw=0.5)
                ax[1].set_title("Prediction of " + r"$\hat{x}_0, t=$" + f"{timestep}")
                ax[1].set_xlim((-1, 1))
                ax[1].set_ylim((-1, 1))
                # Create a temporary file to assemble the GIF
                filename = f"1_3_inference_{timestep}.jpeg"
                save_plot(PLOT_DIR, filename, create_date_dir=False)
                image = imageio.imread(PLOT_DIR / filename)
                writer.append_data(image)
                os.remove(PLOT_DIR / filename)
                plt.close()
