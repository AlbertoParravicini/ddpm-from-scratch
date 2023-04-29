import os
from dataclasses import dataclass
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

from ddpm_from_scratch.models.spiral_denoising_model import SinusoidalEncoding, SpiralDenoisingModel
from ddpm_from_scratch.samplers import DDPM
from ddpm_from_scratch.utils import COOL_GREEN, LinearBetaSchedule, make_spiral


@dataclass(frozen=True)
class Config:
    """
    A simple dataclass to store the basic settings for a training,
    so we can use different configurations for different devices,
    or quickly store different settings.
    """

    num_training_steps: int
    batch_size: int
    lr: float
    device: torch.device


# Let's use the GPU if available! If we use the GPU, train for longer,
# and with a bigger batch size for a more stable training. We should use a larger LR,
# but in this simple example it doesn't matter.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAINING_CONFIG: dict[str, Config] = {
    "cuda": Config(
        num_training_steps=80000,
        batch_size=128,
        lr=1e-3,
        device=torch.device("cuda"),
    ),
    "cpu": Config(
        num_training_steps=20000,
        batch_size=32,
        lr=1e-3,
        device=torch.device("cpu"),
    ),
}

PLOT_DIR = Path(__file__).parent.parent / "plots"

#%%
if __name__ == "__main__":
    # Setup.
    np.random.seed(seed=42)
    torch.manual_seed(42)
    reset_plot_style(xtick_major_pad=4, ytick_major_pad=4, border_width=1.5, label_pad=4)
    PLOT_DIR.mkdir(exist_ok=True, parents=True)

    # Create a Sinusoidal encoding, and plot encodings, to check that they have the expected shape.
    sinus = SinusoidalEncoding(32, 50)
    sns.heatmap(sinus.pe.T.numpy())  # type: ignore
    save_plot(PLOT_DIR, "1_3_sinusoidal_encodings.png", create_date_dir=False)

    # Obtain the configuration to use for training, depending on the device
    config: Config = TRAINING_CONFIG[DEVICE]
    device = config.device

    # Define the denoising model.
    model = SpiralDenoisingModel().to(device)

    # Define the optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Create the diffusion process. It is the same as `notebooks/1_2_gaussian_diffusion.py`, but rewritten in Pytorch.
    # Check out `ddpm_from_scratch/samplers/ddpm.py` for the Pytorch implementation.
    # We also wrap the beta schedule into a class.
    # We add a "device" parameter to support GPUs, and we use it if available
    # Note that we keep the same coefficients of beta as before.
    # They don't guarantee that we obtain Gaussian Noise at time T.
    # We stick to them for consistency with "Denoising diffusion probabilistic models from first principles".
    num_timesteps = 1000
    betas = LinearBetaSchedule(num_train_timesteps=1000, β_start=8e-6, β_end=9e-5)
    ddpm = DDPM(betas, model, device=device, num_timesteps=num_timesteps)

    # Create a spiral, and add noise using the new distribution.
    X = make_spiral(1000, normalize=True).to(device)

    #%% Train the model.
    num_training_steps = config.num_training_steps
    batch_size = config.batch_size
    losses = []
    # Replicate the spiral to obtain the desired batch size.
    X_train = X.repeat([batch_size, 1, 1])
    # Training loop
    progress_bar = tqdm(range(num_training_steps), desc="training")
    for i in progress_bar:
        # Zero gradients at every step
        optimizer.zero_grad()
        # Take a random batch of timesteps
        t = torch.randint(low=0, high=num_timesteps, size=(batch_size,), device=device)
        # Add some noise to the spiral (this is done without gradient!)
        with torch.no_grad():
            X_noisy, noise = ddpm.forward_sample(t, X_train)
        # Predict the noise
        predicted_noise = model(t, X_noisy)
        # Compute loss, as L2 of real and predicted noise
        loss = torch.mean((noise - predicted_noise) ** 2)
        # Backward step
        loss.backward()
        optimizer.step()
        losses += [loss.item()]
        progress_bar.set_postfix({"loss": loss.item()}, refresh=False)

    #%% Plot the loss function
    plt.figure(figsize=(6, 6))
    plt.plot(list(range(num_training_steps)), losses, lw=0.2)
    plt.plot(
        list(range(num_training_steps)),
        pd.Series(losses).rolling(100).mean(),
        lw=1,
        zorder=2,
    )
    plt.xlim(0, num_training_steps)
    save_plot(PLOT_DIR, "1_3_loss_function.png", create_date_dir=False)

    #%% Do inference, starting from a noisy spiral
    X = make_spiral(1000, normalize=True).to(device)  # Create the spiral
    # Create a DDPM that does inference in fewer steps, but still uses the trained model
    inference_steps = 50
    ddpm_inference = DDPM(betas, model, device=device, num_timesteps=inference_steps)
    X_noisy, _ = ddpm_inference.forward_sample(inference_steps - 1, X)  # Add noise
    with imageio.get_writer(PLOT_DIR / "1_3_inference.gif", mode="I") as writer:  # Create a GIF!
        steps = np.linspace(1, 0, inference_steps)
        for i, t in tqdm(enumerate(steps), desc="inference", total=len(steps)):
            # Get timestep, in the range [0, num_timesteps)
            timestep = min(int(t * inference_steps), inference_steps - 1)
            # Inference, predict the next step given the current one
            with torch.no_grad():
                X_noisy, X_0 = ddpm_inference.backward_sample(timestep, X_noisy, add_noise=bool(t != 0))
            # Plot the denoised spiral, every few steps
            if timestep % 5 == 0 or timestep == inference_steps - 1:
                fig, ax = plt.subplots(ncols=2, figsize=(6 * 2, 6))
                X_noisy_cpu = X_noisy.cpu().numpy()  # Always move to CPU for plotting
                X_0_cpu = X_0.cpu().numpy()
                ax[0].scatter(
                    X_noisy_cpu[:, 0],
                    X_noisy_cpu[:, 1],
                    color=PALETTE_1[-2],
                    alpha=0.8,
                    edgecolor="#2f2f2f",
                    lw=0.5,
                )
                ax[0].set_title("Noise becoming a spiral, " + r"$q(x_{t - 1} | x_t, \hat{x}_0), t=$" + f"{timestep}")
                ax[0].set_xlim((-1, 1))
                ax[0].set_ylim((-1, 1))
                ax[1].scatter(
                    X_0_cpu[:, 0],
                    X_0_cpu[:, 1],
                    color=COOL_GREEN,
                    alpha=0.8,
                    edgecolor="#2f2f2f",
                    lw=0.5,
                )
                ax[1].set_title("Prediction of " + r"$\hat{x}_0, t=$" + f"{timestep}")
                ax[1].set_xlim((-1, 1))
                ax[1].set_ylim((-1, 1))
                # Create a temporary file to assemble the GIF
                filename = f"1_3_inference_{timestep}.jpeg"
                save_plot(PLOT_DIR, filename, create_date_dir=False)
                image = imageio.imread(PLOT_DIR / filename)
                writer.append_data(image)  # type: ignore
                os.remove(PLOT_DIR / filename)
                plt.close()
