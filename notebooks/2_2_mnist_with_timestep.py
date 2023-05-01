from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from segretini_matplottini.utils.plot_utils import reset_plot_style, save_plot

from ddpm_from_scratch.engines.mnist import (
    MnistInferenceGifCallback,
    get_one_element_per_digit,
    inference,
    load_mnist,
    train,
)
from ddpm_from_scratch.models import UNetWithTimestep
from ddpm_from_scratch.samplers import DDPM
from ddpm_from_scratch.utils import LinearBetaSchedule


@dataclass(frozen=True)
class Config:
    """
    A simple dataclass to store the basic settings for a training,
    so we can use different configurations for different devices,
    or quickly store different settings.
    """

    num_training_epochs: int
    batch_size: int
    lr: float
    device: torch.device


# Let's use the GPU if available! If we use the GPU, train for longer,
# and with a bigger batch size for a more stable training. We should use a larger LR,
# but in this simple example it doesn't matter.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAINING_CONFIG: dict[str, Config] = {
    "cuda": Config(
        num_training_epochs=96,
        batch_size=128,
        lr=1e-3,
        device=torch.device("cuda"),
    ),
    "cpu": Config(
        num_training_epochs=3,
        batch_size=32,
        lr=1e-3,
        device=torch.device("cpu"),
    ),
}

PLOT_DIR = Path(__file__).parent.parent / "plots"
DATA_DIR = Path(__file__).parent.parent / "data"

#%%
if __name__ == "__main__":
    # Setup. This is identical to `2_1_mnist.py`, but we train a simple UNet with timestep conditioning.
    np.random.seed(seed=42)
    torch.manual_seed(42)
    reset_plot_style(xtick_major_pad=4, ytick_major_pad=4, border_width=1.5, label_pad=4)
    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    # Let's obtain the right configuration for the training
    config = TRAINING_CONFIG[DEVICE]
    device = config.device

    # Define the denoising model.
    model = UNetWithTimestep().to(device)
    print(model)
    print(f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Define the optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Create the diffusion process.
    num_timesteps = 1000
    betas = LinearBetaSchedule(num_train_timesteps=num_timesteps)
    ddpm = DDPM(betas, model, device=device, num_timesteps=num_timesteps)

    # Load the MNIST dataset. Split between training and test set.
    mnist_train, dataloader_train, mnist_test, dataloader_test = load_mnist(DATA_DIR, batch_size=config.batch_size)

    #%% Train the model.
    # We wrap the training code into a function for simplicity, but the code is unchanged from `2_1_mnist.py`.
    losses = train(
        dataloader=dataloader_train,
        sampler=ddpm,
        optimizer=optimizer,
        epochs=config.num_training_epochs,
        device=device,
    )

    # Save the model
    torch.save(model.state_dict(), DATA_DIR / "2_2_unet.pt")

    #%% Plot the loss function
    plt.figure(figsize=(6, 6))
    plt.grid(axis="y", color="0.9", zorder=0, lw=0.4)
    plt.plot(np.arange(len(losses)), losses, lw=0.2)
    plt.plot(
        np.arange(len(losses)),
        pd.Series(losses).rolling(100).mean(),
        lw=1,
        zorder=2,
    )
    plt.xlim(0, len(losses))
    plt.ylim(0, 0.2)
    save_plot(PLOT_DIR, "2_2_loss_function.png", create_date_dir=False)

    #%% Do inference, denoising one sample digit for each category (0, 1, 2, ...).
    # We wrap the inference code into a function for simplicity, but the code is unchanged from `2_1_mnist.py`.
    # We also add a callback to save the intermediate results into a GIF, rather than manually handling the plotting.
    x = get_one_element_per_digit(mnist_test).to(device)
    # Add noise to the digits.
    x_noisy, _ = ddpm.forward_sample(num_timesteps - 1, x)
    # Do inference, and store results into the GIF, using the callback.
    x_denoised = inference(
        x=x_noisy,
        sampler=ddpm,
        callback=MnistInferenceGifCallback(filename=PLOT_DIR / "2_2_inference.gif"),
        call_callback_every_n_steps=50,
    )
    # Compute error, as L2 norm.
    l2 = torch.nn.functional.mse_loss(x_denoised, x, reduction="mean").item()
    print(f"L2 norm after denoising: {l2:.6f}")
