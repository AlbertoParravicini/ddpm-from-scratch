import os
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from segretini_matplottini.utils.plot_utils import reset_plot_style, save_plot
from torch.utils.data import DataLoader
from torchtyping import TensorType
from torchvision.datasets.mnist import MNIST
from torchvision.utils import make_grid
from tqdm import tqdm

from ddpm_from_scratch.samplers.ddpm import DDPM
from ddpm_from_scratch.models.unet_simple import UNetSimple
from ddpm_from_scratch.utils import LinearBetaSchedule

PLOT_DIR = Path(__file__).parent.parent / "plots"
DATA_DIR = Path(__file__).parent.parent / "data"


def get_one_element_per_digit(mnist) -> TensorType[10, 1, 28, 28]:
    """
    Get a single sample digit for each target category in MNIST, and return the result as a tensor.
    The output is a `[10, 1, 28, 28]` tensor, with digits `[0, 1, 2, ..., 9]`
    """
    targets = sorted(set(mnist.targets.numpy()))
    digits = []
    for t in targets:
        digits += [mnist[np.argwhere(mnist.targets == t)[0, 0]][0]]
    return torch.stack(digits)


#%%
if __name__ == "__main__":
    # Setup
    np.random.seed(seed=42)
    reset_plot_style(xtick_major_pad=4, ytick_major_pad=4, border_width=1.5, label_pad=4)
    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    # Let's use a GPU if available!
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define the denoising model
    model = UNetSimple()
    print(model)
    print(f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # Create the diffusion process. Here we use "realistic" values for the beta schedule,
    # so that we obtain pure Gaussian noise at timestep T=1000.
    num_timesteps = 1000
    betas = LinearBetaSchedule(num_train_timesteps=num_timesteps)
    ddpm = DDPM(betas, model, device=device, num_timesteps=num_timesteps)

    # Load the MNIST dataset. We'll use just the training split for now.
    mnist = MNIST(
        root=DATA_DIR,
        download=True,
        train=True,
        transform=T.Compose(
            [
                T.ToTensor(),
                T.Normalize(0.5, 0.5),
            ]
        ),
    )
    # Small batch size meant for CPUs. If you have a GPU, feel free to scale it to 128 or more.
    # You should increase the learning rate when using a larger batch size,
    # but in this example it doesn't matter much.
    batch_size = 128 if device == "cuda" else 32
    dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True, num_workers=1)

    #%% Train the model.
    # This training is identical to denoising the spiral,
    # but we do multiple epochs over the full MNIST dataset.
    num_training_epochs = 16 if device == "cuda" else 3
    losses = []
    progress_bar_epoch = tqdm(range(num_training_epochs), desc="training")
    for e in progress_bar_epoch:
        progress_bar_step = tqdm(dataloader, desc=f"epoch {e}")
        # Iterate over the dataset, but ignore the class for now
        for i, (x, _) in enumerate(progress_bar_step):
            # Move the data to the GPU if available
            x = x.to(device)
            # Zero gradients at every step
            optimizer.zero_grad()
            # Take a random batch of timesteps
            t = torch.randint(low=0, high=num_timesteps, size=(x.shape[0],))
            # Add some noise to the data
            with torch.no_grad():
                x_noisy, noise = ddpm.forward_sample(t, x)
            # Predict the noise
            predicted_noise = model(t, x_noisy)
            # Compute loss, as L2 of real and predicted noise
            loss = torch.mean((noise - predicted_noise) ** 2)
            # Backward step
            loss.backward()
            optimizer.step()
            losses += [loss.item()]
            if i % 100 == 0:
                progress_bar_step.set_postfix({"loss": loss.item()})
        progress_bar_epoch.set_postfix({"loss": loss.item()})

    #%% Plot the loss function
    plt.figure(figsize=(6, 6))
    plt.plot(np.arange(len(losses)), losses, lw=0.2)
    plt.plot(np.arange(len(losses)), pd.Series(losses).rolling(100).mean(), lw=1, zorder=2)
    plt.xlim(0, len(losses))
    save_plot(PLOT_DIR, "2_1_loss_function.png", create_date_dir=False)

    #%% Do inference, denoising one sample digit for each category (0, 1, 2, ...)
    x = get_one_element_per_digit(mnist).to(device)
    x_noisy, _ = ddpm.forward_sample(num_timesteps - 1, x)  # Add noise
    with imageio.get_writer(PLOT_DIR / "2_1_inference.gif", mode="I") as writer:  # Create a GIF!
        steps = np.linspace(1, 0, num_timesteps)
        for i, t in tqdm(enumerate(steps), desc="inference", total=len(steps)):
            # Get timestep, in the range [0, num_timesteps)
            timestep = min(int(t * num_timesteps), num_timesteps - 1)
            # Inference, predict the next step given the current one
            with torch.no_grad():
                x_noisy, _ = ddpm.backward_sample(timestep, x_noisy, add_noise=t != 0)
            # Plot the denoised digit, every few steps
            if timestep % (num_timesteps // 20) == 0 or timestep == num_timesteps - 1:
                # Plot multiple digits as a grid, each in a separate column
                grid = make_grid(x_noisy, padding=False, nrow=len(x_noisy))
                # Create a temporary file to assemble the GIF
                filename = f"2_1_inference_{timestep}.jpeg"
                T.ToPILImage()(grid).save(PLOT_DIR / filename)
                image = imageio.imread(PLOT_DIR / filename)
                writer.append_data(image)
                os.remove(PLOT_DIR / filename)
