import os
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from jaxtyping import Float
from segretini_matplottini.utils.plot_utils import reset_plot_style, save_plot
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.utils import make_grid
from tqdm import tqdm

from ddpm_from_scratch.models.unet_simple import UNetSimple
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
        num_training_epochs=16,
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


def get_one_element_per_digit(
    mnist: MNIST,
) -> Float[torch.Tensor, "10 1 28 28"]:
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
    torch.manual_seed(42)
    reset_plot_style(xtick_major_pad=4, ytick_major_pad=4, border_width=1.5, label_pad=4)
    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    # Let's obtain the right configuration for the training
    config = TRAINING_CONFIG[DEVICE]
    device = config.device

    # Define the denoising model.
    # Here we are using an extremely simple UNet. It doesn't even use time-conditioning!
    # We don't expect it to work well, but we'll make it better soon.
    model = UNetSimple().to(device)
    print(model)
    print(f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

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
    batch_size = config.batch_size
    dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True, num_workers=1)

    #%% Train the model.
    # This training is identical to denoising the spiral,
    # but we do multiple epochs over the full MNIST dataset.
    # This is a very scrappy training: we don't have a validation set,
    # and L2 norm might not be fully sufficient
    # to judge our ability to generate digits from scratch.
    # We'll think about these problems later!
    num_training_epochs = config.num_training_epochs
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
            t = torch.randint(low=0, high=num_timesteps, size=(x.shape[0],), device=device)
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

    # Save the model
    torch.save(model.state_dict(), DATA_DIR / "2_1_unet.pt")

    #%% Plot the loss function
    plt.figure(figsize=(6, 6))
    plt.plot(np.arange(len(losses)), losses, lw=0.2)
    plt.plot(
        np.arange(len(losses)),
        pd.Series(losses).rolling(100).mean(),
        lw=1,
        zorder=2,
    )
    plt.xlim(0, len(losses))
    save_plot(PLOT_DIR, "2_1_loss_function.png", create_date_dir=False)

    #%% Do inference, denoising one sample digit for each category (0, 1, 2, ...).
    # Here we do 1000 steps of DDPM for inference, which is highly inefficient!
    # We could do fewer steps and still get an acceptable quality,
    # even more so if we use a different sampler.
    x = get_one_element_per_digit(mnist).to(device)
    num_timesteps = 1000
    ddpm = DDPM(betas, model, device=device, num_timesteps=num_timesteps)
    x_noisy, _ = ddpm.forward_sample(num_timesteps - 1, x)  # Add noise
    with imageio.get_writer(PLOT_DIR / "2_1_inference.gif", mode="I") as writer:  # Create a GIF!
        # We must be in eval mode
        ddpm.model.eval()
        steps = np.linspace(1, 0, num_timesteps)
        for i, t in tqdm(enumerate(steps), desc="inference", total=len(steps)):
            # Get timestep, in the range [0, num_timesteps)
            timestep = min(int(t * num_timesteps), num_timesteps - 1)
            # Inference, predict the next step given the current one
            with torch.no_grad():
                x_noisy, _ = ddpm.backward_sample(
                    timestep,
                    x_noisy,
                    add_noise=bool(t != 0),
                    clip_predicted_x_0=False,
                )
            # Plot the denoised digit, every few steps
            if timestep % (num_timesteps // 20) == 0 or timestep == num_timesteps - 1:
                # Plot multiple digits as a grid, each in a separate column
                grid = make_grid(x_noisy, padding=False, nrow=len(x_noisy), normalize=True)
                # Create a temporary file to assemble the GIF
                filename = f"2_1_inference_{timestep}.jpeg"
                F.to_pil_image(grid).save(PLOT_DIR / filename)
                image = imageio.imread(PLOT_DIR / filename)
                writer.append_data(image)  # type: ignore
                os.remove(PLOT_DIR / filename)
