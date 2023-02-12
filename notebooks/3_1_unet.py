from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from segretini_matplottini.utils.plot_utils import reset_plot_style, save_plot

from ddpm_from_scratch.ddpm import DDPM
from ddpm_from_scratch.engines.mnist import (
    MnistInferenceGifCallback,
    get_one_element_per_digit,
    inference,
    load_mnist,
    train_with_class_conditioning,
)
from ddpm_from_scratch.models.unet_conditioned import UNetConditioned
from ddpm_from_scratch.utils import cosine_beta_schedule

PLOT_DIR = Path(__file__).parent.parent / "plots"
DATA_DIR = Path(__file__).parent.parent / "data"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


#%%
if __name__ == "__main__":
    # Setup. This is identical to `2_1_mnist.py`, but we train a UNet with timestep conditioning.
    np.random.seed(seed=42)
    torch.manual_seed(seed=42)
    reset_plot_style(xtick_major_pad=4, ytick_major_pad=4, border_width=1.5, label_pad=4)
    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    # Define the denoising model. This time, use a full UNet with timestep conditioning,
    # residual blocks, and self-attention.
    model = UNetConditioned(num_classes=10).to(device)
    print(model)
    print(f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Define the optimizer.
    optimizer = torch.optim.RAdam(model.parameters())

    # Create the diffusion process.
    # This time, we use a cosine schedule.
    # We also decrease the number of steps to 100, following the original codebase.
    num_timesteps = 100
    betas = cosine_beta_schedule(num_timesteps).to(device)
    ddpm = DDPM(betas, model)

    # Load the MNIST dataset.
    mnist_train, dataloader_train, mnist_test, dataloader_test = load_mnist(DATA_DIR, batch_size=32)

    #%% Train the model, also using class conditioning
    losses = train_with_class_conditioning(
        dataloader=dataloader_train, sampler=ddpm, optimizer=optimizer, epochs=15, device=device
    )

    # Save the model
    torch.save(model.state_dict(), DATA_DIR / "unet_conditioned.pt")

    #%% Plot the loss function
    plt.figure(figsize=(6, 6))
    plt.plot(np.arange(len(losses)), losses, lw=0.2)
    plt.plot(np.arange(len(losses)), pd.Series(losses).rolling(100).mean(), lw=1, zorder=2)
    plt.xlim(0, len(losses))
    plt.ylim(0.0, 1.6)
    save_plot(PLOT_DIR, "3_1_loss_function.png", create_date_dir=False)

    #%% Do inference, denoising one sample digit for each category (0, 1, 2, ...)
    x = get_one_element_per_digit(mnist_test).to(device)
    # Add noise to the digits, with some specified strengths.
    # The model can denoise very well digits with a small amount of noise,
    # but it will have problems denoising digits with a lot of noise.
    # The extreme case, where we start with Gaussian noise, is pure image generation.
    noise_strengths = [0.25, 0.5, 0.75, 1]
    for noise_strength in noise_strengths:
        x_noisy, _ = ddpm.forward_sample(int((num_timesteps - 1) * noise_strength), x)
        # Do inference, and store results into the GIF, using the callback.
        x_denoised = inference(
            x=x_noisy,
            sampler=ddpm,
            conditioning=torch.arange(0, 10, device=device),
            callback=MnistInferenceGifCallback(filename=PLOT_DIR / f"3_1_inference_{noise_strength:.2f}.gif"),
            call_callback_every_n_steps=50,
            initial_step_percentage=noise_strength,
        )
        # Compute error, as L2 norm.
        l2 = torch.nn.functional.mse_loss(x_denoised, x, reduction="mean").item()
        print(f"L2 norm after denoising, noise strength {noise_strength:.2f}: {l2:.6f}")
