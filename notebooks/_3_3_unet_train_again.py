from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from segretini_matplottini.utils.colors import PALETTE_OG
from segretini_matplottini.utils.plot_utils import reset_plot_style, save_plot

from ddpm_from_scratch.engines.mnist import (MnistInferenceGifCallback,
                                             get_one_element_per_digit,
                                             inference, load_mnist,
                                             train_with_class_conditioning)
from ddpm_from_scratch.models.unet_conditioned_v2 import UNetConditioned
from ddpm_from_scratch.samplers.ddpm import DDPM
from ddpm_from_scratch.utils import ScaledLinearBetaSchedule

PLOT_DIR = Path(__file__).parent.parent / "plots"
DATA_DIR = Path(__file__).parent.parent / "data"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


###############################################################################
# * Increased batch to 512, to have 75%+ utilization on A100
# * Moved from RAdam to AdamW, since RAdam had numerical issues and spiked at around 17 epochs
# * Adding StepLR after 48 epochs, it reduces the loss from 0.14 to 0.13. The second reduction is not noticeable
# * Changed steps from 100 to 1000, it reduces loss by 0.01
# * Changed LR to 3e-3
#   * Current loss: 0.0972, L2 at strength 1: 0.633
# * Changed LR scheduler to CosineAnnealingWarmRestarts, it broke the training (giga loss)
# * Back to StepLR, 64 steps, 0.1 gamma
# * Modify UpDownBlock to have ResNet followed by Attention, +100K params -> loss goes down to ~0.055 after 192 epochs
#   * Loss stops going down from 0.06 after ~60-80 steps, but it does not overfit either.
#   * L2 at strength 1: 0.621
# * scaled_linear_beta_schedule instead of cosine_beta_schedule -> loss goes down to 0.04
#   * L2 at strength 1: 0.629 -> Why is it higher?
# * Add 24 as hidden dimension, go wider. 1.1M params, let's not get bigger than this
#   * At 0.001, loss spikes up, at 0.0005, it goes at around 0.38. Not great. We have to go deeper instead
#   * L2 at strength 1: 0.55
###############################################################################


#%%
if __name__ == "__main__":
    # Setup. This is identical to `2_1_mnist.py`, but we train a UNet with timestep conditioning.
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    np.random.seed(seed=42)
    torch.manual_seed(seed=42)
    reset_plot_style(xtick_major_pad=4, ytick_major_pad=4, border_width=1.5, label_pad=4)
    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    # Define the denoising model. This time, use a full UNet with timestep conditioning,
    # residual blocks, and self-attention.
    model = UNetConditioned(num_classes=10, hidden_channels=24, channel_multipliers=(1, 2, 3)).to(device)
    print(model)
    print(f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Define the optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.5e-3)
    # Create a LR scheduler that reduces by 10x the LR after 10 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=64, gamma=0.1)

    # Create the diffusion process.
    # This time, we use a cosine schedule.
    # We also decrease the number of steps to 100, following the original codebase.
    num_timesteps = 1000
    betas = ScaledLinearBetaSchedule(num_train_timesteps=num_timesteps)
    ddpm = DDPM(betas, model, num_timesteps=num_timesteps, device=device)

    # Load the MNIST dataset.
    mnist_train, dataloader_train, mnist_test, dataloader_test = load_mnist(DATA_DIR, batch_size=512)

    #%% Train the model, also using class conditioning
    epochs = 192
    validation_every_n_epochs = 8
    losses, val_losses = train_with_class_conditioning(
        dataloader=dataloader_train,
        sampler=ddpm,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        scheduler=scheduler,
        classifier_free_probability=0.1,
        validation_dataloader=dataloader_test,
        validation_every_n_epochs=validation_every_n_epochs,
        seed=42,
    )

    # Save the model
    torch.save(model.state_dict(), DATA_DIR / "unet_conditioned.pt")

    #%% Plot the loss functions
    plt.figure(figsize=(6, 6))
    plt.grid(axis="y", color="0.9", zorder=0, lw=0.4)
    plt.plot(np.arange(len(losses)), losses, lw=0.6, color=PALETTE_OG[0])
    plt.plot(
        np.arange(len(losses)),
        pd.Series(losses).rolling(100).mean(),
        lw=1.2,
        zorder=2,
        color=PALETTE_OG[0],
        ls=":",
        label="train",
    )
    plt.plot(
        (validation_every_n_epochs * len(losses) // epochs) * np.arange(1, len(val_losses) + 1),
        val_losses,
        lw=1.2,
        color=PALETTE_OG[1],
        label="val",
    )
    plt.xlim(0, len(losses))
    plt.ylim(0.0, 0.4)
    plt.legend(loc="upper right", frameon=True)
    save_plot(PLOT_DIR, f"3_3_loss_function_{timestamp}.png", create_date_dir=False)

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
            callback=MnistInferenceGifCallback(
                filename=PLOT_DIR / f"3_3_inference_{noise_strength:.2f}_{timestamp}.gif"
            ),
            call_callback_every_n_steps=5,
            initial_step_percentage=noise_strength,
            classifier_free_scale=7.5,
        )
        # Compute error, as L2 norm.
        l2 = torch.nn.functional.mse_loss(x_denoised, x, reduction="mean").item()
        print(f"L2 norm after denoising, noise strength {noise_strength:.2f}: {l2:.6f}")
