from dataclasses import dataclass
from pathlib import Path
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from segretini_matplottini.utils.plot_utils import reset_plot_style, save_plot
from segretini_matplottini.utils.colors import PALETTE_OG

from ddpm_from_scratch.engines.mnist import (
    MnistInferenceGifCallback,
    get_one_element_per_digit,
    inference,
    load_mnist,
    train_with_class_conditioning,
    fid,
    generate_digits,
)
from ddpm_from_scratch.models import UNetWithConditioning, LeNet5
from ddpm_from_scratch.samplers import DDPM, DDIM
from ddpm_from_scratch.utils import ScaledLinearBetaSchedule


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
    validation_every_n_epochs: int = 8


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
        validation_every_n_epochs=8,
    ),
    "cpu": Config(
        num_training_epochs=3,
        batch_size=32,
        lr=1e-3,
        device=torch.device("cpu"),
        validation_every_n_epochs=1,
    ),
}

PLOT_DIR = Path(__file__).parent.parent / "plots"
DATA_DIR = Path(__file__).parent.parent / "data"

#%%
if __name__ == "__main__":
    # Setup.
    np.random.seed(seed=42)
    torch.manual_seed(42)
    reset_plot_style(xtick_major_pad=4, ytick_major_pad=4, border_width=1.5, label_pad=4)
    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    # Let's obtain the right configuration for the training
    config = TRAINING_CONFIG[DEVICE]
    device = config.device

    # Define the denoising model.
    model = UNetWithConditioning(classes=10).to(device)
    print(model)
    print(f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Define the optimizer. This time, replace Adam for a better optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    # Create a LR scheduler that reduces by 10x the LR after 40% of the epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config.num_training_epochs * 0.4), gamma=0.1)

    # Create the diffusion process. Here, swap the linear schedule with a scaled linear schedule,
    # which introduces noise in a smoother way and should provide better results.
    num_timesteps = 1000
    betas = ScaledLinearBetaSchedule(num_train_timesteps=num_timesteps)
    ddpm = DDPM(betas, model, device=device, num_timesteps=num_timesteps)

    # Load the MNIST dataset.
    mnist_train, dataloader_train, mnist_test, dataloader_test = load_mnist(
        DATA_DIR, batch_size=config.batch_size, seed=42
    )

    #%% Train the model.
    losses, val_losses = train_with_class_conditioning(
        dataloader=dataloader_train,
        sampler=ddpm,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=config.num_training_epochs,
        device=device,
        classifier_free_probability=0.1,
        validation_dataloader=dataloader_test,
        validation_every_n_epochs=config.validation_every_n_epochs,
        seed=42,
    )

    # Save the model
    torch.save(model.state_dict(), DATA_DIR / "3_3_unet.pt")

    #%% Plot the training loss and the validation loss.
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
        (config.validation_every_n_epochs * len(losses) // config.num_training_epochs)
        * np.arange(1, len(val_losses) + 1),
        val_losses,
        lw=1.2,
        color=PALETTE_OG[1],
        label="val",
    )
    plt.xlim(0, len(losses))
    plt.ylim(0.0, 0.2)
    plt.legend(loc="upper right", frameon=True)
    save_plot(PLOT_DIR, f"3_3_loss_function.png", create_date_dir=False)
    plt.close()

    #%% Do inference, denoising one sample digit for each category (0, 1, 2, ...).
    num_timesteps = 50
    sampler = DDIM(betas, model, device=device, num_timesteps=num_timesteps)
    x = get_one_element_per_digit(mnist_test).to(device)
    # Add noise to the digits.
    x_noisy, _ = sampler.forward_sample(num_timesteps - 1, x)
    # Do inference, and store results into the GIF, using the callback.
    x_denoised = inference(
        x=x_noisy,
        sampler=sampler,
        callback=MnistInferenceGifCallback(filename=PLOT_DIR / "3_3_inference.gif"),
        call_callback_every_n_steps=2,
        conditioning=torch.arange(0, 10, device=device),
        classifier_free_guidance_scale=5,
    )
    # Compute error, as L2 norm.
    l2 = torch.nn.functional.mse_loss(x_denoised, x, reduction="mean").item()
    print(f"L2 norm after denoising: {l2:.6f}")
    plt.close()

    #%% Measure FID, generating a few digits from scratch
    # and comparing their feature distribution against the real data.
    lenet5 = LeNet5(num_classes=10)
    lenet5.load_state_dict(torch.load(DATA_DIR / "2_3_lenet5.pt"))
    lenet5.to(device)
    fid_score = fid(
        sampler=sampler,
        dataloader=dataloader_test,
        conditioning=True,
        feature_extractor_model=lenet5,
        num_batches=50,
        device=device,
        classifier_free_guidance_scale=5,
        generator=torch.Generator(device=device).manual_seed(42),
    )
    print(f"test-ddim FID: {fid_score:.4f}")

    #%% Generate a grid of digits.
    grid = generate_digits(
        sampler,
        conditioning=True,
        device=device,
        generator=torch.Generator(device=device).manual_seed(42),
        classifier_free_guidance_scale=5,
    )
    F.to_pil_image(grid).save(PLOT_DIR / "3_3_generated_digits.png")
    plt.close()

    #%% Plot a heatmap with the class embedings learnt by the model.
    class_embeddings = model.class_embedding.weight.detach().cpu().numpy()
    sns.heatmap(class_embeddings)
    save_plot(PLOT_DIR, "3_3_class_embeddings.png", create_date_dir=False)
    plt.close()
    # Also plot the correlation matrix of the class embeddings.
    sns.heatmap(pd.DataFrame(class_embeddings).T.corr())
    save_plot(PLOT_DIR, "3_3_class_embeddings_correlation.png", create_date_dir=False)
    plt.close()
