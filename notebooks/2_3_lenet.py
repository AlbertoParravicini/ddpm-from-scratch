from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from segretini_matplottini.utils.plot_utils import reset_plot_style, save_plot
from torch.nn.functional import binary_cross_entropy_with_logits
from tqdm import tqdm

from ddpm_from_scratch.engines.mnist import load_mnist
from ddpm_from_scratch.models.lenet5 import LeNet5

PLOT_DIR = Path(__file__).parent.parent / "plots"
DATA_DIR = Path(__file__).parent.parent / "data"


#%%
if __name__ == "__main__":
    # Setup. This is identical to `2_1_mnist.py`, but we train a UNet with timestep conditioning.
    np.random.seed(seed=42)
    torch.manual_seed(seed=42)
    reset_plot_style(xtick_major_pad=4, ytick_major_pad=4, border_width=1.5, label_pad=4)
    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    # Load the MNIST dataset.
    mnist_train, dataloader_train, mnist_test, dataloader_test = load_mnist(DATA_DIR, batch_size=8)

    # Load our LeNet5
    model = LeNet5(num_classes=10)

    # Define the optimizer.
    optimizer = torch.optim.Adam(model.parameters())

    #%% Train the model. A standard loop, iterating over the dataset.
    epochs = 10
    losses: list[float] = []
    val_losses_per_epoch: list[float] = []
    val_errors_per_epoch: list[float] = []
    for e in range(epochs):
        progress_bar_step = tqdm(dataloader_train, desc=f"epoch {e + 1}/{epochs}")
        # Iterate over the dataset
        for i, (x, y) in enumerate(progress_bar_step):
            # Turn y into one-hot encoding
            y_one_hot = torch.nn.functional.one_hot(y, 10).float()
            # Zero gradients at every step
            optimizer.zero_grad()
            # Predict the digits
            _, y_pred = model(x)
            # Compute loss, as BCE
            loss = binary_cross_entropy_with_logits(y_pred, y_one_hot, reduction="mean")
            # Also compute number of errors
            errors = (y != y_pred.argmax(axis=1)).float().mean().item()
            # Backward step
            loss.backward()
            optimizer.step()
            losses += [loss.item()]
            if i % 100 == 0:
                bar_suffix = {"loss": loss.item(), "errors": errors}
                # Add validation scores if available
                if len(val_losses_per_epoch) > 0:
                    bar_suffix |= {"val_loss": val_losses_per_epoch[-1]}
                if len(val_errors_per_epoch) > 0:
                    bar_suffix |= {"val_errors": val_errors_per_epoch[-1]}
                progress_bar_step.set_postfix(bar_suffix)
        # Do a validation step
        val_progress_bar_step = tqdm(dataloader_test, desc="validation")
        val_losses: list[float] = []
        val_errors: list[float] = []
        for i, (x, y) in enumerate(val_progress_bar_step):
            y_one_hot = torch.nn.functional.one_hot(y, 10).float()
            with torch.inference_mode():
                _, y_pred = model(x)
                # Compute BCE and errors
                val_loss = binary_cross_entropy_with_logits(y_pred, y_one_hot, reduction="mean").item()
                e = (y != y_pred.argmax(axis=1)).float().mean().item()
                val_losses += [val_loss]
                val_errors += [e]
        val_losses_per_epoch += [float(np.mean(val_losses))]
        val_errors_per_epoch += [float(np.mean(val_errors))]

    #%% Plot the training loss function
    plt.figure(figsize=(6, 6))
    plt.plot(np.arange(len(losses)), losses, lw=0.2)
    plt.plot(
        np.arange(len(losses)),
        pd.Series(losses).rolling(100).mean(),
        lw=1,
        zorder=2,
    )
    plt.xlim(0, len(losses))
    save_plot(PLOT_DIR, "2_3_loss_function.png", create_date_dir=False)
    # Print a summary
    print(
        f"final validation loss: {val_losses_per_epoch[-1]:.4f}, "
        + f"probability of error: {100 * val_errors_per_epoch[-1]:.4f}%"
    )
    # Save the model
    torch.save(model.state_dict(), DATA_DIR / "2_3_lenet5.pt")
