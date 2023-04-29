from pathlib import Path

import numpy as np
import torch
from segretini_matplottini.utils.plot_utils import reset_plot_style

from ddpm_from_scratch.engines.mnist import MnistInferenceGifCallback, get_one_element_per_digit, inference, load_mnist
from ddpm_from_scratch.models.unet_conditioned_v2 import UNetConditioned
from ddpm_from_scratch.samplers.ddim import DDIM
from ddpm_from_scratch.samplers.ddpm import DDPM
from ddpm_from_scratch.utils import CosineBetaSchedule, ScaledLinearBetaSchedule

PLOT_DIR = Path(__file__).parent.parent / "plots"
DATA_DIR = Path(__file__).parent.parent / "data"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


#%%
if __name__ == "__main__":
    # Setup.
    np.random.seed(seed=42)
    torch.manual_seed(seed=42)
    reset_plot_style(xtick_major_pad=4, ytick_major_pad=4, border_width=1.5, label_pad=4)
    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    # Create the diffusion process.
    num_timesteps = 50
    betas = ScaledLinearBetaSchedule()
    # Define the denoising model.
    model = UNetConditioned(num_classes=10, hidden_channels=24)
    # Create the sampler.
    ddpm = DDIM(betas, model, device=device, num_timesteps=num_timesteps)

    # Load the MNIST dataset.
    mnist_train, dataloader_train, mnist_test, dataloader_test = load_mnist(DATA_DIR, batch_size=8)

    # Load the trained model
    model.load_state_dict(torch.load(DATA_DIR / "unet_conditioned.pt"))

    #%% Do inference, denoising one sample digit for each category (0, 1, 2, ...).
    x = get_one_element_per_digit(mnist_test).to(device)
    # This time, we use classifier-free guidance, balancing conditioned and unconditioned sampling.
    noise_strengths = [1]
    for noise_strength in noise_strengths:
        x_noisy, _ = ddpm.forward_sample(int((num_timesteps - 1) * noise_strength), x)
        # Do inference, and store results into the GIF, using the callback.
        x_denoised = inference(
            x=x_noisy,
            sampler=ddpm,
            conditioning=torch.arange(0, 10, device=device),
            callback=MnistInferenceGifCallback(filename=PLOT_DIR / f"3_2_inference_{noise_strength:.2f}.gif"),
            call_callback_every_n_steps=5,
            initial_step_percentage=noise_strength,
            classifier_free_scale=7,
        )
        # Compute error, as L2 norm.
        l2 = torch.nn.functional.mse_loss(x_denoised, x, reduction="mean").item()
        print(f"L2 norm after denoising, noise strength {noise_strength:.2f}: {l2:.6f}")
