from pathlib import Path

import numpy as np
import torch
from segretini_matplottini.utils.plot_utils import reset_plot_style

from ddpm_from_scratch.samplers.ddpm import DDPM

from ddpm_from_scratch.samplers.ddim import DDIM
from ddpm_from_scratch.engines.mnist import (
    MnistInferenceGifCallback,
    get_one_element_per_digit,
    inference,
    load_mnist,
)
from ddpm_from_scratch.models.unet_conditioned_v2 import UNetConditioned
from ddpm_from_scratch.utils import ScaledLinearBetaSchedule
from diffusers import DDIMScheduler
from tqdm import tqdm

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

    # Define the denoising model.
    model = UNetConditioned(num_classes=10, hidden_channels=24).to(device)

    # Create the diffusion process.
    num_timesteps = 50
    betas = ScaledLinearBetaSchedule()
    ddpm_fake = DDIM(betas, model, device=device, num_timesteps=num_timesteps)
    
    # Load the MNIST dataset.
    mnist_train, dataloader_train, mnist_test, dataloader_test = load_mnist(DATA_DIR, batch_size=8)

    # Load the trained model
    model.load_state_dict(torch.load(DATA_DIR / "unet_conditioned.pt"))

    #%% Do inference, denoising one sample digit for each category (0, 1, 2, ...).
    x = get_one_element_per_digit(mnist_test).to(device)
    # This time, we use classifier-free guidance, balancing conditioned and unconditioned sampling.
    ddim = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        clip_sample=False,
        set_alpha_to_one=False,
        prediction_type="epsilon",
    )
    scale = 9
    ddim.set_timesteps(num_timesteps)
    ddim.timesteps = ddim.timesteps.to(x.device)
    callback = MnistInferenceGifCallback(filename=PLOT_DIR / f"3_4_inference_diffusers.gif")
    noise = torch.randn(*x.shape, device=x.device)
    x_noisy_fake = ddpm_fake.forward_sample(50 -1 , x, noise=noise)[0]
    x_noisy = ddim.add_noise(
        x, noise=noise, timesteps=torch.tensor(1000 - 1, device=x.device, dtype=torch.long)
    )
    model.eval()
    x_t = x_noisy
    for t in tqdm(ddim.timesteps, desc="inference"):
        # Inference, predict the next step given the current one
        with torch.inference_mode():
            noise_pred_cond = model(t.unsqueeze(0), x_t, torch.arange(0, 10, device=device))
            noise_pred_uncond = model(t.unsqueeze(0), x_t)
            noise_pred = noise_pred_uncond + scale * (noise_pred_cond - noise_pred_uncond)
            output = ddim.step(
                noise_pred,
                t,
                x_t,
                return_dict=True,
            )
            x_t, x_0 = output.prev_sample, output.pred_original_sample
        # Call the optional callback, every few steps
        if (t % 50 == 0 or t == num_timesteps - 1) and callback is not None:
            assert callback is not None
            callback(t, x_t)
    # Compute error, as L2 norm.
    l2 = torch.nn.functional.mse_loss(x_t, x, reduction="mean").item()
    print(f"L2 norm after denoising, noise strength diffusers: {l2:.6f}")
