from pathlib import Path
import torchvision.transforms.functional as F
import seaborn as sns
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
    fid,
    generate_digits,
)
from ddpm_from_scratch.models import UNetWithConditioning, LeNet5
from ddpm_from_scratch.samplers import DDPM, DDIM
from ddpm_from_scratch.utils import LinearBetaSchedule

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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 128 if device == torch.device("cuda") else 32

    # Load the model.
    model = UNetWithConditioning(classes=10).to(device)
    model.load_state_dict(torch.load(DATA_DIR / "3_1_unet.pt"))

    # Create the diffusion process.
    num_timesteps = 1000
    betas = LinearBetaSchedule(num_train_timesteps=num_timesteps)
    ddpm = DDIM(betas, model, device=device, num_timesteps=num_timesteps)

    # Load the MNIST dataset. Split between training and test set.
    # Pass a seed to the training set dataloader, for reproducibility.
    mnist_train, dataloader_train, mnist_test, dataloader_test = load_mnist(DATA_DIR, batch_size=batch_size, seed=42)

    # Classifier-free guidances (CFG) to use. 1 means that no CFG is used,
    # and higher values should give better results, up to a certain amount.
    cfgs = [1, 5, 7, 10, 15]

    #%% Do inference, denoising one sample digit for each category (0, 1, 2, ...).
    num_timesteps = 50
    sampler = DDIM(betas, model, device=device, num_timesteps=num_timesteps)
    x = get_one_element_per_digit(mnist_test).to(device)
    # Add noise to the digits.
    x_noisy, _ = sampler.forward_sample(num_timesteps - 1, x)
    # Do inference with different classifier-free guidance scales.
    # Higher scales should be better, up to a certain amount.
    # We observe that a CFG between 5 and 7 reduces the L2 error significantly, from 0.7 to 0.5.
    # Some digits start looking acceptable: 3s look good, and also 4s and 8s are often acceptable.
    for cfg in cfgs:
        x_denoised = inference(
            x=x_noisy,
            sampler=sampler,
            callback=MnistInferenceGifCallback(filename=PLOT_DIR / f"3_2_inference_cfg={cfg}.gif"),
            call_callback_every_n_steps=2,
            classifier_free_guidance_scale=cfg,
            conditioning=torch.arange(0, 10, device=device),
        )
        # Compute error, as L2 norm.
        l2 = torch.nn.functional.mse_loss(x_denoised, x, reduction="mean").item()
        print(f"CFG={cfg} - L2 norm after denoising: {l2:.6f}")
        plt.close()

    #%% Measure FID, generating a few digits from scratch
    # and comparing their feature distribution against the real data.
    # Again, a value between 5 and 7 drops FID significantly, from > 800 to ~300
    lenet5 = LeNet5(num_classes=10)
    lenet5.load_state_dict(torch.load(DATA_DIR / "2_3_lenet5.pt"))
    lenet5.to(device)
    for cfg in cfgs:
        fid_score = fid(
            sampler=sampler,
            dataloader=dataloader_test,
            conditioning=True,
            feature_extractor_model=lenet5,
            num_batches=50,
            device=device,
            classifier_free_guidance_scale=cfg,
            generator=torch.Generator(device=device).manual_seed(42),
        )
        print(f"CFG={cfg} - test-ddim FID: {fid_score:.4f}")

    #%% Generate a grid of digits.
    for cfg in cfgs:
        grid = generate_digits(
            sampler,
            conditioning=True,
            device=device,
            generator=torch.Generator(device=device).manual_seed(42),
            classifier_free_guidance_scale=cfg,
        )
        F.to_pil_image(grid).save(PLOT_DIR / f"3_2_generated_digits_cfg={cfg}.png")
        plt.close()
