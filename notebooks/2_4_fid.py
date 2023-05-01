from pathlib import Path

import torch
from jaxtyping import Float
from tqdm import tqdm
from torchvision.utils import make_grid
import torchvision.transforms.functional as F

from ddpm_from_scratch.engines.mnist import inference, load_mnist
from ddpm_from_scratch.models import LeNet5, UNetWithTimestep
from ddpm_from_scratch.samplers import DDIM, DDPM
from ddpm_from_scratch.utils import LinearBetaSchedule, gaussian_frechet_distance

PLOT_DIR = Path(__file__).parent.parent / "plots"
DATA_DIR = Path(__file__).parent.parent / "data"


if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    #%% Load the LeNet5 model.
    lenet5 = LeNet5(num_classes=10)
    lenet5.load_state_dict(torch.load(DATA_DIR / "2_3_lenet5.pt"))
    lenet5.to(device)

    # Load the MNIST dataset. Split between training and test set.
    mnist_train, dataloader_train, mnist_test, dataloader_test = load_mnist(DATA_DIR, batch_size=8)

    # Compute features for training and test set.
    with torch.inference_mode():
        features_train_list: list[Float[torch.Tensor, "b n"]] = []
        for (x, _) in tqdm(dataloader_train, desc="lenet5 - training set"):
            x = x.to(device)
            f, _ = lenet5(x)
            features_train_list += [f]
        features_train = torch.concat(features_train_list, dim=0)
        features_test_list: list[Float[torch.Tensor, "b n"]] = []
        for (x, _) in tqdm(dataloader_test, desc="lenet5 - test"):
            x = x.to(device)
            f, _ = lenet5(x)
            features_test_list += [f]
        features_test = torch.concat(features_test_list, dim=0)

    # Compute statistics for training and test set,
    mean_train = features_train.mean(dim=0)
    cov_train = features_train.T.cov()
    mean_test = features_test.mean(dim=0)
    cov_test = features_test.T.cov()

    # Compute test-test and test-train FID, as lower bounds.
    test_test_fid = gaussian_frechet_distance(mean_test, cov_test, mean_test, cov_test).item()
    test_train_fid = gaussian_frechet_distance(mean_test, cov_test, mean_train, cov_train).item()
    # Test-test FID should be 0, since we're comparing the same values. Just a sanity check.
    print(f"test-test FID: {test_test_fid:.4f}")
    # Train-test FID should be ~0.8, a very low value since the distributions should also match.
    print(f"test-train FID: {test_train_fid:.4f}")

    #%% Compute test-random FID, as upper bound.
    # To estimate the FID of noise, create random noise, and compute its LeNet5 features.
    # Here the FID is super high, ~1500. This is expected, since the noise is not a digit.
    with torch.inference_mode():
        features_rand_list: list[Float[torch.Tensor, "b n"]] = []
        for (x, _) in tqdm(dataloader_test, desc="lenet5 - random data"):
            x = x.to(device)
            f, _ = lenet5(torch.rand_like(x))
            features_rand_list += [f]
        features_rand = torch.concat(features_rand_list, dim=0)
    mean_rand = features_rand.mean(dim=0)
    cov_rand = features_rand.T.cov()
    test_rand_fid = gaussian_frechet_distance(mean_test, cov_test, mean_rand, cov_rand).item()
    print(f"test-random FID: {test_rand_fid:.4f}")

    #%% Generate a bunch of digits using the pretrained model.
    # Note how we generate digits from Gaussian noise, instead of using the forward process,
    # to ensure that we are really doing generation and we don't have any information
    # that is left from real digits.
    # We only use 50 steps in inference, to keep the process fast.
    # We'll get an upper bound of FID, but that's ok! Feel free to use more steps, if you want.
    # We also limit the generation to 50 batches of 8 digits, to keep the process fast.
    unet = UNetWithTimestep().to(device)
    unet.load_state_dict(torch.load(DATA_DIR / "2_2_unet.pt"))
    betas = LinearBetaSchedule()
    num_timesteps = 50
    num_batches = 50
    sampler = DDPM(betas, unet, num_timesteps=num_timesteps, device=device)

    with torch.inference_mode():
        features_ddpm_list: list[Float[torch.Tensor, "b n"]] = []
        generated_digits: list[Float[torch.Tensor, "b n c h w"]] = []
        for i, (x, _) in enumerate(tqdm(dataloader_test, desc="lenet5 - ddpm")):
            x = x.to(device)
            noise = torch.randn_like(x)
            x_denoised = inference(x=noise, sampler=sampler, verbose=False)
            # Compute features with LeNet5
            f, _ = lenet5(x_denoised)
            features_ddpm_list += [f]
            generated_digits += [x_denoised]
            if len(features_ddpm_list) > num_batches:
                break
        features_ddpm = torch.concat(features_ddpm_list, dim=0)
    mean_ddpm = features_ddpm.mean(dim=0)
    cov_ddpm = features_ddpm.T.cov()
    test_ddpm_fid = gaussian_frechet_distance(mean_test, cov_test, mean_ddpm, cov_ddpm).item()
    # We should get a FID of ~900, which is better than random but still very far from a realistic distribution.
    print(f"test-ddpm FID: {test_ddpm_fid:.4f}")
    # Let's save all the generated digits
    generated_digits = torch.concat(generated_digits, dim=0)
    generated_digits_grid = make_grid(
        generated_digits, padding=False, nrow=len(x), normalize=True, value_range=(-1, 1)
    )
    F.to_pil_image(generated_digits_grid).save(PLOT_DIR / "2_4_generated_digits_ddpm.png")

    #%% Now we redo the same process, but we use DDIM instead of DDPM.
    # DDIM should work better than DDPM with a low number of steps, let's see if
    # we get better results.
    # For the implementation of DDIM, refer to `ddpm_from_scratch/samplers/ddim.py`
    # It's not very different from DDPM, with the main change being that we do not add random noise
    # at every timestep, but we have a fully deterministic generation.
    # DDIM paper: https://arxiv.org/pdf/2010.02502.pdf
    sampler = DDIM(betas, unet, num_timesteps=num_timesteps, device=device)

    with torch.inference_mode():
        features_ddim_list: list[Float[torch.Tensor, "b n"]] = []
        generated_digits: list[Float[torch.Tensor, "b n c h w"]] = []
        for i, (x, _) in enumerate(tqdm(dataloader_test, desc="lenet5 - ddim")):
            x = x.to(device)
            noise = torch.randn_like(x)
            x_denoised = inference(x=noise, sampler=sampler, verbose=False)
            # Compute features with LeNet5
            f, _ = lenet5(x_denoised)
            features_ddim_list += [f]
            generated_digits += [x_denoised]
            if len(features_ddim_list) > num_batches:
                break
        features_ddim = torch.concat(features_ddim_list, dim=0)
    mean_ddim = features_ddim.mean(dim=0)
    cov_ddim = features_ddim.T.cov()
    test_ddim_fid = gaussian_frechet_distance(mean_test, cov_test, mean_ddim, cov_ddim).item()
    # We should get a FID of ~600, which is better than DDPM but also far from great!
    # Let's see what we can do to make it better
    print(f"test-ddim FID: {test_ddim_fid:.4f}")
    # Let's save all the generated digits
    generated_digits = torch.concat(generated_digits, dim=0)
    generated_digits_grid = make_grid(
        generated_digits, padding=False, nrow=len(x), normalize=True, value_range=(-1, 1)
    )
    F.to_pil_image(generated_digits_grid).save(PLOT_DIR / "2_4_generated_digits_ddim.png")
