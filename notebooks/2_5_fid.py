from pathlib import Path

import torch
from jaxtyping import Float
from tqdm import tqdm

from ddpm_from_scratch.engines.mnist import inference, load_mnist
from ddpm_from_scratch.models.lenet5 import LeNet5
from ddpm_from_scratch.models.unet import UNet
from ddpm_from_scratch.samplers.ddpm import DDPM
from ddpm_from_scratch.utils import CosineBetaSchedule, gaussian_frechet_distance

PLOT_DIR = Path(__file__).parent.parent / "plots"
DATA_DIR = Path(__file__).parent.parent / "data"


if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    #%% Load the LeNet5 model.
    lenet5 = LeNet5(num_classes=10)
    lenet5.load_state_dict(torch.load(DATA_DIR / "2_4_lenet.pt")).to(device)

    # Load the MNIST dataset. Split between training and test set.
    mnist_train, dataloader_train, mnist_test, dataloader_test = load_mnist(DATA_DIR, batch_size=8)

    # Compute features for training and test set.
    with torch.inference_mode():
        features_train_list: list[Float[torch.Tensor, "b n"]] = []
        for (x, _) in tqdm(dataloader_train, desc="lenet5 - training"):
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
    print(f"test-test FID: {test_test_fid:.4f}")
    print(f"test-train FID: {test_train_fid:.4f}")

    #%% Compute test-random FID, as upper bound.
    # To estimate the FID of noise, create random noise, and compute its LeNet5 features.
    with torch.inference_mode():
        features_rand_list: list[Float[torch.Tensor, "b n"]] = []
        for (x, _) in tqdm(dataloader_test, desc="lenet5 - random"):
            x = x.to(device)
            f, _ = lenet5(torch.rand_like(x))
            features_rand_list += [f]
        features_rand = torch.concat(features_rand_list, dim=0)
    mean_rand = features_rand.mean(dim=0)
    cov_rand = features_rand.T.cov()
    test_rand_fid = gaussian_frechet_distance(mean_test, cov_test, mean_rand, cov_rand).item()
    print(f"test-random FID: {test_rand_fid:.4f}")

    #%% Generate a bunch of digits using the pretrained model

    unet = UNet()
    unet.load_state_dict(torch.load(DATA_DIR / "2_3_unet.pt"))
    betas = CosineBetaSchedule()
    ddpm = DDPM(betas, unet, num_timesteps=1000, device=device)

    with torch.inference_mode():
        features_ddpm_list: list[Float[torch.Tensor, "b n"]] = []
        for (x, _) in tqdm(dataloader_test, desc="lenet5 - ddpm"):
            x = x.to(device)
            x_denoised = inference(x=torch.rand_like(x), sampler=ddpm, verbose=False)
            f, _ = lenet5(x_denoised)
            features_ddpm_list += [f]
            if len(features_ddpm_list) > 10:
                break
        features_ddpm = torch.concat(features_ddpm_list, dim=0)
    mean_ddpm = features_ddpm.mean(dim=0)
    cov_ddpm = features_ddpm.T.cov()
    test_ddpm_fid = gaussian_frechet_distance(mean_test, cov_test, mean_ddpm, cov_ddpm).item()
    print(f"test-ddpm FID: {test_ddpm_fid:.4f}")
