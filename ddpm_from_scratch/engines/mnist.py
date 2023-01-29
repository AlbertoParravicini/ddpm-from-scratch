"""
Code used to train diffusion models on MNIST.
Everything related to preparing the dataset, training, and inference on MNIST is here.
"""

from pathlib import Path
from typing import Callable, Optional, Union

import imageio.v2 as imageio
import numpy as np
import torch
import torchvision.transforms as T
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchtyping import TensorType
from torchvision.datasets.mnist import MNIST
from torchvision.utils import make_grid
from tqdm import tqdm

from ddpm_from_scratch.ddpm import DDPM
from ddpm_from_scratch.utils import B, C, H, W


class MnistInferenceGifCallback:
    def __init__(self, filename: Union[str, Path]):
        """
        Callback used at inference time to show the denoising of MNIST digits, plotted as a grid.

        :param filename: path where the GIF is stored.
        """
        self.writer = imageio.get_writer(filename, mode="I")

    def __call__(self, timestep: int, x: TensorType["float"]) -> None:
        # Plot multiple digits as a grid, each in a separate column
        grid = make_grid(x, padding=False, nrow=len(x))
        # Append the result to the GIF
        self.writer.append_data(np.array(T.ToPILImage()(grid)))

    def __del__(self) -> None:
        self.writer.close()


def load_mnist(data_root: Path, batch_size: int = 4) -> tuple[MNIST, DataLoader]:
    """
    Load the MNIST dataset, and wrap it into a DataLoader

    :param data_root: folder where the dataset is stored, or where it is downloaded if missing.
    :param batch_size: batch size used in the DataLoader.
    :return: the MNIST training dataset, with its DataLoader, and the MNIST test set, with its dataloader.
    """
    mnist_train = MNIST(
        root=data_root,
        download=True,
        train=True,
        transform=T.Compose(
            [
                T.ToTensor(),
                T.Normalize(0.5, 0.5),
            ]
        ),
    )
    dataloader_train = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=1)
    mnist_test = MNIST(
        root=data_root,
        download=True,
        train=False,
        transform=T.Compose(
            [
                T.ToTensor(),
                T.Normalize(0.5, 0.5),
            ]
        ),
    )
    dataloader_test = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=1)
    return mnist_train, dataloader_train, mnist_test, dataloader_test


def get_one_element_per_digit(mnist) -> TensorType[10, 1, 28, 28]:
    """
    Get a single sample digit for each target category in MNIST, and return the result as a tensor.
    The output is a `[10, 1, 28, 28]` tensor, with digits `[0, 1, 2, ..., 9]`
    """
    targets = sorted(set(mnist.targets.numpy()))
    digits = []
    for t in targets:
        digits += [mnist[np.argwhere(mnist.targets == t)[0, 0]][0]]
    return torch.stack(digits)


def train(dataloader: DataLoader, sampler: DDPM, optimizer: Optimizer, epochs: int = 1) -> list[float]:
    """
    Train a diffusion model on MNIST. At each step, sample a digit,
    sample a random timestep, add noise to the digit with intensity proportional to the timestep,
    and predict the noise that was added.

    :param dataloader: DataLoader for MNIST.
    :param sampler: instance of DDPM, containing the model to be trained.
    :param optimizer: optimizer used in the training, e.g. Adam.
    :param epochs: number of epochs for training, each corresponding to a full pass over the dataset.
    :return: the list of losses, for each step of training.
    """
    losses: list[float] = []
    progress_bar_epoch = tqdm(range(epochs), desc="training")
    for e in progress_bar_epoch:
        progress_bar_step = tqdm(dataloader, desc=f"epoch {e}")
        # Iterate over the dataset, but ignore the class for now
        for i, (x, _) in enumerate(progress_bar_step):
            # Zero gradients at every step
            optimizer.zero_grad()
            # Take a random timestep
            t = np.random.randint(sampler.num_timesteps, size=dataloader.batch_size)
            # Add some noise to the data
            with torch.no_grad():
                x_noisy, noise = sampler.forward_sample(t, x)
            # Predict the noise
            _, predicted_noise = sampler.predict_x_0_and_noise(t, x_noisy)
            # Compute loss, as L2 of real and predicted noise
            loss = torch.mean((noise - predicted_noise) ** 2)
            # Backward step
            loss.backward()
            optimizer.step()
            losses += [loss.item()]
            if i % 100 == 0:
                progress_bar_step.set_postfix({"loss": loss.item()})
        progress_bar_epoch.set_postfix({"loss": loss.item()})
    return losses


def inference(
    x: TensorType["B", "C", "H", "W", "float"],
    sampler: DDPM,
    callback: Optional[Callable[[int, TensorType["float"]], None]] = None,
    call_callback_every_n_steps: int = 50,
) -> TensorType["B", "C", "H", "W", "float"]:
    """
    Loop for diffusion inference on a batch of MNIST digits.

    :param x: batch of MNIST digits, as a tensor normalized in `[-1, 1]`.
    :param sampler: sampler used in inference, e.g. instance of `DDPM`.
    :param callback: a callback the can be called every `call_callback_every_n_steps` steps,
        e.g. to plot the intermediate results.
    :param call_callback_every_n_steps: interval, in steps, that elapses between calling the callback.
        The callback is not called if `call_callback_every_n_steps <= 0`.
        It is always called at the last step of inference.
    :return: the denoised MNIST digits, as a tensor normalized in `[-1, 1]`.
    """
    steps = np.linspace(1, 0, sampler.num_timesteps)
    for t in tqdm(steps, desc="inference", total=len(steps)):
        # Get timestep, in the range [0, num_timesteps)
        timestep = min(int(t * sampler.num_timesteps), sampler.num_timesteps - 1)
        # Inference, predict the next step given the current one
        with torch.inference_mode():
            x, _ = sampler.backward_sample(timestep, x, add_noise=t != 0)
        # Call the optional callback, every few steps
        if (
            call_callback_every_n_steps > 0
            and timestep % call_callback_every_n_steps == 0
            or timestep == sampler.num_timesteps - 1
            and callback is not None
        ):
            assert callback is not None
            callback(timestep, x)
    return x
