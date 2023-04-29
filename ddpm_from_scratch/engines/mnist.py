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
from jaxtyping import Float, Integer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.utils import make_grid
from tqdm import tqdm

from ddpm_from_scratch.samplers import Sampler


class MnistInferenceGifCallback:
    def __init__(self, filename: Union[str, Path]):
        """
        Callback used at inference time to show the denoising of MNIST digits, plotted as a grid.

        :param filename: path where the GIF is stored.
        """
        self.writer = imageio.get_writer(filename, mode="I")

    def __call__(self, timestep: int, x: Float[torch.Tensor, "b ..."]) -> None:
        x = torch.clip(x, -1, 1)
        # Plot multiple digits as a grid, each in a separate column
        grid = make_grid(x, padding=False, nrow=len(x), normalize=True)
        # Append the result to the GIF
        self.writer.append_data(np.array(T.ToPILImage()(grid)))

    def __del__(self) -> None:
        self.writer.close()


def load_mnist(data_root: Path, batch_size: int = 4) -> tuple[MNIST, DataLoader, MNIST, DataLoader]:
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
    dataloader_train = DataLoader(
        mnist_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )
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
    dataloader_test = DataLoader(
        mnist_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )
    return mnist_train, dataloader_train, mnist_test, dataloader_test


def get_one_element_per_digit(mnist: MNIST) -> Float[torch.Tensor, "10 1 28 28"]:
    """
    Get a single sample digit for each target category in MNIST, and return the result as a tensor.
    The output is a `[10, 1, 28, 28]` tensor, with digits `[0, 1, 2, ..., 9]`
    """
    targets = sorted(set(mnist.targets.numpy()))
    digits = []
    for t in targets:
        digits += [mnist[np.argwhere(mnist.targets == t)[0, 0]][0]]
    return torch.stack(digits)


def train(
    dataloader: DataLoader,
    sampler: Sampler,
    optimizer: Optimizer,
    epochs: int = 1,
    device: torch.device = torch.device("cpu"),
) -> list[float]:
    """
    Train a diffusion model on MNIST. At each step, sample a digit,
    sample a random timestep, add noise to the digit with intensity proportional to the timestep,
    and predict the noise that was added.

    :param dataloader: DataLoader for MNIST.
    :param sampler: instance of DDPM or another sampler, containing the model to be trained.
    :param optimizer: optimizer used in the training, e.g. Adam.
    :param epochs: number of epochs for training, each corresponding to a full pass over the dataset.
    :return: the list of losses, for each step of training.
    """
    losses: list[float] = []
    progress_bar_epoch = tqdm(range(epochs), desc="training")
    for e in progress_bar_epoch:
        progress_bar_step = tqdm(dataloader, desc=f"epoch {e + 1}/{epochs}")
        losses_epoch: list[float] = []
        # Iterate over the dataset, but ignore the class for now
        for i, (x, _) in enumerate(progress_bar_step):
            x = x.to(device)
            # Zero gradients at every step
            optimizer.zero_grad()
            # Take a random timestep
            t = torch.randint(
                low=0,
                high=sampler.num_timesteps,
                size=(x.shape[0],),
                device=device,
            )
            # Add some noise to the data
            with torch.no_grad():
                x_noisy, noise = sampler.forward_sample(t, x)
            # Predict the noise
            predicted_noise = sampler.model(t, x_noisy)
            # Compute loss, as L2 of real and predicted noise
            loss = torch.nn.functional.mse_loss(predicted_noise, noise, reduction="mean")
            # Backward step
            loss.backward()
            optimizer.step()
            losses_epoch += [loss.item()]
            if i % 10 == 0:
                progress_bar_step.set_postfix({"loss": loss.item()})
        losses += losses_epoch
        progress_bar_epoch.set_postfix({"loss": np.mean(losses_epoch)})
    return losses


def train_with_class_conditioning(
    dataloader: DataLoader,
    sampler: Sampler,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler] = None,
    epochs: int = 1,
    device: torch.device = torch.device("cpu"),
    classifier_free_probability: float = 0.1,
    validation_dataloader: Optional[DataLoader] = None,
    validation_every_n_epochs: int = 1,
    seed: Optional[int] = None,
) -> tuple[list[float], list[float]]:
    """
    Train a diffusion model on MNIST, using class conditioning. At each step, sample a digit,
    sample a random timestep, add noise to the digit with intensity proportional to the timestep,
    and predict the noise that was added.

    :param dataloader: DataLoader for MNIST.
    :param sampler: instance of DDPM or another sampler, containing the model to be trained.
    :param optimizer: optimizer used in the training, e.g. Adam.
    :param scheduler: scheduler used to update the learning rate. The scheduler step is done after each epoch.
    :param epochs: number of epochs for training, each corresponding to a full pass over the dataset.
    :param device: device where the training is performed.
    :param classifier_free_probability: probability, in `[0, 1]` of ignoring the classes of the current samples,
        and doing a class-free prediction instead.
    :param validation_dataloader: DataLoader for the validation set.
        If not None, the model is evaluated on this dataset
    :param validation_every_n_epochs: perform a validation step on the
        validation set every `validation_every_n_epochs` epochs.
    :param seed: seed for the random number generators used in training and validation.
    :return: the list of losses, for each step of training.
        If validation_dataloader is not None, also returns the list of validation losses.
        Otherwise return an empty list instead of the validation losses.
    """
    losses: list[float] = []
    validation_losses: list[float] = []
    progress_bar_epoch = tqdm(range(epochs), desc="training")
    generator_training = torch.Generator(device=device)
    if seed is not None:
        generator_training.manual_seed(seed)
    for e in progress_bar_epoch:
        progress_bar_step = tqdm(dataloader, desc=f"epoch {e + 1}/{epochs}")
        losses_epoch: list[float] = []
        # Iterate over the dataset, this is an epoch of training
        for i, (x, y) in enumerate(progress_bar_step):
            x = x.to(device)
            y = y.to(device)
            # Swap some classes with the "empty class", marked using 10 (since MNIST classes go from 0 to 9)
            y[torch.rand(len(y)) <= classifier_free_probability] = 10
            # Zero gradients at every step
            optimizer.zero_grad()
            # Take a random timestep
            t = torch.randint(
                low=0,
                high=sampler.num_timesteps,
                size=(x.shape[0],),
                device=device,
                generator=generator_training,
            )
            # Add some noise to the data
            with torch.no_grad():
                x_noisy, noise = sampler.forward_sample(t, x, generator=generator_training)
            # Predict the noise
            _, predicted_noise = sampler.predict_x_0_and_noise(t, x_noisy, y)
            # Compute loss, as L2 of real and predicted noise
            loss = torch.nn.functional.mse_loss(predicted_noise, noise, reduction="mean")
            # Backward step
            loss.backward()
            optimizer.step()
            losses_epoch += [loss.item()]
            if i % 10 == 0:
                progress_bar_step.set_postfix({"loss": loss.item()})
        losses += losses_epoch
        progress_bar_postfix: dict[str, float] = {}  # Used to update the progress bar
        if scheduler is not None:
            scheduler.step()
            progress_bar_postfix["lr"] = scheduler.get_last_lr()[0]
        progress_bar_postfix["loss"] = float(np.mean(losses_epoch))

        # Validation loop, every validation_every_n_epochs epochs
        if validation_dataloader is not None and e % validation_every_n_epochs == 0:
            progress_bar_validation_step = tqdm(validation_dataloader, desc="validation", leave=False)
            # Create a new generator for validation. This is always the same for each validation epoch,
            # so we are sure to validate on exactly the same timesteps and random noise
            generator_validation = torch.Generator(device=device)
            if seed is not None:
                generator_validation.manual_seed(seed)
            validation_losses_epoch: list[float] = []
            with torch.no_grad():
                sampler.model.eval()
                for i, (x, y) in enumerate(progress_bar_validation_step):
                    # Identical to the training loop
                    x = x.to(device)
                    y = y.to(device)
                    t = torch.randint(
                        low=0,
                        high=sampler.num_timesteps,
                        size=(x.shape[0],),
                        device=device,
                        generator=generator_validation,
                    )
                    with torch.no_grad():
                        x_noisy, noise = sampler.forward_sample(t, x, generator=generator_validation)
                    _, predicted_noise = sampler.predict_x_0_and_noise(t, x_noisy, y)
                    loss = torch.nn.functional.mse_loss(predicted_noise, noise, reduction="mean")
                    validation_losses_epoch += [loss.item()]
                    if i % 10 == 0:
                        progress_bar_validation_step.set_postfix({"val_loss": loss.item()})
                sampler.model.train()

            # Track the mean loss during the validation step, instead of single steps,
            # to obtain a smoother estimate of the validation loss
            validation_loss_epoch = float(np.mean(validation_losses_epoch))
            validation_losses += [validation_loss_epoch]
            if validation_dataloader is not None:  # Track the validation loss in the main progress bar
                progress_bar_postfix["val_loss"] = validation_loss_epoch
        # Update the main progress bar
        progress_bar_epoch.set_postfix(progress_bar_postfix)
    return losses, validation_losses


def inference(
    x: Float[torch.Tensor, "b c h w"],
    sampler: Sampler,
    conditioning: Optional[Integer[torch.Tensor, " b"]] = None,
    callback: Optional[Callable[[int, Float[torch.Tensor, "b ..."]], None]] = None,
    call_callback_every_n_steps: int = 50,
    initial_step_percentage: float = 1,
    classifier_free_scale: float = 1,
    verbose: bool = True,
) -> Float[torch.Tensor, "b c h w"]:
    """
    Loop for diffusion inference on a batch of MNIST digits.

    :param x: batch of MNIST digits, as a tensor normalized in `[-1, 1]`.
    :param sampler: sampler used in inference, e.g. instance of `DDPM`.
    :param conditioning: a batch of class conditioning values, e.g. the classes of MNIST digits.
        If None, do not use conditioning. If not None, the model is expected to support class conditioning.
    :param callback: a callback the can be called every `call_callback_every_n_steps` steps,
        e.g. to plot the intermediate results.
    :param call_callback_every_n_steps: interval, in steps, that elapses between calling the callback.
        The callback is not called if `call_callback_every_n_steps <= 0`.
        It is always called at the last step of inference.
    :param initial_step_percentage: strength of the noise applied to the forward sample, in [0, 1].
        By default, assume we are starting from pure noise, and we are generating data from pure noise.
        If less than 1, we perform denoising starting from a noisy image, doing only a fraction of the timesteps.
    :param classifier_free_scale: if != 1, apply classifier-free guidance.
        This means that each noise prediction is computed as ϵ(x_t) + classifier_free_scale * (ϵ(x_t | y) - ϵ(x_t)).
        The value is ignored if `conditioning` is None.
    :param verbose: if True, print a progress bar during inference
    :return: the denoised MNIST digits, as a tensor normalized in `[-1, 1]`.
    """
    # We must be in eval mode
    sampler.model.eval()
    # Compute the timestep we start from, as percentage of the total
    num_timesteps = max(1, int(sampler.num_timesteps * initial_step_percentage))
    steps = np.linspace(initial_step_percentage, 0, num_timesteps)
    for t in tqdm(steps, desc="inference", total=len(steps), disable=not verbose):
        # Get timestep, in the range [0, num_timesteps)
        timestep = min(int(t * sampler.num_timesteps), num_timesteps - 1)
        # Inference, predict the next step given the current one
        with torch.inference_mode():
            x, _ = sampler.backward_sample(
                timestep,
                x,
                conditioning=conditioning,
                classifier_free_scale=classifier_free_scale,
                add_noise=t != 0,
                clip_predicted_x_0=False,
            )
        # Call the optional callback, every few steps
        if (
            call_callback_every_n_steps > 0
            and (timestep % call_callback_every_n_steps == 0 or timestep == num_timesteps - 1)
            and callback is not None
        ):
            assert callback is not None
            callback(timestep, x)
    # Clip the output in [-1, 1]
    return torch.clip(x, -1, 1)

