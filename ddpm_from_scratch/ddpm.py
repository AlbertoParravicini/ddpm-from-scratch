from typing import Callable, Optional

import torch
from torchtyping import TensorType

from ddpm_from_scratch.utils import B, T, Timestep, expand_to_dims


class DDPM:
    def __init__(
        self,
        betas: TensorType["T", "float"],
        denoise_function: Callable[[TensorType["B", "int"], TensorType["float"]], TensorType["float"]],
    ) -> None:
        self.num_timesteps = len(betas)

        ############################################
        # Coefficients used by the forward process #
        ############################################

        self.betas = betas
        self.alphas = 1 - betas
        self.alpha_cumprods = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprods_prevs = torch.concatenate([torch.tensor([1.0], device=self.alpha_cumprods.device), self.alpha_cumprods[:-1]])
        self.sqrt_alpha_cumprods = torch.sqrt(self.alpha_cumprods)
        self.sqrt_one_minus_alpha = torch.sqrt(1 - self.alpha_cumprods)

        #########################################################
        # Coefficients used by the backward process / posterior #
        #########################################################

        # 1 / sqrt(α_hat_t), used to estimate x_hat_0
        self.sqrt_reciprocal_alpha_cumprods = 1 / torch.sqrt(self.alpha_cumprods)
        # sqrt(1 / α_hat_t - 1), used to estimate x_hat_0
        self.sqrt_reciprocal_alpha_cumprods_minus_one = torch.sqrt(1 / self.alpha_cumprods - 1)
        # "beta_hat_t", β_t * (1 - α_hat_t-1) /  (1 - α_hat_t), variance of q(x_t-1 | x_t, x_0)
        self.posterior_variance = self.betas * (1 - self.alpha_cumprods_prevs) / (1 - self.alpha_cumprods)
        # "alpha_hat_t", mean of the backward process, as linear combination of x_t and x_0.
        self.posterior_mean_x_0_coeff = self.betas * torch.sqrt(self.alpha_cumprods_prevs) / (1 - self.alpha_cumprods)
        self.posterior_mean_x_t_coeff = (
            (1 - self.alpha_cumprods_prevs) * torch.sqrt(self.alphas) / (1 - self.alpha_cumprods)
        )

        ################################
        # Function learnt by the model #
        ################################

        self.denoise_function = denoise_function

    def forward_sample(
        self, t: Timestep, x_0: TensorType["float"], noise: Optional[TensorType["float"]] = None
    ) -> tuple[TensorType["float"], TensorType["float"]]:
        """
        Compute `q(x_i | x_0)`, as a sample of a Gaussian process with equation
        ```
        sqrt_alpha_cumprods[t] * x_start + sqrt_one_minus_alpha[t] * noise
        ```

        :param t: current timestep, as integer. It must be `[0, self.num_timesteps]`
        :param x_start: value of `x_0`, the value on which the forward process q is conditioned
        :param noise: noise added to the forward process. If None, sample from a standard Gaussian
        :return: the sampled value of `q(x_i | x_0)`, and the added noise
        """
        if noise is None:
            noise = torch.randn(*x_0.shape, device=x_0.device)
        # Since `t` can be also be an array, we have to replicate it so that it can be broadcasted on `x_0`.
        sqrt_alpha_cumprod = expand_to_dims(self.sqrt_alpha_cumprods[t], x_0)
        sqrt_one_minus_alpha = expand_to_dims(self.sqrt_one_minus_alpha[t], x_0)
        return sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha * noise, noise

    def predict_x_0_and_noise(
        self, t: Timestep, x_t: TensorType["float"]
    ) -> tuple[TensorType["float"], TensorType["float"]]:
        """
        Compute a sample of the backward process `q(x_0 | x_t)`, by denoising `x_t` using a model,
        and return both the sample and the predicted noise.
        This is computed as `x_hat_0 = 1/sqrt(α_hat_t) * x_t  - sqrt(1 - α_hat_t) / sqrt(α_hat_t)

        :param t: timestep(s) for the prediction, in the range `[0, self.num_timesteps)`
        :param x_start: tensor used for prediction, it represents `x_t`
        :return: prediction of `x_0` and prediction of the noise added to `x_0` to obtain `x_start`
        """
        # Ensure the timestep is an integer tensor
        _t: torch.Tensor = torch.tensor(t, dtype=torch.long, device=x_t.device) if not torch.is_tensor(t) else t
        # Ensure the timestep is not a scalar, it must have at least 1 dimension
        if len(_t.shape) == 0:
            _t = _t.unsqueeze(0)
        # Predict noise with our model
        noise = self.denoise_function(_t, x_t)
        # Since `t` can be also be an array, we have to replicate it so that it can be broadcasted on `x_t`.
        coeff_x_t = expand_to_dims(self.sqrt_reciprocal_alpha_cumprods[_t], x_t)
        coeff_noise = expand_to_dims(self.sqrt_reciprocal_alpha_cumprods_minus_one[_t], x_t)
        x_hat_0 = coeff_x_t * x_t - coeff_noise * noise / coeff_x_t
        return x_hat_0, noise

    def _posterior_mean_variance(
        self, t: Timestep, x_0: TensorType["float"], x_t: TensorType["float"]
    ) -> tuple[float, float]:
        """
        Obtain the mean and variance of q(x_t-1 | x_t, x_0)
        """
        # Since `t` can be also be an array, we have to replicate it so that it can be broadcasted on `x_0`.
        posterior_mean_x_0_coeff = expand_to_dims(self.posterior_mean_x_0_coeff[t], x_0)
        posterior_mean_x_t_coeff = expand_to_dims(self.posterior_mean_x_t_coeff[t], x_0)
        posterior_mean = posterior_mean_x_0_coeff * x_0 + posterior_mean_x_t_coeff * x_t
        posterior_variance = self.posterior_variance[t]
        return posterior_mean, posterior_variance

    def backward_sample(
        self,
        t: Timestep,
        x_t: TensorType["float"],
        clip_predicted_x_0: bool = True,
        add_noise: bool = True,
    ) -> tuple[TensorType["float"], TensorType["float"]]:
        """
        Obtain a sample of the backward process `q(x_t-1 | x_t, x_0)`,
        by predicting `x_0` using a denoising model, and then taking a step of the backward process.
        We have that `q(x_t-1 | x_t, x_0) ~ N(μ_hat_t, β_hat_t)`, where `μ_hat_t` is a function of `x_t`
        and of the predicted `x_0`, and `β_hat_t` is only a function of the `β` schedule.

        :param t: timestep of `x_t`
        :param x_start: value of `x_t`
        :param clip_predicted_x_0: if True, clip the predicted value of `x_0` in `[-1, 1]`
            This is meaningful only for denoising the spiral! We mights other values for images
        :param add_noise: if True, add noise, scaled by the posterior variance, to the predicted sample of `x_t-1`.
            If False, the backward sample is deterministic. It should be False for `t = 0`, True otherwise (in DDPM)
        :return: the sample of `x_t-1`, and the predicted `x_0`
        """
        # Predict x_0 using the model
        x_hat_0, _ = self.predict_x_0_and_noise(t, x_t)
        if clip_predicted_x_0:
            x_hat_0 = torch.clip(x_hat_0, -1, 1)
        # Obtain the posterior mean and variance, and obtain a sample of q(x_t-1 | x_t, x_0)
        posterior_mean, posterior_variance = self._posterior_mean_variance(t, x_0=x_hat_0, x_t=x_t)
        x_t_minus_one = posterior_mean
        # Add noise to the sample, instead of taking a deterministic step
        if add_noise:
            noise = torch.randn(*x_t.shape, device=x_t.device)
            x_t_minus_one += torch.sqrt(posterior_variance) * noise
        return x_t_minus_one, x_hat_0
