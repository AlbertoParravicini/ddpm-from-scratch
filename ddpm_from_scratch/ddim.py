from typing import Callable, Optional

import torch
from torchtyping import TensorType

from ddpm_from_scratch.utils import B, T, Timestep, expand_to_dims


class DDIM:
    def __init__(
        self,
        betas: TensorType["T", "float"],
        denoise_function: Callable[[TensorType["B", "int"], TensorType["float"]], TensorType["float"]],
    ) -> None:
        self.num_timesteps = len(betas)

        ############################################
        # Coefficients used by the forward process #
        ############################################

        # Use the notation of DDIM, where α is the same as α_hat in DDPM
        alphas = 1 - betas
        alpha_cumprods = torch.cumprod(alphas, dim=0)
        # α_t
        self.alphas = alpha_cumprods
        # sqrt(α_t)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        # sqrt(1 - α_t)
        self.sqrt_one_minus_alpha = torch.sqrt(1 - self.alphas)

        #########################################################
        # Coefficients used by the backward process / posterior #
        #########################################################

        # α_t-1
        self.alphas_prev = torch.concatenate([torch.tensor([1.0]), self.alphas[:-1]])
        # sqrt(α_t-1 / α_t), used as x_t coefficient to predict x_0
        self.posterior_mean_x_0_x_t_coeff = torch.sqrt(self.alphas_prev / self.alphas)
        # sqrt(α_t-1 / α_t) * sqrt(1 - α_t), used as ε_t coefficient to predict x_0
        self.posterior_mean_x_0_epsilon_coeff = self.posterior_mean_x_0_x_t_coeff * torch.sqrt(1 - self.alphas)
        # sqrt(1 - α_t-1), direction pointing to x_t
        self.posterior_mean_x_t_coeff = torch.sqrt(1 - self.alphas_prev)

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
        sqrt_alpha[t] * x_0 + sqrt_one_minus_alpha[t] * noise
        ```

        :param t: current timestep, as integer. It must be `[0, self.num_timesteps]`
        :param x_0: value of `x_0`, the value on which the forward process q is conditioned
        :param noise: noise added to the forward process. If None, sample from a standard Gaussian
        :return: the sampled value of `q(x_i | x_0)`, and the added noise
        """
        if noise is None:
            noise = torch.randn(*x_0.shape)
        # Since `t` can be also be an array, we have to replicate it so that it can be broadcasted on `x_0`.
        sqrt_alpha = expand_to_dims(self.sqrt_alphas[t], x_0)
        sqrt_one_minus_alpha = expand_to_dims(self.sqrt_one_minus_alpha[t], x_0)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise, noise

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
        noise = self.denoise_function(x_t, _t)
        # Since `t` can be also be an array, we have to replicate it so that it can be broadcasted on `x_start`.
        coeff_x_t = expand_to_dims(self.posterior_mean_x_0_x_t_coeff[_t], x_t)
        coeff_noise = expand_to_dims(self.posterior_mean_x_0_epsilon_coeff[_t], x_t)
        x_hat_0 = coeff_x_t * x_t - coeff_noise * noise
        return x_hat_0, noise

    def backward_sample(
        self,
        t: Timestep,
        x_t: TensorType["float"],
        add_noise: bool = False,
    ) -> tuple[TensorType["float"], TensorType["float"]]:
        """
        Obtain a sample of the backward process `q(x_t-1 | x_t, x_0)`,
        by predicting `x_0` using a denoising model, and then taking a step of the backward process.
        We have that `q(x_t-1 | x_t, x_0) ~ N(μ_hat_t, β_hat_t)`, where `μ_hat_t` is a function of `x_t`
        and of the predicted `x_0`, and `β_hat_t` is only a function of the `β` schedule.

        :param t: timestep of `x_t`
        :param x_start: value of `x_t`
        :param add_noise: if True, add noise, scaled by the posterior variance, to the predicted sample of `x_t-1`.
            If False, the backward sample is deterministic. It should be False for `t = 0`, True otherwise (in DDPM)
        :return: the sample of `x_t-1`, and the predicted `x_0`
        """
        # Predict x_0 using the model
        x_hat_0, pred_noise = self.predict_x_0_and_noise(t=t, x_t=x_t)
        # Obtain the posterior mean, and obtain a sample of q(x_t-1 | x_t, x_0)
        noise_coeff = expand_to_dims(self.posterior_mean_x_t_coeff[t], x_t)
        x_t_minus_one = x_hat_0 + noise_coeff * pred_noise
        return x_t_minus_one, x_hat_0
