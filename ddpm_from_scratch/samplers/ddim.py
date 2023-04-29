from typing import Optional

import torch
from torch import nn
from torchtyping import TensorType

from ddpm_from_scratch.utils import (B, BetaSchedule, T, Timestep,
                                     expand_to_dims)


class DDIM:
    def __init__(self, betas: BetaSchedule, model: nn.Module, num_timesteps: int, device: torch.device) -> None:
        """
        DDIM sampler from "Diffusion Denoising Implicit Models", Jiaming Song et al., 2020.
        https://arxiv.org/pdf/2010.02502.pdf
        """

        self.num_timesteps = num_timesteps
        # Store the number of steps on which the model is trained, since if we do inference on fewer timesteps
        # we need to multiply the inference timestep by the training timesteps / inference timesteps.
        # That's because the model is trained with a time-step conditioning
        # that assumes the number of training timesteps.
        self._num_train_timesteps = betas.num_train_timesteps

        ############################################
        # Coefficients used by the forward process #
        ############################################

        self.betas = betas.betas(self.num_timesteps).to(device)
        # α_t. Use the notation of DDIM, where α is the same as α_hat in DDPM
        self.alpha_cumprods = torch.cumprod(1 - self.betas, dim=0)

        #########################################################
        # Coefficients used by the backward process / posterior #
        #########################################################

        # α_t-1
        self.alpha_cumprods_prev = torch.concatenate(
            [torch.tensor([1.0], device=self.alpha_cumprods.device), self.alpha_cumprods[:-1]]
        )

        ################################
        # Function learnt by the model #
        ################################

        self.model = model
        try:  # Move the denoise function to the device, if it's a torch nn.Module
            self.model.to(device)
        except AttributeError:
            pass

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
            noise = torch.randn(*x_0.shape, device=x_0.device)
        # Since `t` can be also be an array, we have to replicate it so that it can be broadcasted on `x_0`.
        sqrt_alpha = expand_to_dims(self.alpha_cumprods[t] ** 0.5, x_0)
        sqrt_one_minus_alpha = expand_to_dims((1 - self.alpha_cumprods[t]) ** 0.5, x_0)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise, noise

    def predict_x_0_and_noise(
        self,
        t: Timestep,
        x_t: TensorType["float"],
        conditioning: Optional[TensorType["float"]] = None,
        classifier_free_scale: float = 1,
    ) -> tuple[TensorType["float"], TensorType["float"]]:
        """
        Compute a sample of the backward process `q(x_0 | x_t)`, by denoising `x_t` using a model,
        and return both the sample and the predicted noise.
        This is computed as `x_hat_0 = 1/sqrt(α_hat_t) * x_t  - sqrt(1 - α_hat_t) / sqrt(α_hat_t)

        :param t: timestep(s) for the prediction, in the range `[0, self.num_timesteps)`
        :param x_t: tensor used for prediction, it represents `x_t`
        :param conditioning: additional conditioning applied to the model, e.g. to specify classes or text.
        :param classifier_free_scale: if != 1, apply classifier-free guidance.
            This means that each noise prediction is computed as ϵ(x_t) + classifier_free_scale * (ϵ(x_t | y) - ϵ(x_t)).
            The value is ignored if `conditioning` is None.
        :return: prediction of `x_0` and prediction of the noise added to `x_0` to obtain `x_start`
        """
        # Ensure the timestep is an integer tensor.
        _t: torch.Tensor = torch.tensor(t, dtype=torch.long, device=x_t.device) if not torch.is_tensor(t) else t
        # Ensure the timestep is not a scalar, it must have at least 1 dimension
        if len(_t.shape) == 0:
            _t = _t.unsqueeze(0)
        # The timestep used for conditioning must be scaled so that it matches the number of timesteps
        # used during training.
        _t_scaled = (_t * self._num_train_timesteps / self.num_timesteps).to(torch.long)
        ## Predict noise with our model. Pass conditioning only if not None.
        # This allows supporting models that don't expect an additional conditioning
        noise = self.model(_t_scaled, x_t, conditioning) if conditioning is not None else self.model(_t_scaled, x_t)
        # Apply classifier-free guidance if required, by denoising again but without conditioning,
        # then blending the two predictions.
        if conditioning is not None and classifier_free_scale != 1:
            noise_cf = self.model(_t_scaled, x_t)
            noise = noise_cf + classifier_free_scale * (noise - noise_cf)
        # Equation (12) of DDIM (https://arxiv.org/pdf/2010.02502.pdf), "predicted x_0".
        # Since `_t` can be also be an array, we have to replicate it so that it can be broadcasted on `x_t`.
        x_hat_0 = expand_to_dims(
            (x_t - noise * (1 - self.alpha_cumprods[_t]) ** 0.5) / (self.alpha_cumprods[_t] ** 0.5), x_t
        )

        return x_hat_0, noise

    def backward_sample(
        self,
        t: Timestep,
        x_t: TensorType["float"],
        conditioning: Optional[TensorType["float"]] = None,
        classifier_free_scale: float = 1,
        clip_predicted_x_0: bool = True,
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
        # Predict x_0 using the model.
        x_hat_0, pred_noise = self.predict_x_0_and_noise(
            t=t, x_t=x_t, conditioning=conditioning, classifier_free_scale=classifier_free_scale
        )
        if clip_predicted_x_0:
            x_hat_0 = torch.clip(x_hat_0, -1, 1)
        # Obtain the posterior mean, and obtain a sample of q(x_t-1 | x_t, x_0).
        # Equation (12) of DDIM paper, https://arxiv.org/pdf/2010.02502.pdf, assuming sigma_t = 0.
        noise_coeff = expand_to_dims((1 - self.alpha_cumprods_prev[t]) ** 0.5, x_t)
        x_hat_0_coeff = expand_to_dims(self.alpha_cumprods_prev[t] ** 0.5, x_t)
        x_t_minus_one = x_hat_0_coeff * x_hat_0 + noise_coeff * pred_noise
        return x_t_minus_one, x_hat_0
