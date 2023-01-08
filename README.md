# DDPM from scratch

Python & Pytorch implementation of [Denoising diffusion probabilistic models from first principles](https://liorsinai.github.io/coding/2022/12/03/denoising-diffusion-1-spiral.html)

## Installation

```shell
cd ddpm-from-scratch
# Install poetry, skip if already available
curl -sSL https://install.python-poetry.org | python3 -  
# Install this project as a package, and install its dependencies
poetry install
```

## Notebooks

Notebooks inside `notebooks` follow the original blogs. You can run them to obtain plots similar to the ones in the blogs, and look at the commented code to understand what's going on under the hood.

* `1_1_spiral.py`. Create a Gaussian distribution, and create a sprial. Also define a *variance schedule*, a.k.a. the `β schedule`, and progressively add noise to the spiral.
* `1_2_gaussian_diffusion.py`. Here we create a Gaussian Diffusion process, with forward and backward processes that we can use for sampling. We add again noise to the spiral, but this time using the forward process. We also define the backward process to iteratively remove the noise we added, but since we don't have a noise prediction model yet, we can't really denoise the spiral! 
* `1_3_train_denoiser.py`. Here we rewrite the Gaussian Diffusion process using Pytorch, and define a simple model that we plug into the Gaussian Diffusion process. We train it, and show that we can finally denoise the spiral.

## References

* **Denoising diffusion probabilistic models from first principles**, Lior Sinai, 2023
    * Part 1: [link](https://liorsinai.github.io/coding/2022/12/03/denoising-diffusion-1-spiral.html)
    * Part 2: [link](https://liorsinai.github.io/coding/2022/12/29/denoising-diffusion-2-unet.html)
    * Part 3: [link](https://liorsinai.github.io/coding/2023/01/04/denoising-diffusion-3-guidance.html)
    * GitHub: [library](https://github.com/LiorSinai/DenoisingDiffusion.jl) and [examples](https://github.com/LiorSinai/DenoisingDiffusion-examples)
