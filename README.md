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

* **From ELBO to DDPM**, Jake Tae, 2021. [link](https://jaketae.github.io/study/elbo/)
    * A very optimistic 7-minutes read that explains how to derive the loss function in DDPM starting from ELBO, and its connection to KL divergence.

* **What are Diffusion Models?**, Lilian Weng, 2021. [Link](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
    * Introduction to diffusion models. It has a good derivation of the loss function and of DDIM.

* **Denoising Diffusion Probabilistic Models**, Jonathan Ho et al., 2020. [link](https://arxiv.org/pdf/2006.11239.pdf)
    * The original DDPM paper. Very well written, but some steps are better explained in the blogs above.

* **Denoising Diffusion Implicit Models**, Jiaming Song et al., 2021. [link](https://arxiv.org/pdf/2010.02502.pdf)
    * The original DDIM paper.

* **Deep Unsupervised Learning using Nonequilibrium Thermodynamics**, Jascha Sohl-Dickstein et al., 2015. [link](https://arxiv.org/pdf/1503.03585.pdf)
    * The paper that introduced diffusion models. The section `2.3` on Model Probability has a good derivation of the loss function.

* **Score-Based Generative Modeling Through Stochastic Differential Equations**, Yang Song et al., 2021. [link](https://arxiv.org/pdf/2011.13456.pdf)
    * Treating diffusion models as SDE. A more advanced read, necessary to understand samplers such as DPM