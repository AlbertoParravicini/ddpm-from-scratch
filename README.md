# DDPM from scratch

Python & Pytorch implementation of [Denoising diffusion probabilistic models from first principles](https://liorsinai.github.io/coding/2022/12/03/denoising-diffusion-1-spiral.html)

## Installation

```shell
cd ddpm-from-scratch
# Install poetry, skip if already available
curl -sSL https://install.python-poetry.org | python3 -
# Create a conda environment and activate it
conda create -n ddpm_from_scratch python=3.9 -y; conda activate ddpm_from_scratch
# Install this project as a package, and install its dependencies
poetry install
```

## Notebooks

Notebooks inside `notebooks` follow the original blogs. You can run them to obtain plots similar to the ones in the blogs, and look at the commented code to understand what's going on under the hood.
Notebooks are meant to be read starting from the `if __name__ == "__main__"` block. You'll find plenty of comments that will guide you towards the relevant parts of the code.

### Denoising the spiral

In the first part, we learn how to create a DDPM using a toy problem: denoising a spiral. 

* `1_1_spiral.py`. Create a Gaussian distribution, and create a sprial. Also define a *variance schedule*, a.k.a. the `β schedule`, and progressively add noise to the spiral.
* `1_2_gaussian_diffusion.py`. Here we create a Gaussian Diffusion process, with forward and backward processes that we can use for sampling. We add again noise to the spiral, but this time using the forward process. We also define the backward process to iteratively remove the noise we added, but since we don't have a noise prediction model yet, we can't really denoise the spiral! 
* `1_3_train_denoiser.py`. Here we rewrite the Gaussian Diffusion process using Pytorch, and define a simple model that we plug into the Gaussian Diffusion process. We train it, and show that we can finally denoise the spiral.
* `1_4_plot_coefficients.py`. What do the coefficients of a DDPM look like? How are the boundary values of `β schedule` chosen? What's the intuition behind the DDPM posterior process? Here we visualize the coefficients of DDPM, and answer these questions.

### Tackling MNIST

In the second part, we build a realistic UNet to generate digits from the MNIST dataset, and learn how to evaluate the quality of a diffusion model.

* `2_1_mnist.py`. Let's create a very simple UNet without time-step conditioning, and see if we can use it to generate MNIST digits. Unsurprisingly, it doesn't work very well. If we look at the loss curve, we see that it goes down slightly over time, but with only 55000 parameters the model does not have enough capacity to generate digits from scratch. Indeed, we generate things that look like digits, but not quite.
* `2_2_mnist.py`. Here we modify the UNet to have conditioning on the time-step. It turns out that results are not much different than before, hinting that the model might be too simple for the task.
* `2_3_unet.py`. In this notebook, we train a much more complex UNet, not too different from the ones you might see in real papers. It has ResNet blocks, self-attention blocks, and time-step conditioning. Also, we use a cosine `β schedule` to improve the quality of the diffusion process. Results are significantly better than before! While we can denoise digits very well, we observe that we cannot quite create digits when starting from pure noise. Some additional tricks are needed!
* `2_4_lenet.py`. Although we can evaluate the quality of denosing with a simple L2 norm, measuring the quality of digits generated from scratch is more difficult. We'll use FID, a distance that measures the distribution shift between features of real and generated digits. To compute this metric, we need a classifier, or some other kind of model trained on the images that we want to generate. So, we train a simple LeNet5, and we will later use it to compute the FID.
* `2_5_fid.py`. In this notebook, we finally generate some digits, and measure their quality when compared to the real digits, using FID. While we are better than random, there's still room to improve!

## References

* **Denoising diffusion probabilistic models from first principles**, Lior Sinai, 2023
    * Part 1: [link](https://liorsinai.github.io/coding/2022/12/03/denoising-diffusion-1-spiral.html)
    * Part 2: [link](https://liorsinai.github.io/coding/2022/12/29/denoising-diffusion-2-unet.html)
    * Part 3: [link](https://liorsinai.github.io/coding/2023/01/04/denoising-diffusion-3-guidance.html)
    * GitHub: [library](https://github.com/LiorSinai/DenoisingDiffusion.jl) and [examples](https://github.com/LiorSinai/DenoisingDiffusion-examples)

* **Understanding Diffusion Models: A Unified Perspective**, Calvin Luo, 2022. [Link](https://calvinyluo.com/2022/08/26/diffusion-tutorial.html)
    * A thorough introduction to diffusion models. It defines them starting from VAEs, it derives ELBO and it explains its relevance for diffusion models. It also derives different loss functions (`x`, `epsilon` and `v` prediction), and connects diffusion models to score-based generative models. Finally, it explains conditioning, classifier guidance, and classifier-free guidance.

* **From ELBO to DDPM**, Jake Tae, 2021. [Link](https://jaketae.github.io/study/elbo/)
    * A very optimistic 7-minutes read that explains how to derive the loss function in DDPM starting from ELBO, and its connection to KL divergence.

* **What are Diffusion Models?**, Lilian Weng, 2021. [Link](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
    * Introduction to diffusion models. It has a good derivation of the loss function and of DDIM.

* **Denoising Diffusion Probabilistic Models**, Jonathan Ho et al., 2020. [Link](https://arxiv.org/pdf/2006.11239.pdf)
    * The original DDPM paper. Very well written, but some steps are better explained in the blogs above.

* **Denoising Diffusion Implicit Models**, Jiaming Song et al., 2021. [Link](https://arxiv.org/pdf/2010.02502.pdf)
    * The original DDIM paper.

* **Deep Unsupervised Learning using Nonequilibrium Thermodynamics**, Jascha Sohl-Dickstein et al., 2015. [Link](https://arxiv.org/pdf/1503.03585.pdf)
    * The paper that introduced diffusion models. The section `2.3` on Model Probability has a good derivation of the loss function.

* **Score-Based Generative Modeling Through Stochastic Differential Equations**, Yang Song et al., 2021. [Link](https://arxiv.org/pdf/2011.13456.pdf)
    * Treating diffusion models as SDE. A more advanced read, necessary to understand samplers such as DPM