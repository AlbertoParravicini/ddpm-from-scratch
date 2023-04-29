from typing import Union
from ddpm_from_scratch.samplers.ddpm import DDPM
from ddpm_from_scratch.samplers.ddim import DDIM

Sampler = Union[DDPM, DDIM]