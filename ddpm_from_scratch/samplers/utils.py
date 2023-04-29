from typing import Union

from ddpm_from_scratch.samplers.ddim import DDIM
from ddpm_from_scratch.samplers.ddpm import DDPM

Sampler = Union[DDPM, DDIM]
