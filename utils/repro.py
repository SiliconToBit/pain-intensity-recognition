"""Reproducibility utilities — centralized seeding for deterministic runs.

For a research project that reports cross-validation results, full
reproducibility (bit-for-bit identical across runs) is the baseline
expectation. This module sets every RNG that can affect results:

    - Python ``random``
    - NumPy global RNG
    - PyTorch CPU and all CUDA devices
    - cuDNN deterministic / benchmark flags
    - DataLoader worker initialization

Note: full determinism has a small speed cost on GPU (typically 5-10%).
Disable it via ``Config.deterministic = False`` when speed matters more
than reproducibility.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed every RNG and configure cuDNN for deterministic execution.

    Args:
        seed: integer seed applied to random / numpy / torch / cuda.
        deterministic: if True, enable cuDNN deterministic mode and disable
            cuDNN benchmark (fully reproducible, slightly slower). If False,
            enable cuDNN benchmark (faster, non-deterministic).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` — give each worker a deterministic RNG.

    PyTorch spawns each worker with the base seed derived from the loader's
    ``generator``; this re-seeds NumPy and Python RNGs inside the worker so
    that any augmentation using them is also reproducible.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
