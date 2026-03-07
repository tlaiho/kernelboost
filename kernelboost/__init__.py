"""KernelBooster: Gradient boosting with Nadaraya-Watson (local constant) estimator as base learners."""

__version__ = "0.3.0"

from .booster import KernelBooster
from .multiclassbooster import MulticlassBooster

__all__ = [
    "KernelBooster",
    "MulticlassBooster",
]
