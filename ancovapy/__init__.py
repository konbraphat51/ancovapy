"""
ancovapy - A Python library for ANCOVA and Bayesian ANCOVA analysis.

This library provides scientifically rigorous implementations of:
- Classical ANCOVA with post-hoc tests
- Bayesian ANCOVA with MCMC sampling
- Support for post-pre experimental designs
- Strict type checking and validation
"""

from ancovapy.ancova import ANCOVA
from ancovapy.bayesian_ancova import BayesianANCOVA
from ancovapy.results import ANCOVAResult, BayesianANCOVAResult
from ancovapy.types import CovariateType, HypothesisType

__version__ = "0.1.0"
__all__ = [
    "ANCOVA",
    "ANCOVAResult",
    "BayesianANCOVA",
    "BayesianANCOVAResult",
    "CovariateType",
    "HypothesisType",
]
