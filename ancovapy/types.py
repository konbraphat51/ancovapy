"""Type definitions for ancovapy."""

from typing import Literal, Tuple, Union
import numpy as np
import numpy.typing as npt

# Covariate types
# Q: Quantitative (continuous numerical)
# C: Categorical (no group comparison)
# G: Categorical with Group comparison (requires post-hoc tests)
CovariateType = Literal["Q", "C", "G"]

# Hypothesis types for Bayesian ANCOVA
# two-sided: Test for difference in either direction
# one-sided: Test for directional difference
HypothesisType = Literal["two-sided", "one-sided"]

# SS (Sum of Squares) types
SSType = Literal[1, 2, 3]

# Covariate specification: (data array, type)
Covariate = Tuple[npt.NDArray[np.float64], CovariateType]

# For representing dependent variable data
DependentVariable = npt.NDArray[np.float64]
