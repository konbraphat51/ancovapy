"""Common helper functions for ANCOVA analyses."""

import pandas as pd

from ancovapy.constants import VALID_COVARIATE_TYPES
from ancovapy.types import Covariate


def convert_to_categorical(data: pd.Series | list | tuple) -> pd.Categorical:
    """
    Convert data to pandas Categorical, handling both strings and numbers.
    
    Args:
        data: Input data (can be strings or numbers)
        
    Returns:
        pd.Categorical: Categorical representation
    """
    return pd.Categorical(pd.Series(data).astype(str))


def validate_covariate_type(cov_type: str, covariate_name: str) -> None:
    """
    Validate that covariate type is valid.
    
    Args:
        cov_type: Covariate type to validate
        covariate_name: Name of the covariate (for error messages)
        
    Raises:
        ValueError: If covariate type is not valid
    """
    if cov_type not in VALID_COVARIATE_TYPES:
        raise ValueError(
            f"Invalid covariate type '{cov_type}' for '{covariate_name}'. "
            f"Must be one of {VALID_COVARIATE_TYPES}"
        )
