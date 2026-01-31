"""Data validation and preparation utilities."""

import pandas as pd

from ancovapy.base import DataPreparator, DataValidator
from ancovapy.types import Covariate, DependentVariable


class StandardDataValidator(DataValidator):
    """Standard validator for ANCOVA data."""

    def validate(
        self, dependent_var: DependentVariable, covariates: dict[str, Covariate]
    ) -> None:
        """Validate input data for ANCOVA analysis."""
        if len(dependent_var) == 0:
            raise ValueError("Dependent variable cannot be empty")

        n = len(dependent_var)
        for name, (data, cov_type) in covariates.items():
            if len(data) != n:
                raise ValueError(
                    f"Covariate '{name}' length ({len(data)}) does not match "
                    f"dependent variable length ({n})"
                )
            if cov_type not in ("Q", "C", "G"):
                raise ValueError(
                    f"Invalid covariate type '{cov_type}' for '{name}'. "
                    "Must be 'Q', 'C', or 'G'"
                )


class PandasDataPreparator(DataPreparator):
    """Prepare data as pandas DataFrame for statsmodels."""

    def prepare(
        self, dependent_var: DependentVariable, covariates: dict[str, Covariate]
    ) -> pd.DataFrame:
        """Convert data to pandas DataFrame."""
        data = {"y": dependent_var}

        for name, (cov_data, cov_type) in covariates.items():
            if cov_type in ("C", "G"):
                # Categorical variables
                data[name] = pd.Categorical(cov_data.astype(str))
            else:
                # Quantitative variables
                data[name] = cov_data.astype(float)

        return pd.DataFrame(data)


class BayesianDataPreparator(DataPreparator):
    """Prepare data for Bayesian analysis (standardized)."""

    def prepare(
        self, dependent_var: DependentVariable, covariates: dict[str, Covariate]
    ) -> pd.DataFrame:
        """Convert and standardize data for Bayesian analysis."""
        import numpy as np

        data = {"y": dependent_var}

        for name, (cov_data, cov_type) in covariates.items():
            if cov_type in ("C", "G"):
                # Categorical variables - convert to codes
                cat = pd.Categorical(cov_data.astype(str))
                data[name] = cat.codes
                data[f"{name}_labels"] = cat
            else:
                # Quantitative variables - standardize
                mean_val = np.mean(cov_data)
                std_val = np.std(cov_data)
                if std_val > 0:
                    data[name] = (cov_data - mean_val) / std_val
                else:
                    data[name] = cov_data - mean_val

        return pd.DataFrame(data)
