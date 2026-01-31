"""Constants and shared values for ancovapy."""

from typing import Literal

# Shared literal types for validation
VALID_COVARIATE_TYPES: tuple[str, str, str] = ("Q", "C", "G")
CovariateTypeLiteral = Literal["Q", "C", "G"]

# Descriptions of covariate types
COVARIATE_TYPE_DESCRIPTIONS = {
    "Q": "Quantitative - Continuous numerical variables",
    "C": "Categorical - Categorical factors without pairwise comparisons",
    "G": "Group - Categorical factors with automatic post-hoc tests",
}
