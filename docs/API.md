# API Reference

## ANCOVA Class

### `class ANCOVA(ss_type: SSType = 2)`

Classical ANCOVA (Analysis of Covariance) implementation using statsmodels.

**Parameters:**
- `ss_type` (int, optional): Sum of Squares type (1, 2, or 3). Default is 2.
  - Type I: Sequential
  - Type II: Hierarchical (recommended for balanced designs)
  - Type III: Marginal (recommended for unbalanced designs)

### Methods

#### `fit(dependent_var, covariates, alpha=0.05)`

Fit ANCOVA model to data.

**Parameters:**
- `dependent_var` (NDArray[np.float64]): Dependent variable array
- `covariates` (dict[str, Covariate]): Dictionary mapping covariate names to (data, type) tuples
- `alpha` (float, optional): Significance level for statistical tests. Default: 0.05

**Returns:**
- `ANCOVAResult`: Object containing all analysis results

**Example:**
```python
from ancovapy import ANCOVA
import numpy as np

ancova = ANCOVA(ss_type=2)
result = ancova.fit(
    dependent_var=outcomes,
    covariates={
        "age": (ages, "Q"),
        "treatment": (groups, "G"),
    }
)
print(f"F = {result.f_statistic:.2f}, p = {result.p_value:.4f}")
```

#### `fit_postpre(pre_scores, post_scores, groups, additional_covariates=None, alpha=0.05)`

Fit ANCOVA model for post-pre experimental design.

**Parameters:**
- `pre_scores` (NDArray[np.float64]): Pre-intervention scores
- `post_scores` (NDArray[np.float64]): Post-intervention scores
- `groups` (NDArray[np.str_]): Group labels for each observation
- `additional_covariates` (Optional[dict[str, Covariate]]): Optional additional covariates
- `alpha` (float, optional): Significance level. Default: 0.05

**Returns:**
- `ANCOVAResult`: Result with adjusted means and differences

## BayesianANCOVA Class

### `class BayesianANCOVA(hypothesis_type="two-sided", mcmc_samples=2000, mcmc_tune=1000, mcmc_chains=4, random_seed=None)`

Bayesian ANCOVA implementation using PyMC.

**Parameters:**
- `hypothesis_type` (Literal["two-sided", "one-sided"]): Hypothesis testing type
- `mcmc_samples` (int): Number of MCMC samples per chain. Default: 2000
- `mcmc_tune` (int): Number of tuning steps. Default: 1000
- `mcmc_chains` (int): Number of MCMC chains. Default: 4
- `random_seed` (Optional[int]): Random seed for reproducibility

### Methods

#### `fit(dependent_var, covariates, hdi_prob=0.95, rope=None)`

Fit Bayesian ANCOVA model using MCMC.

**Parameters:**
- `dependent_var` (NDArray[np.float64]): Dependent variable array
- `covariates` (dict[str, Covariate]): Dictionary mapping covariate names to (data, type) tuples
- `hdi_prob` (float, optional): Probability for HDI. Default: 0.95
- `rope` (Optional[tuple[float, float]]): Region of Practical Equivalence

**Returns:**
- `BayesianANCOVAResult`: Object containing all Bayesian analysis results

## Type Definitions

### Covariate Types

- **"Q"**: Quantitative (continuous numerical variables)
- **"C"**: Categorical (without group comparisons)
- **"G"**: Group (categorical with pairwise post-hoc tests)

For complete type definitions and result objects, see the [source code](../ancovapy/).
