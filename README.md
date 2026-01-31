# ancovapy

A Python library for scientifically rigorous ANCOVA (Analysis of Covariance) and Bayesian ANCOVA analysis.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Classical ANCOVA**: Frequentist analysis of covariance using statsmodels
- **Bayesian ANCOVA**: MCMC-based Bayesian inference using PyMC
- **Flexible Covariate Specification**: Support for quantitative, categorical, and group-comparison covariates
- **Post-hoc Tests**: Automatic pairwise comparisons for group covariates
- **Post-Pre Design Support**: Specialized functions for pre-post experimental designs
- **Strict Type Checking**: Full type hints for better code quality
- **Scientific Rigor**: Implementations follow established statistical best practices

## Installation

### Using pip

```bash
pip install ancovapy
```

### Using uv (recommended for development)

```bash
uv pip install ancovapy
```

### From source

```bash
git clone https://github.com/konbraphat51/ancovapy.git
cd ancovapy
uv pip install -e .
```

## Quick Start

### Classical ANCOVA

```python
import numpy as np
from ancovapy import ANCOVA

# Prepare data
dependent_var = np.array([...])  # Your outcome variable

# Define covariates with types:
# "Q" = Quantitative (continuous)
# "C" = Categorical (no group comparison)
# "G" = Group (categorical with pairwise comparisons)
covariates = {
    "age": (np.array([...]), "Q"),
    "treatment": (np.array([...]), "G"),
    "site": (np.array([...]), "C"),
}

# Fit model
ancova = ANCOVA(ss_type=2)  # Type II Sum of Squares (default)
result = ancova.fit(dependent_var, covariates)

# View results
print(f"F-statistic: {result.f_statistic:.4f}")
print(f"p-value: {result.p_value:.4f}")
print(f"R²: {result.r_squared:.4f}")

# Check covariate effects
for stat in result.covariate_stats:
    print(f"{stat.name}: β={stat.coefficient:.3f}, p={stat.p_value:.4f}")

# Post-hoc tests (for G-type covariates)
if result.posthoc_results:
    for ph in result.posthoc_results:
        print(f"{ph.group1} vs {ph.group2}: diff={ph.mean_diff:.3f}, p={ph.p_value:.4f}")
```

### Bayesian ANCOVA

```python
from ancovapy import BayesianANCOVA

# Prepare data (same as above)
dependent_var = np.array([...])
covariates = {
    "age": (np.array([...]), "Q"),
    "treatment": (np.array([...]), "G"),
}

# Fit Bayesian model
bayesian_ancova = BayesianANCOVA(
    hypothesis_type="two-sided",  # or "one-sided"
    mcmc_samples=2000,
    mcmc_chains=4,
)
result = bayesian_ancova.fit(dependent_var, covariates)

# View results
for stat in result.covariate_stats:
    print(f"{stat.name}:")
    print(f"  Mean: {stat.mean:.3f}")
    print(f"  95% HDI: [{stat.hdi_lower:.3f}, {stat.hdi_upper:.3f}]")
    print(f"  P(β > 0): {stat.prob_positive:.3f}")

# Check MCMC diagnostics
print(f"Max R-hat: {result.rhat_max:.4f} (should be < 1.01)")
print(f"Min ESS: {result.ess_bulk_min:.0f} (should be > 400)")
print(f"Divergences: {result.divergences}")

# Group comparisons
if result.group_comparisons:
    for comp in result.group_comparisons:
        print(f"{comp.group1} vs {comp.group2}:")
        print(f"  Mean diff: {comp.mean_diff:.3f}")
        print(f"  95% HDI: [{comp.hdi_lower:.3f}, {comp.hdi_upper:.3f}]")
        print(f"  P(group1 > group2): {comp.prob_greater:.3f}")
```

### Post-Pre Design Analysis

```python
from ancovapy import ANCOVA, BayesianANCOVA

# Prepare data
pre_scores = np.array([...])   # Baseline measurements
post_scores = np.array([...])  # Follow-up measurements
groups = np.array(["control", "treatment", ...])  # Group labels

# Classical approach
ancova = ANCOVA(ss_type=2)
result = ancova.fit_postpre(pre_scores, post_scores, groups)

# Adjusted means for each group
for group, mean in result.adjusted_means.items():
    print(f"{group}: {mean:.3f}")

# Pairwise differences
for (g1, g2), diff in result.adjusted_mean_diffs.items():
    ci = result.credible_intervals[(g1, g2)]
    print(f"{g1} - {g2}: {diff:.3f} (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])")

# Bayesian approach
bayesian_ancova = BayesianANCOVA()
result = bayesian_ancova.fit_postpre(pre_scores, post_scores, groups)

# Adjusted means with credible intervals
for group, (mean, hdi_low, hdi_high) in result.adjusted_means.items():
    print(f"{group}: {mean:.3f} (95% CrI: [{hdi_low:.3f}, {hdi_high:.3f}])")
```

## Covariate Types

ancovapy supports three types of covariates:

1. **"Q" (Quantitative)**: Continuous numerical variables (e.g., age, baseline score)
   - Treated as linear predictors
   - No post-hoc tests

2. **"C" (Categorical)**: Categorical variables without group comparisons (e.g., study site, sex)
   - Included as fixed effects
   - No post-hoc tests

3. **"G" (Group)**: Categorical variables requiring pairwise comparisons (e.g., treatment groups)
   - Included as fixed effects
   - Automatic post-hoc tests for all pairwise comparisons
   - Tukey HSD for classical ANCOVA
   - Bayesian pairwise comparisons with credible intervals

## Sum of Squares Types

Classical ANCOVA supports different SS (Sum of Squares) types:

- **Type I (Sequential)**: Effects are adjusted for covariates entered before them
- **Type II (Hierarchical)**: Effects are adjusted for all other effects (default, recommended for balanced designs)
- **Type III (Marginal)**: Effects are adjusted for all other effects, including interactions (recommended for unbalanced designs)

```python
# Specify SS type when creating ANCOVA object
ancova = ANCOVA(ss_type=2)  # Type II (default)
ancova = ANCOVA(ss_type=3)  # Type III for unbalanced designs
```

## Bayesian Hypothesis Testing

Bayesian ANCOVA supports both one-sided and two-sided hypothesis testing:

```python
# Two-sided: test for any difference
bayesian_ancova = BayesianANCOVA(hypothesis_type="two-sided")

# One-sided: test for directional difference
bayesian_ancova = BayesianANCOVA(hypothesis_type="one-sided")
```

Results include:
- Mean posterior estimates
- Highest Density Intervals (HDI)
- Probability of positive/negative effects
- ROPE (Region of Practical Equivalence) decision making

## MCMC Diagnostics

Bayesian ANCOVA provides diagnostic information to assess MCMC convergence:

- **R-hat**: Should be < 1.01 (indicates convergence)
- **ESS (Effective Sample Size)**: Should be > 400 for reliable inference
- **Divergences**: Should be 0 (indicates sampling issues if > 0)

```python
result = bayesian_ancova.fit(dependent_var, covariates)

# Check diagnostics
if result.rhat_max > 1.01:
    print("Warning: MCMC may not have converged. Consider more samples.")
if result.ess_bulk_min < 400:
    print("Warning: Low effective sample size. Consider more samples.")
if result.divergences > 0:
    print(f"Warning: {result.divergences} divergences detected.")
```

## References

### ANCOVA Methodology

1. **Maxwell, S. E., & Delaney, H. D. (2004).** *Designing experiments and analyzing data: A model comparison perspective* (2nd ed.). Lawrence Erlbaum Associates.
   - Comprehensive treatment of ANCOVA and experimental design

2. **Rutherford, A. (2011).** *ANOVA and ANCOVA: A GLM approach* (2nd ed.). John Wiley & Sons.
   - Modern approach to ANCOVA using general linear models

3. **Huitema, B. E. (2011).** *The analysis of covariance and alternatives: Statistical methods for experiments, quasi-experiments, and single-case studies* (2nd ed.). John Wiley & Sons.
   - In-depth coverage of ANCOVA assumptions and alternatives

### Post-Pre Design Analysis

4. **Vickers, A. J., & Altman, D. G. (2001).** Analysing controlled trials with baseline and follow up measurements. *BMJ*, 323(7321), 1123-1124.
   - Best practices for analyzing change scores with baseline adjustment

5. **Senn, S. (2006).** Change from baseline and analysis of covariance revisited. *Statistics in Medicine*, 25(24), 4334-4344.
   - Statistical properties of change score analysis

### Bayesian Methods

6. **Kruschke, J. K. (2015).** *Doing Bayesian data analysis: A tutorial with R, JAGS, and Stan* (2nd ed.). Academic Press.
   - Practical guide to Bayesian analysis including ANCOVA-like models

7. **Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013).** *Bayesian data analysis* (3rd ed.). Chapman and Hall/CRC.
   - Comprehensive reference for Bayesian statistical methods

8. **McElreath, R. (2020).** *Statistical rethinking: A Bayesian course with examples in R and Stan* (2nd ed.). CRC Press.
   - Modern introduction to Bayesian inference with practical examples

### Multiple Comparisons

9. **Tukey, J. W. (1949).** Comparing individual means in the analysis of variance. *Biometrics*, 5(2), 99-114.
   - Original presentation of Tukey's HSD test

10. **Kruschke, J. K. (2013).** Bayesian estimation supersedes the t test. *Journal of Experimental Psychology: General*, 142(2), 573-603.
    - Bayesian approach to group comparisons

## API Reference

### ANCOVA Class

```python
class ANCOVA:
    def __init__(self, ss_type: Literal[1, 2, 3] = 2)
    
    def fit(
        self,
        dependent_var: NDArray[np.float64],
        covariates: Dict[str, Tuple[NDArray, Literal["Q", "C", "G"]]],
        alpha: float = 0.05,
    ) -> ANCOVAResult
    
    def fit_postpre(
        self,
        pre_scores: NDArray[np.float64],
        post_scores: NDArray[np.float64],
        groups: NDArray[np.str_],
        additional_covariates: Optional[Dict[str, Covariate]] = None,
        alpha: float = 0.05,
    ) -> ANCOVAResult
```

### BayesianANCOVA Class

```python
class BayesianANCOVA:
    def __init__(
        self,
        hypothesis_type: Literal["two-sided", "one-sided"] = "two-sided",
        mcmc_samples: int = 2000,
        mcmc_tune: int = 1000,
        mcmc_chains: int = 4,
        random_seed: Optional[int] = None,
    )
    
    def fit(
        self,
        dependent_var: NDArray[np.float64],
        covariates: Dict[str, Tuple[NDArray, Literal["Q", "C", "G"]]],
        hdi_prob: float = 0.95,
        rope: Optional[Tuple[float, float]] = None,
    ) -> BayesianANCOVAResult
    
    def fit_postpre(
        self,
        pre_scores: NDArray[np.float64],
        post_scores: NDArray[np.float64],
        groups: NDArray[np.str_],
        additional_covariates: Optional[Dict[str, Covariate]] = None,
        hdi_prob: float = 0.95,
        rope: Optional[Tuple[float, float]] = None,
    ) -> BayesianANCOVAResult
```

## Development

### Setting up development environment

```bash
# Clone repository
git clone https://github.com/konbraphat51/ancovapy.git
cd ancovapy

# Install uv (if not already installed)
pip install uv

# Install package with development dependencies
uv pip install -e ".[dev]"
```

### Running tests

```bash
pytest tests/
```

### Type checking

```bash
mypy ancovapy/
```

### Code formatting

```bash
black ancovapy/ tests/
ruff ancovapy/ tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use ancovapy in your research, please cite:

```bibtex
@software{ancovapy,
  title = {ancovapy: A Python library for ANCOVA and Bayesian ANCOVA},
  author = {ancovapy contributors},
  year = {2026},
  url = {https://github.com/konbraphat51/ancovapy}
}
```

## Support

For bugs and feature requests, please open an issue on [GitHub](https://github.com/konbraphat51/ancovapy/issues).
