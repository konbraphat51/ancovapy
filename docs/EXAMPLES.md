# Usage Examples

## Classical ANCOVA

### Basic Usage

```python
import numpy as np
from ancovapy import ANCOVA

# Generate example data
n = 90
groups = np.repeat(["Control", "Treatment_A", "Treatment_B"], n // 3)
age = np.random.normal(45, 10, n)
outcome = np.random.normal(100, 15, n)
outcome[groups == "Treatment_A"] += 10
outcome[groups == "Treatment_B"] += 15
outcome += 0.5 * (age - 45)

# Define covariates
covariates = {
    "age": (age, "Q"),          # Quantitative covariate
    "treatment": (groups, "G"),  # Group covariate (requires post-hoc tests)
}

# Fit ANCOVA model
ancova = ANCOVA(ss_type=2)  # Type II Sum of Squares
result = ancova.fit(outcome, covariates)

# View results
print(f"F-statistic: {result.f_statistic:.4f}")
print(f"p-value: {result.p_value:.4e}")
print(f"R²: {result.r_squared:.4f}")

# Covariate effects
for stat in result.covariate_stats:
    print(f"\n{stat.name} ({stat.covariate_type}):")
    print(f"  Coefficient: {stat.coefficient:.4f}")
    print(f"  p-value: {stat.p_value:.4e}")

# Post-hoc tests
if result.posthoc_results:
    print("\nPost-hoc Tests (Tukey HSD):")
    for ph in result.posthoc_results:
        print(f"  {ph.group1} vs {ph.group2}: diff={ph.mean_diff:.2f}, p={ph.p_value:.4f}")
```

## Bayesian ANCOVA

### Basic Usage

```python
from ancovapy import BayesianANCOVA

# Use same data as above
covariates = {
    "age": (age, "Q"),
    "treatment": (groups, "G"),
}

# Fit Bayesian ANCOVA
bayesian = BayesianANCOVA(
    hypothesis_type="two-sided",
    mcmc_samples=2000,
    mcmc_chains=4,
    random_seed=42,
)

result = bayesian.fit(
    outcome,
    covariates,
    hdi_prob=0.95,
    rope=(-2, 2),  # Region of Practical Equivalence
)

# Check diagnostics
print(f"Max R-hat: {result.rhat_max:.4f} (should be < 1.01)")
print(f"Min ESS: {result.ess_bulk_min:.0f} (should be > 400)")
print(f"Divergences: {result.divergences}")

# Covariate effects
for stat in result.covariate_stats:
    print(f"\n{stat.name}:")
    print(f"  Mean: {stat.mean:.3f}")
    print(f"  95% HDI: [{stat.hdi_lower:.3f}, {stat.hdi_upper:.3f}]")
    print(f"  P(β > 0): {stat.prob_positive:.3f}")

# Group comparisons
if result.group_comparisons:
    print("\nGroup Comparisons:")
    for comp in result.group_comparisons:
        print(f"  {comp.group1} vs {comp.group2}:")
        print(f"    Mean diff: {comp.mean_diff:.3f}")
        print(f"    95% HDI: [{comp.hdi_lower:.3f}, {comp.hdi_upper:.3f}]")
        if comp.rope_decision:
            print(f"    ROPE: {comp.rope_decision}")
```

## Post-Pre Design

### Classical Approach

```python
# Generate pre-post data
n = 60
groups = np.repeat(["Control", "Treatment"], n // 2)
pre_scores = np.random.normal(100, 15, n)
post_scores = pre_scores + np.random.normal(5, 10, n)
post_scores[groups == "Treatment"] += 8

# Fit post-pre ANCOVA
ancova = ANCOVA(ss_type=2)
result = ancova.fit_postpre(pre_scores, post_scores, groups)

# Adjusted means
print("Adjusted Mean Changes:")
for group, mean in result.adjusted_means.items():
    print(f"  {group}: {mean:.3f}")

# Pairwise differences
print("\nPairwise Differences:")
for (g1, g2), diff in result.adjusted_mean_diffs.items():
    ci = result.credible_intervals[(g1, g2)]
    print(f"  {g1} - {g2}: {diff:.3f} (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])")
```

### Bayesian Approach

```python
# Fit Bayesian post-pre ANCOVA
bayesian = BayesianANCOVA(mcmc_samples=2000, random_seed=42)
result = bayesian.fit_postpre(pre_scores, post_scores, groups)

# Adjusted means with credible intervals
print("Adjusted Mean Changes (with 95% CrI):")
for group, (mean, hdi_low, hdi_high) in result.adjusted_means.items():
    print(f"  {group}: {mean:.3f} [{hdi_low:.3f}, {hdi_high:.3f}]")

# Pairwise differences
print("\nPairwise Differences:")
for (g1, g2), diff in result.adjusted_mean_diffs.items():
    ci = result.credible_intervals_95[(g1, g2)]
    print(f"  {g1} - {g2}: {diff:.3f} (95% CrI: [{ci[0]:.3f}, {ci[1]:.3f}])")
```

## Advanced Features

### Multiple Covariate Types

```python
# Combine different covariate types
covariates = {
    "age": (age_data, "Q"),           # Quantitative
    "baseline": (baseline_data, "Q"),  # Quantitative
    "treatment": (treatment_groups, "G"),  # Group (with post-hoc)
    "site": (study_sites, "C"),        # Categorical (no post-hoc)
    "sex": (sex_data, "C"),            # Categorical (no post-hoc)
}

result = ancova.fit(outcome, covariates)
```

### Configuring Sum of Squares Type

```python
# Type I: Sequential
ancova_type1 = ANCOVA(ss_type=1)

# Type II: Hierarchical (default, for balanced designs)
ancova_type2 = ANCOVA(ss_type=2)

# Type III: Marginal (for unbalanced designs)
ancova_type3 = ANCOVA(ss_type=3)
```

### MCMC Configuration

```python
# Customize MCMC parameters
bayesian = BayesianANCOVA(
    hypothesis_type="one-sided",  # or "two-sided"
    mcmc_samples=5000,  # More samples for better accuracy
    mcmc_tune=2000,     # More tuning for complex models
    mcmc_chains=6,      # More chains for better convergence
    random_seed=42,     # For reproducibility
)
```

For more examples, see the [examples/](../examples/) directory.
