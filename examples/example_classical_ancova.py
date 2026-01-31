"""
Example: Classical ANCOVA with multiple covariate types.

This example demonstrates:
- Using quantitative covariates (continuous variables)
- Using categorical covariates with group comparisons
- Post-hoc tests for group differences
"""

import numpy as np

from ancovapy import ANCOVA

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_per_group = 30
n_groups = 3

# Create groups
groups = np.repeat(["Control", "Treatment_A", "Treatment_B"], n_per_group)

# Generate age (quantitative covariate)
age = np.random.normal(45, 10, n_per_group * n_groups)

# Generate outcome with treatment effects
outcome = np.zeros(n_per_group * n_groups)
outcome[groups == "Control"] = np.random.normal(100, 15, n_per_group)
outcome[groups == "Treatment_A"] = np.random.normal(110, 15, n_per_group)
outcome[groups == "Treatment_B"] = np.random.normal(115, 15, n_per_group)

# Add age effect
outcome += 0.5 * (age - 45)

# Add some noise
outcome += np.random.normal(0, 5, len(outcome))

# Define covariates
covariates = {
    "age": (age, "Q"),  # Quantitative covariate
    "treatment": (groups, "G"),  # Group covariate (requires post-hoc tests)
}

# Fit ANCOVA model
print("=" * 60)
print("Classical ANCOVA Example")
print("=" * 60)

ancova = ANCOVA(ss_type=2)  # Type II Sum of Squares
result = ancova.fit(outcome, covariates)

# Print overall model statistics
print("\nModel Statistics:")
print(f"  F-statistic: {result.f_statistic:.4f}")
print(f"  p-value: {result.p_value:.4e}")
print(f"  R²: {result.r_squared:.4f}")
print(f"  Adjusted R²: {result.adj_r_squared:.4f}")

# Print covariate statistics
print("\nCovariate Effects:")
for stat in result.covariate_stats:
    print(f"\n  {stat.name} ({stat.covariate_type}):")
    print(f"    Coefficient: {stat.coefficient:.4f}")
    print(f"    Std Error: {stat.std_error:.4f}")
    print(f"    t-value: {stat.t_value:.4f}")
    print(f"    p-value: {stat.p_value:.4e}")
    print(f"    95% CI: [{stat.ci_lower:.4f}, {stat.ci_upper:.4f}]")

# Print post-hoc test results
if result.posthoc_results:
    print("\nPost-hoc Tests (Tukey HSD):")
    for ph in result.posthoc_results:
        sig = "*" if ph.reject else ""
        print(f"\n  {ph.group1} vs {ph.group2}:")
        print(f"    Mean difference: {ph.mean_diff:.4f} {sig}")
        print(f"    p-value: {ph.p_value:.4e}")
        print(f"    95% CI: [{ph.ci_lower:.4f}, {ph.ci_upper:.4f}]")
        print(f"    Significant: {ph.reject}")

print("\n" + "=" * 60)
