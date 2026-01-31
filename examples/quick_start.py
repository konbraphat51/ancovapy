"""
Quick start example for ancovapy.

This example shows basic usage of both classical and Bayesian ANCOVA.
"""

import numpy as np

from ancovapy import ANCOVA, BayesianANCOVA

# Set random seed for reproducibility
np.random.seed(42)

# Generate simple example data
n = 60
groups = np.repeat(["Control", "Treatment"], n // 2)
age = np.random.normal(50, 10, n)
outcome = 100 + 0.5 * age + 10 * (groups == "Treatment") + np.random.normal(0, 5, n)

# Prepare covariates
covariates = {
    "age": (age, "Q"),  # Quantitative covariate
    "group": (groups, "G"),  # Group covariate (with post-hoc tests)
}

print("=" * 60)
print("Classical ANCOVA - Quick Example")
print("=" * 60)

# Fit classical ANCOVA
ancova = ANCOVA(ss_type=2)
result = ancova.fit(outcome, covariates)

print(f"\nF-statistic: {result.f_statistic:.4f}, p-value: {result.p_value:.4e}")
print(f"R²: {result.r_squared:.4f}")

for stat in result.covariate_stats:
    print(f"\n{stat.name}: β={stat.coefficient:.3f}, p={stat.p_value:.4f}")

if result.posthoc_results:
    print("\nPost-hoc comparisons:")
    for ph in result.posthoc_results:
        print(
            f"  {ph.group1} vs {ph.group2}: "
            f"diff={ph.mean_diff:.2f}, p={ph.p_value:.4f}"
        )

print("\n" + "=" * 60)
print("Bayesian ANCOVA - Quick Example")
print("=" * 60)

# Fit Bayesian ANCOVA (fewer samples for quick demo)
bayesian = BayesianANCOVA(mcmc_samples=500, mcmc_tune=250, random_seed=42)
result_bayes = bayesian.fit(outcome, covariates)

print(f"\nDiagnostics: R-hat={result_bayes.rhat_max:.4f}, ")
print(f"ESS={result_bayes.ess_bulk_min:.0f}")

for stat in result_bayes.covariate_stats:
    print(
        f"\n{stat.name}: mean={stat.mean:.3f}, "
        f"95% HDI=[{stat.hdi_lower:.3f}, {stat.hdi_upper:.3f}]"
    )
    print(f"  P(β>0)={stat.prob_positive:.3f}")

print("\n" + "=" * 60)
