"""
Example: Bayesian ANCOVA with MCMC sampling.

This example demonstrates:
- Bayesian inference with MCMC
- Credible intervals (HDI)
- Probability statements about effects
- MCMC diagnostics
"""

import numpy as np
from ancovapy import BayesianANCOVA

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_per_group = 25
n_groups = 3

# Create groups
groups = np.repeat(["Control", "Treatment_A", "Treatment_B"], n_per_group)

# Generate age (quantitative covariate)
age = np.random.normal(45, 10, n_per_group * n_groups)

# Generate outcome with treatment effects
outcome = np.zeros(n_per_group * n_groups)
outcome[groups == "Control"] = np.random.normal(100, 15, n_per_group)
outcome[groups == "Treatment_A"] = np.random.normal(110, 15, n_per_group)
outcome[groups == "Treatment_B"] = np.random.normal(118, 15, n_per_group)

# Add age effect
outcome += 0.5 * (age - 45)

# Add some noise
outcome += np.random.normal(0, 5, len(outcome))

# Define covariates
covariates = {
    "age": (age, "Q"),          # Quantitative covariate
    "treatment": (groups, "G"),  # Group covariate
}

# Fit Bayesian ANCOVA model
print("=" * 60)
print("Bayesian ANCOVA Example")
print("=" * 60)

bayesian_ancova = BayesianANCOVA(
    hypothesis_type="two-sided",
    mcmc_samples=2000,
    mcmc_tune=1000,
    mcmc_chains=4,
    random_seed=42,
)

print("\nFitting Bayesian model (this may take a minute)...")
result = bayesian_ancova.fit(
    outcome,
    covariates,
    hdi_prob=0.95,
    rope=(-2, 2),  # Region of Practical Equivalence
)

# Print MCMC diagnostics
print("\nMCMC Diagnostics:")
print(f"  Max R-hat: {result.rhat_max:.4f} (should be < 1.01)")
print(f"  Min Bulk ESS: {result.ess_bulk_min:.0f} (should be > 400)")
print(f"  Min Tail ESS: {result.ess_tail_min:.0f} (should be > 400)")
print(f"  Divergences: {result.divergences} (should be 0)")

if result.rhat_max > 1.01:
    print("\n  ⚠ Warning: R-hat > 1.01 indicates poor convergence")
if result.ess_bulk_min < 400:
    print("\n  ⚠ Warning: Low effective sample size")
if result.divergences > 0:
    print(f"\n  ⚠ Warning: {result.divergences} divergences detected")

# Print covariate statistics
print("\nCovariate Effects (Posterior Statistics):")
for stat in result.covariate_stats:
    print(f"\n  {stat.name} ({stat.covariate_type}):")
    print(f"    Mean: {stat.mean:.4f}")
    print(f"    Std: {stat.std:.4f}")
    print(f"    95% HDI: [{stat.hdi_lower:.4f}, {stat.hdi_upper:.4f}]")
    print(f"    P(β > 0): {stat.prob_positive:.4f}")
    print(f"    P(β < 0): {stat.prob_negative:.4f}")

# Print group comparisons
if result.group_comparisons:
    print("\nGroup Comparisons (Bayesian):")
    for comp in result.group_comparisons:
        print(f"\n  {comp.group1} vs {comp.group2}:")
        print(f"    Mean difference: {comp.mean_diff:.4f}")
        print(f"    95% HDI: [{comp.hdi_lower:.4f}, {comp.hdi_upper:.4f}]")
        print(f"    P({comp.group1} > {comp.group2}): {comp.prob_greater:.4f}")
        if comp.rope_decision:
            print(f"    ROPE decision: {comp.rope_decision}")

print("\n" + "=" * 60)
