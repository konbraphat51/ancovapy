"""
Example: Post-Pre design analysis with ANCOVA.

This example demonstrates:
- Analysis of change scores
- Adjustment for baseline values
- Calculation of adjusted mean changes
- Comparison between classical and Bayesian approaches
"""

import numpy as np
from ancovapy import ANCOVA, BayesianANCOVA

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic pre-post data
n_per_group = 30

# Create groups
groups = np.repeat(["Control", "Treatment"], n_per_group)
n_total = len(groups)

# Generate pre-scores (baseline)
pre_scores = np.random.normal(100, 15, n_total)

# Generate post-scores with treatment effect
post_scores = np.zeros(n_total)
# Control: small improvement
post_scores[groups == "Control"] = pre_scores[groups == "Control"] + np.random.normal(3, 10, n_per_group)
# Treatment: larger improvement
post_scores[groups == "Treatment"] = pre_scores[groups == "Treatment"] + np.random.normal(12, 10, n_per_group)

# Calculate raw change scores
change_scores = post_scores - pre_scores

print("=" * 60)
print("Post-Pre Design Analysis Example")
print("=" * 60)

# Print descriptive statistics
print("\nDescriptive Statistics:")
for group in ["Control", "Treatment"]:
    mask = groups == group
    print(f"\n  {group}:")
    print(f"    Pre:  Mean={pre_scores[mask].mean():.2f}, SD={pre_scores[mask].std():.2f}")
    print(f"    Post: Mean={post_scores[mask].mean():.2f}, SD={post_scores[mask].std():.2f}")
    print(f"    Raw Change: Mean={change_scores[mask].mean():.2f}, SD={change_scores[mask].std():.2f}")

# Classical ANCOVA approach
print("\n" + "=" * 60)
print("Classical ANCOVA Analysis")
print("=" * 60)

ancova = ANCOVA(ss_type=2)
result_classical = ancova.fit_postpre(pre_scores, post_scores, groups)

print(f"\nOverall Model:")
print(f"  F-statistic: {result_classical.f_statistic:.4f}")
print(f"  p-value: {result_classical.p_value:.4e}")
print(f"  R²: {result_classical.r_squared:.4f}")

print("\nAdjusted Mean Changes:")
if result_classical.adjusted_means:
    for group, mean in result_classical.adjusted_means.items():
        print(f"  {group}: {mean:.4f}")

print("\nPairwise Differences:")
if result_classical.adjusted_mean_diffs:
    for (g1, g2), diff in result_classical.adjusted_mean_diffs.items():
        ci = result_classical.credible_intervals[(g1, g2)]
        print(f"  {g1} - {g2}:")
        print(f"    Difference: {diff:.4f}")
        print(f"    95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

# Bayesian ANCOVA approach
print("\n" + "=" * 60)
print("Bayesian ANCOVA Analysis")
print("=" * 60)

bayesian_ancova = BayesianANCOVA(
    hypothesis_type="two-sided",
    mcmc_samples=2000,
    mcmc_tune=1000,
    mcmc_chains=4,
    random_seed=42,
)

print("\nFitting Bayesian model (this may take a minute)...")
result_bayesian = bayesian_ancova.fit_postpre(
    pre_scores,
    post_scores,
    groups,
    hdi_prob=0.95,
)

print("\nMCMC Diagnostics:")
print(f"  Max R-hat: {result_bayesian.rhat_max:.4f}")
print(f"  Min ESS: {result_bayesian.ess_bulk_min:.0f}")
print(f"  Divergences: {result_bayesian.divergences}")

print("\nAdjusted Mean Changes (with 95% Credible Intervals):")
if result_bayesian.adjusted_means:
    for group, (mean, hdi_low, hdi_high) in result_bayesian.adjusted_means.items():
        print(f"  {group}:")
        print(f"    Mean: {mean:.4f}")
        print(f"    95% CrI: [{hdi_low:.4f}, {hdi_high:.4f}]")

print("\nPairwise Differences (with 95% Credible Intervals):")
if result_bayesian.adjusted_mean_diffs:
    for (g1, g2), diff in result_bayesian.adjusted_mean_diffs.items():
        ci = result_bayesian.credible_intervals_95[(g1, g2)]
        print(f"  {g1} - {g2}:")
        print(f"    Difference: {diff:.4f}")
        print(f"    95% CrI: [{ci[0]:.4f}, {ci[1]:.4f}]")

# Compare group effects
if result_bayesian.group_comparisons:
    print("\nBayesian Group Effect:")
    for comp in result_bayesian.group_comparisons:
        if comp.group1 == "Treatment" or comp.group2 == "Treatment":
            print(f"  P(Treatment > Control): {comp.prob_greater:.4f}")

print("\n" + "=" * 60)
print("\nInterpretation:")
print("- Adjusted means account for baseline differences between groups")
print("- Bayesian credible intervals provide probability statements")
print("- Classical confidence intervals provide frequentist guarantees")
print("=" * 60)
