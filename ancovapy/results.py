"""Result dataclasses for ANCOVA analyses."""

from dataclasses import dataclass
from typing import Any, Optional

from ancovapy.types import CovariateType


@dataclass
class PostHocResult:
    """
    Results from post-hoc pairwise comparisons.
    
    Attributes:
        group1: Name of the first group being compared
        group2: Name of the second group being compared
        mean_diff: Mean difference between groups (group1 - group2)
        p_value: P-value for the comparison (Tukey HSD adjusted)
        ci_lower: Lower bound of 95% confidence interval
        ci_upper: Upper bound of 95% confidence interval
        reject: Whether to reject the null hypothesis (no difference)
    """

    group1: str
    group2: str
    mean_diff: float
    p_value: float
    ci_lower: float
    ci_upper: float
    reject: bool


@dataclass
class CovariateStats:
    """
    Statistics for a single covariate in classical ANCOVA.
    
    Attributes:
        name: Name of the covariate
        covariate_type: Type of covariate ("Q", "C", or "G")
        coefficient: Regression coefficient (beta)
        std_error: Standard error of the coefficient
        t_value: T-statistic for testing coefficient = 0
        p_value: P-value for the t-test
        ci_lower: Lower bound of 95% confidence interval for coefficient
        ci_upper: Upper bound of 95% confidence interval for coefficient
    """

    name: str
    covariate_type: CovariateType
    coefficient: float
    std_error: float
    t_value: float
    p_value: float
    ci_lower: float
    ci_upper: float


@dataclass
class ANCOVAResult:
    """
    Results from classical ANCOVA analysis.
    
    Attributes:
        f_statistic: Overall F-statistic for the model
        p_value: P-value for the overall F-test
        r_squared: R² (proportion of variance explained)
        adj_r_squared: Adjusted R² (penalized for number of predictors)
        covariate_stats: List of statistics for each covariate
        posthoc_results: Results from post-hoc tests (for G-type covariates)
        model_summary: Full text summary from statsmodels
        adjusted_means: Adjusted means for each group (post-pre design)
        adj_mean_diffs: Pairwise differences between adjusted means
        credible_intervals: 95% confidence intervals for differences
    """

    # Overall model statistics
    f_statistic: float
    p_value: float
    r_squared: float
    adj_r_squared: float

    # Covariate statistics
    covariate_stats: list[CovariateStats]

    # Post-hoc test results (only for G-type covariates)
    posthoc_results: Optional[list[PostHocResult]]

    # Full model summary
    model_summary: str

    # For post-pre design
    adjusted_means: Optional[dict[str, float]]
    adj_mean_diffs: Optional[dict[tuple[str, str], float]]
    credible_intervals: Optional[dict[tuple[str, str], tuple[float, float]]]


@dataclass
class BayesianCovariateStats:
    """
    Bayesian statistics for a single covariate.
    
    Attributes:
        name: Name of the covariate
        covariate_type: Type of covariate ("Q", "C", or "G")
        mean: Posterior mean of the coefficient
        std: Posterior standard deviation
        hdi_lower: Lower bound of 95% Highest Density Interval
        hdi_upper: Upper bound of 95% Highest Density Interval
        prob_positive: Probability that coefficient > 0
        prob_negative: Probability that coefficient < 0
    """

    name: str
    covariate_type: CovariateType
    mean: float
    std: float
    hdi_lower: float
    hdi_upper: float
    prob_positive: float
    prob_negative: float


@dataclass
class BayesianGroupComparison:
    """
    Bayesian comparison between two groups.
    
    Attributes:
        group1: Name of the first group
        group2: Name of the second group
        mean_diff: Posterior mean of difference (group1 - group2)
        hdi_lower: Lower bound of 95% HDI for difference
        hdi_upper: Upper bound of 95% HDI for difference
        prob_greater: Probability that group1 > group2
        rope_decision: ROPE decision ("accept", "reject", or "undecided")
    """

    group1: str
    group2: str
    mean_diff: float
    hdi_lower: float
    hdi_upper: float
    prob_greater: float
    rope_decision: Optional[str]


@dataclass
class BayesianANCOVAResult:
    """
    Results from Bayesian ANCOVA analysis.
    
    Attributes:
        covariate_stats: List of Bayesian statistics for each covariate
        group_comparisons: Pairwise comparisons for G-type covariates
        rhat_max: Maximum R-hat value (convergence diagnostic, should be < 1.01)
        ess_bulk_min: Minimum bulk effective sample size (should be > 400)
        ess_tail_min: Minimum tail effective sample size (should be > 400)
        divergences: Number of divergent transitions (should be 0)
        trace: Posterior samples (ArviZ InferenceData object)
        adjusted_means: Adjusted means with credible intervals (post-pre design)
        adj_mean_diffs: Pairwise differences between adjusted means
        credible_intervals_95: 95% credible intervals for differences
    """

    # Covariate statistics
    covariate_stats: list[BayesianCovariateStats]

    # Group comparisons (for G-type covariates)
    group_comparisons: Optional[list[BayesianGroupComparison]]

    # Model diagnostics
    rhat_max: float
    ess_bulk_min: float
    ess_tail_min: float
    divergences: int

    # Posterior samples (for further analysis)
    trace: Any  # az.InferenceData

    # For post-pre design
    adjusted_means: Optional[dict[str, tuple[float, float, float]]]
    adj_mean_diffs: Optional[dict[tuple[str, str], float]]
    credible_intervals_95: Optional[dict[tuple[str, str], tuple[float, float]]]
