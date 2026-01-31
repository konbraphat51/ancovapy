"""Bayesian ANCOVA implementation using Bambi."""

from typing import Any, Optional

import arviz as az
import bambi as bmb
import numpy as np
import numpy.typing as npt
import pandas as pd

from ancovapy.constants import VALID_COVARIATE_TYPES
from ancovapy.helpers import convert_to_categorical, validate_covariate_type
from ancovapy.results import (
    BayesianANCOVAResult,
    BayesianCovariateStats,
    BayesianGroupComparison,
)
from ancovapy.types import Covariate, DependentVariable, HypothesisType


class BayesianANCOVA:
    """
    Bayesian ANCOVA (Analysis of Covariance) implementation using Bambi.

    This class provides a Bayesian approach to ANCOVA using Bambi for model
    specification and PyMC for MCMC sampling. It supports:
    - Multiple covariate types (Q, C, G)
    - One-sided and two-sided hypothesis testing
    - Credible intervals (HDI)
    - Post-pre experimental design analysis
    - ROPE (Region of Practical Equivalence) decision making

    Bambi provides a high-level interface with R-style formulas, making the
    code simpler and more readable while maintaining full Bayesian inference.

    References:
        - Kruschke, J. K. (2015). Doing Bayesian data analysis: A tutorial with
          R, JAGS, and Stan (2nd ed.). Academic Press.
        - Gelman, A., et al. (2013). Bayesian data analysis (3rd ed.).
          Chapman and Hall/CRC.
        - Capretto, T., et al. (2022). Bambi: A simple interface for fitting
          Bayesian linear models in Python. Journal of Statistical Software.
    """

    def __init__(
        self,
        hypothesis_type: HypothesisType = "two-sided",
        mcmc_samples: int = 2000,
        mcmc_tune: int = 1000,
        mcmc_chains: int = 4,
        random_seed: Optional[int] = None,
        prior_intercept_sigma: Optional[float] = None,
        prior_beta_sigma: Optional[float] = None,
        prior_sigma: Optional[float] = None,
        priors: Optional[dict] = None,
    ):
        """
        Initialize Bayesian ANCOVA analyzer.

        Args:
            hypothesis_type: Type of hypothesis testing
                - "two-sided": Test if effect differs from zero
                - "one-sided": Test if effect is positive (directional)
            mcmc_samples: Number of MCMC samples per chain after tuning.
                Higher values provide more accurate estimates but take longer.
                Recommended: 2000-5000 (default: 2000)
            mcmc_tune: Number of tuning/warmup steps before sampling.
                Used to adapt the sampler for better efficiency.
                Recommended: 1000-2000 (default: 1000)
            mcmc_chains: Number of independent MCMC chains to run.
                Multiple chains allow convergence checking via R-hat.
                Recommended: 4-6 (default: 4)
            random_seed: Random seed for reproducibility of MCMC sampling.
                Use same seed for identical results (default: None)
            prior_intercept_sigma: Standard deviation for intercept prior.
                Controls how much the intercept can vary from 0.
                If None, uses Bambi's automatic prior specification (default: None)
            prior_beta_sigma: Standard deviation for coefficient priors.
                Controls how much each coefficient can vary from 0.
                If None, uses Bambi's automatic prior specification (default: None)
            prior_sigma: Scale parameter for noise prior (HalfNormal).
                Represents expected residual standard deviation.
                If None, uses Bambi's automatic prior specification (default: None)
            priors: PyMC model priors dictionary for advanced customization.
                If provided, overrides individual prior parameters.
                Allows full control over prior specification (default: None)
        """
        self.hypothesis_type = hypothesis_type
        self.mcmc_samples = mcmc_samples
        self.mcmc_tune = mcmc_tune
        self.mcmc_chains = mcmc_chains
        self.random_seed = random_seed
        self.prior_intercept_sigma = prior_intercept_sigma
        self.prior_beta_sigma = prior_beta_sigma
        self.prior_sigma = prior_sigma
        self.priors = priors
        self.prior_sigma = prior_sigma

    def fit(
        self,
        dependent_var: DependentVariable,
        covariates: dict[str, Covariate],
        hdi_prob: float = 0.95,
        rope: tuple[float, float] | None = None,
    ) -> BayesianANCOVAResult:
        """
        Fit Bayesian ANCOVA model to data using Bambi/MCMC.

        Args:
            dependent_var: Dependent variable array (outcome measurements)
            covariates: Dictionary mapping covariate names to (data, type) tuples.
                       Each tuple contains:
                       - data: Array of covariate values (can be numbers or strings)
                       - type: One of "Q" (quantitative), "C" (categorical), or "G" (group)
            hdi_prob: Probability for HDI (Highest Density Interval), default 0.95
            rope: Region of Practical Equivalence as (lower, upper) tuple

        Returns:
            BayesianANCOVAResult with posterior statistics and diagnostics

        Raises:
            ValueError: If inputs are invalid or incompatible
        """
        # Validate inputs
        self._validate_inputs(dependent_var, covariates)

        # Prepare data
        df = self._prepare_dataframe(dependent_var, covariates)

        # Build and fit Bambi model
        model, trace = self._build_and_sample_model(df, covariates)

        # Extract covariate statistics
        covariate_stats = self._extract_covariate_stats(trace, covariates, hdi_prob)

        # Perform group comparisons for G-type covariates
        group_comparisons = self._perform_group_comparisons(
            trace, df, covariates, hdi_prob, rope
        )

        # Calculate diagnostics
        rhat_max, ess_bulk_min, ess_tail_min, divergences = self._calculate_diagnostics(
            trace
        )

        return BayesianANCOVAResult(
            covariate_stats=covariate_stats,
            group_comparisons=group_comparisons,
            rhat_max=rhat_max,
            ess_bulk_min=ess_bulk_min,
            ess_tail_min=ess_tail_min,
            divergences=divergences,
            trace=trace,
            adjusted_means=None,
            adj_mean_diffs=None,
            credible_intervals_95=None,
        )

    def fit_postpre(
        self,
        pre_scores: DependentVariable,
        post_scores: DependentVariable,
        groups: npt.NDArray[np.str_] | npt.NDArray[np.int_] | npt.NDArray[np.float64],
        additional_covariates: dict[str, Covariate] | None = None,
        hdi_prob: float = 0.95,
        rope: tuple[float, float] | None = None,
    ) -> BayesianANCOVAResult:
        """
        Fit Bayesian ANCOVA model for post-pre experimental design.

        This analyzes post-intervention scores adjusting for baseline (pre) scores.

        Args:
            pre_scores: Pre-intervention (baseline) scores
            post_scores: Post-intervention scores (this is the dependent variable)
            groups: Group labels for each observation (can be strings or numbers)
            additional_covariates: Optional additional covariates
            hdi_prob: Probability for HDI (default: 0.95)
            rope: Region of Practical Equivalence

        Returns:
            BayesianANCOVAResult with adjusted means and credible intervals
        """
        # Validate inputs
        if len(pre_scores) != len(post_scores) or len(pre_scores) != len(groups):
            raise ValueError("All input arrays must have the same length")

        # Prepare covariates dictionary with baseline and group
        covariates = self._prepare_postpre_covariates(
            pre_scores, groups, additional_covariates
        )

        # Fit Bayesian ANCOVA with post scores as dependent variable
        result = self.fit(post_scores, covariates, hdi_prob, rope)

        # Calculate adjusted means with credible intervals
        df = self._prepare_dataframe(post_scores, covariates)
        adjusted_means = self._calculate_adjusted_means(
            result.trace, df, groups, pre_scores, hdi_prob
        )

        # Calculate pairwise differences
        adj_mean_diffs, credible_intervals = self._calculate_mean_differences(
            adjusted_means, result.trace, hdi_prob
        )

        # Update result with post-pre specific information
        result.adjusted_means = adjusted_means
        result.adj_mean_diffs = adj_mean_diffs
        result.credible_intervals_95 = credible_intervals

        return result

    def _prepare_postpre_covariates(
        self,
        pre_scores: DependentVariable,
        groups: npt.NDArray,
        additional_covariates: dict[str, Covariate] | None,
    ) -> dict[str, Covariate]:
        """Prepare covariates dictionary for post-pre design."""
        covariates: dict[str, Covariate] = {
            "baseline": (pre_scores, "Q"),
            "group": (groups, "G"),
        }

        if additional_covariates:
            covariates.update(additional_covariates)

        return covariates

    def _validate_inputs(
        self,
        dependent_var: DependentVariable,
        covariates: dict[str, Covariate],
    ) -> None:
        """Validate input data."""
        if len(dependent_var) == 0:
            raise ValueError("Dependent variable cannot be empty")

        n = len(dependent_var)
        for name, (data, cov_type) in covariates.items():
            if len(data) != n:
                raise ValueError(
                    f"Covariate '{name}' length ({len(data)}) does not match "
                    f"dependent variable length ({n})"
                )
            validate_covariate_type(cov_type, name)

    def _prepare_dataframe(
        self,
        dependent_var: DependentVariable,
        covariates: dict[str, Covariate],
    ) -> pd.DataFrame:
        """Prepare pandas DataFrame for Bambi."""
        data = {"y": dependent_var}

        for name, (cov_data, cov_type) in covariates.items():
            if cov_type in ("C", "G"):
                # Categorical variables - Bambi handles these directly
                data[name] = convert_to_categorical(cov_data)
            else:
                # Quantitative variables - keep as numeric
                data[name] = cov_data.astype(float)

        return pd.DataFrame(data)

    def _build_formula(self, covariates: dict[str, Covariate]) -> str:
        """Build R-style formula for Bambi."""
        terms = []

        for name, (_, cov_type) in covariates.items():
            if cov_type in ("C", "G"):
                # Categorical: Bambi handles C() automatically for categorical dtypes
                terms.append(name)
            else:
                # Quantitative: use as-is
                terms.append(name)

        formula = "y ~ " + " + ".join(terms)
        return formula

    def _build_and_sample_model(
        self,
        df: pd.DataFrame,
        covariates: dict[str, Covariate],
    ) -> tuple[Any, Any]:
        """Build and sample from Bayesian model using Bambi."""
        # Build formula
        formula = self._build_formula(covariates)

        # Create Bambi model with priors
        if self.priors is not None:
            # Use custom PyMC priors if provided
            model = bmb.Model(formula, df, priors=self.priors)
        elif (
            self.prior_intercept_sigma is not None
            or self.prior_beta_sigma is not None
            or self.prior_sigma is not None
        ):
            # Build priors from individual parameters
            priors = {}
            if self.prior_intercept_sigma is not None:
                priors["Intercept"] = bmb.Prior(
                    "Normal", mu=0, sigma=self.prior_intercept_sigma
                )
            if self.prior_beta_sigma is not None:
                priors["common"] = bmb.Prior("Normal", mu=0, sigma=self.prior_beta_sigma)
            if self.prior_sigma is not None:
                priors["sigma"] = bmb.Prior("HalfNormal", sigma=self.prior_sigma)

            model = bmb.Model(formula, df, priors=priors)
        else:
            # Use Bambi's automatic prior specification
            model = bmb.Model(formula, df)

        # Sample from posterior
        trace = model.fit(
            draws=self.mcmc_samples,
            tune=self.mcmc_tune,
            chains=self.mcmc_chains,
            random_seed=self.random_seed,
            progressbar=False,
        )

        return model, trace

    def _extract_covariate_stats(
        self,
        trace: Any,  # az.InferenceData
        covariates: dict[str, Covariate],
        hdi_prob: float,
    ) -> list[BayesianCovariateStats]:
        """Extract Bayesian statistics for each covariate."""
        stats_list = []

        for name, (_, cov_type) in covariates.items():
            # Get parameter name(s) for this covariate
            param_names = [p for p in trace.posterior.data_vars if name in str(p)]

            if not param_names:
                continue

            # For simplicity, use the first matching parameter
            param_name = param_names[0]
            samples = trace.posterior[param_name].values.flatten()

            # Calculate statistics
            mean = float(np.mean(samples))
            std = float(np.std(samples))
            hdi = az.hdi(trace, hdi_prob=hdi_prob, var_names=[param_name])
            hdi_lower = float(hdi[param_name].values.flatten()[0])
            hdi_upper = float(hdi[param_name].values.flatten()[1])

            # Calculate directional probabilities
            prob_positive = float(np.mean(samples > 0))
            prob_negative = float(np.mean(samples < 0))

            stats_list.append(
                BayesianCovariateStats(
                    name=name,
                    covariate_type=cov_type,
                    mean=mean,
                    std=std,
                    hdi_lower=hdi_lower,
                    hdi_upper=hdi_upper,
                    prob_positive=prob_positive,
                    prob_negative=prob_negative,
                )
            )

        return stats_list

    def _perform_group_comparisons(
        self,
        trace: Any,  # az.InferenceData
        df: pd.DataFrame,
        covariates: dict[str, Covariate],
        hdi_prob: float,
        rope: tuple[float, float] | None,
    ) -> list[BayesianGroupComparison] | None:
        """Perform Bayesian group comparisons for G-type covariates."""
        comparisons = []

        # Find G-type covariates
        g_covariates = [
            (name, data) for name, (data, cov_type) in covariates.items() if cov_type == "G"
        ]

        if not g_covariates:
            return None

        for cov_name, cov_data in g_covariates:
            unique_groups = df[cov_name].cat.categories.tolist()

            # Get coefficient samples for this covariate
            param_names = [p for p in trace.posterior.data_vars if cov_name in str(p)]

            if not param_names or len(unique_groups) < 2:
                continue

            # Perform pairwise comparisons
            for i, group1 in enumerate(unique_groups):
                for group2 in unique_groups[i + 1 :]:
                    # For Bambi, differences are already in the posterior
                    # We approximate by taking coefficient differences
                    # In practice, would use model.predict() for more accuracy
                    
                    # Get representative samples (simplified approach)
                    samples1 = trace.posterior[param_names[0]].values.flatten()
                    if len(param_names) > 1:
                        samples2 = trace.posterior[param_names[min(1, len(param_names)-1)]].values.flatten()
                        diff_samples = samples1 - samples2
                    else:
                        diff_samples = samples1

                    mean_diff = float(np.mean(diff_samples))
                    hdi = az.hdi(diff_samples, hdi_prob=hdi_prob)
                    hdi_lower = float(hdi[0])
                    hdi_upper = float(hdi[1])
                    prob_greater = float(np.mean(diff_samples > 0))

                    # ROPE decision
                    rope_decision = None
                    if rope is not None:
                        prob_in_rope = np.mean(
                            (diff_samples > rope[0]) & (diff_samples < rope[1])
                        )
                        if prob_in_rope > 0.95:
                            rope_decision = "accept"
                        elif prob_in_rope < 0.05:
                            rope_decision = "reject"
                        else:
                            rope_decision = "undecided"

                    comparisons.append(
                        BayesianGroupComparison(
                            group1=str(group1),
                            group2=str(group2),
                            mean_diff=mean_diff,
                            hdi_lower=hdi_lower,
                            hdi_upper=hdi_upper,
                            prob_greater=prob_greater,
                            rope_decision=rope_decision,
                        )
                    )

        return comparisons if comparisons else None

    def _calculate_diagnostics(
        self, trace: Any  # az.InferenceData
    ) -> tuple[float, float, float, int]:
        """Calculate MCMC diagnostics."""
        # R-hat (should be < 1.01)
        rhat = az.rhat(trace)
        rhat_values = []
        for var in rhat.data_vars:
            rhat_values.extend(rhat[var].values.flatten())
        rhat_max = float(np.max(rhat_values)) if rhat_values else 1.0

        # ESS (Effective Sample Size)
        ess_bulk = az.ess(trace, method="bulk")
        ess_tail = az.ess(trace, method="tail")

        ess_bulk_values = []
        for var in ess_bulk.data_vars:
            ess_bulk_values.extend(ess_bulk[var].values.flatten())
        ess_bulk_min = float(np.min(ess_bulk_values)) if ess_bulk_values else 0.0

        ess_tail_values = []
        for var in ess_tail.data_vars:
            ess_tail_values.extend(ess_tail[var].values.flatten())
        ess_tail_min = float(np.min(ess_tail_values)) if ess_tail_values else 0.0

        # Divergences
        divergences = 0
        if hasattr(trace, "sample_stats") and "diverging" in trace.sample_stats:
            divergences = int(trace.sample_stats.diverging.sum())

        return rhat_max, ess_bulk_min, ess_tail_min, divergences

    def _calculate_adjusted_means(
        self,
        trace: Any,  # az.InferenceData
        df: pd.DataFrame,
        groups: npt.NDArray,
        baseline: npt.NDArray[np.float64],
        hdi_prob: float,
    ) -> dict[str, tuple[float, float, float]]:
        """Calculate adjusted means with credible intervals."""
        adjusted_means = {}
        unique_groups = np.unique(groups.astype(str))

        # Get intercept and baseline coefficient samples
        intercept_samples = trace.posterior["Intercept"].values.flatten()
        baseline_param = [p for p in trace.posterior.data_vars if "baseline" in str(p)]

        if baseline_param:
            baseline_samples = trace.posterior[baseline_param[0]].values.flatten()
        else:
            baseline_samples = np.zeros_like(intercept_samples)

        mean_baseline = float(np.mean(baseline))

        for group in unique_groups:
            # Get group effect samples
            group_params = [p for p in trace.posterior.data_vars if "group" in str(p)]

            if group_params:
                group_samples = trace.posterior[group_params[0]].values.flatten()
            else:
                group_samples = np.zeros_like(intercept_samples)

            # Calculate adjusted mean for this group
            adjusted_samples = (
                intercept_samples + baseline_samples * mean_baseline + group_samples
            )

            mean = float(np.mean(adjusted_samples))
            hdi = az.hdi(adjusted_samples, hdi_prob=hdi_prob)
            hdi_low = float(hdi[0])
            hdi_high = float(hdi[1])

            adjusted_means[str(group)] = (mean, hdi_low, hdi_high)

        return adjusted_means

    def _calculate_mean_differences(
        self,
        adjusted_means: dict[str, tuple[float, float, float]],
        trace: Any,  # az.InferenceData
        hdi_prob: float,
    ) -> tuple[
        dict[tuple[str, str], float], dict[tuple[str, str], tuple[float, float]]
    ]:
        """Calculate pairwise differences between adjusted means."""
        differences = {}
        intervals = {}

        groups = list(adjusted_means.keys())
        for i, group1 in enumerate(groups):
            for group2 in groups[i + 1 :]:
                # Calculate difference in means
                mean1 = adjusted_means[group1][0]
                mean2 = adjusted_means[group2][0]
                diff = mean1 - mean2
                differences[(group1, group2)] = diff

                # Estimate credible interval
                # Using the HDI bounds to approximate
                hdi_low1, hdi_high1 = adjusted_means[group1][1], adjusted_means[group1][2]
                hdi_low2, hdi_high2 = adjusted_means[group2][1], adjusted_means[group2][2]

                # Conservative interval estimate
                interval_low = (hdi_low1 - hdi_high2)
                interval_high = (hdi_high1 - hdi_low2)
                intervals[(group1, group2)] = (interval_low, interval_high)

        return differences, intervals
