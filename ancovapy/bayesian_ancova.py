"""Bayesian ANCOVA implementation using PyMC."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import arviz as az

from ancovapy.types import Covariate, CovariateType, DependentVariable, HypothesisType


@dataclass
class BayesianCovariateStats:
    """Bayesian statistics for a single covariate."""

    name: str
    covariate_type: CovariateType
    mean: float
    std: float
    hdi_lower: float
    hdi_upper: float
    prob_positive: float  # P(coefficient > 0)
    prob_negative: float  # P(coefficient < 0)


@dataclass
class BayesianGroupComparison:
    """Bayesian comparison between two groups."""

    group1: str
    group2: str
    mean_diff: float
    hdi_lower: float
    hdi_upper: float
    prob_greater: float  # P(group1 > group2)
    rope_decision: Optional[str]  # "accept", "reject", or "undecided"


@dataclass
class BayesianANCOVAResult:
    """Results from Bayesian ANCOVA analysis."""

    # Covariate statistics
    covariate_stats: List[BayesianCovariateStats]
    
    # Group comparisons (for G-type covariates)
    group_comparisons: Optional[List[BayesianGroupComparison]]
    
    # Model diagnostics
    rhat_max: float  # Maximum R-hat (should be < 1.01)
    ess_bulk_min: float  # Minimum bulk ESS (should be > 400)
    ess_tail_min: float  # Minimum tail ESS (should be > 400)
    divergences: int
    
    # Posterior samples (for further analysis)
    trace: Any  # az.InferenceData
    
    # For post-pre design
    adjusted_means: Optional[Dict[str, Tuple[float, float, float]]]  # (mean, hdi_low, hdi_high)
    adjusted_mean_diffs: Optional[Dict[Tuple[str, str], float]]
    credible_intervals_95: Optional[Dict[Tuple[str, str], Tuple[float, float]]]


class BayesianANCOVA:
    """
    Bayesian ANCOVA (Analysis of Covariance) implementation using MCMC.
    
    This class provides a Bayesian approach to ANCOVA using PyMC for MCMC
    sampling. It supports:
    - Multiple covariate types (Q, C, G)
    - One-sided and two-sided hypothesis testing
    - Credible intervals (HDI)
    - Post-pre experimental design analysis
    - ROPE (Region of Practical Equivalence) decision making
    
    References:
        - Kruschke, J. K. (2015). Doing Bayesian data analysis: A tutorial with
          R, JAGS, and Stan (2nd ed.). Academic Press.
        - Gelman, A., et al. (2013). Bayesian data analysis (3rd ed.).
          Chapman and Hall/CRC.
        - McElreath, R. (2020). Statistical rethinking: A Bayesian course with
          examples in R and Stan (2nd ed.). CRC press.
    """

    def __init__(
        self,
        hypothesis_type: HypothesisType = "two-sided",
        mcmc_samples: int = 2000,
        mcmc_tune: int = 1000,
        mcmc_chains: int = 4,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize Bayesian ANCOVA analyzer.
        
        Args:
            hypothesis_type: "two-sided" or "one-sided" hypothesis testing
            mcmc_samples: Number of MCMC samples per chain (default: 2000)
            mcmc_tune: Number of tuning steps (default: 1000)
            mcmc_chains: Number of MCMC chains (default: 4)
            random_seed: Random seed for reproducibility
        """
        self.hypothesis_type = hypothesis_type
        self.mcmc_samples = mcmc_samples
        self.mcmc_tune = mcmc_tune
        self.mcmc_chains = mcmc_chains
        self.random_seed = random_seed

    def fit(
        self,
        dependent_var: DependentVariable,
        covariates: Dict[str, Covariate],
        hdi_prob: float = 0.95,
        rope: Optional[Tuple[float, float]] = None,
    ) -> BayesianANCOVAResult:
        """
        Fit Bayesian ANCOVA model to data using MCMC.
        
        Args:
            dependent_var: Dependent variable array
            covariates: Dictionary mapping covariate names to (data, type) tuples
            hdi_prob: Probability for HDI (Highest Density Interval), default 0.95
            rope: Region of Practical Equivalence as (lower, upper) tuple
            
        Returns:
            BayesianANCOVAResult object containing all analysis results
        """
        # Validate inputs
        self._validate_inputs(dependent_var, covariates)
        
        # Prepare data
        df = self._prepare_dataframe(dependent_var, covariates)
        
        # Build and fit Bayesian model
        model, trace = self._build_and_sample_model(df, covariates)
        
        # Extract covariate statistics
        covariate_stats = self._extract_covariate_stats(
            trace, covariates, hdi_prob
        )
        
        # Perform group comparisons for G-type covariates
        group_comparisons = self._perform_group_comparisons(
            trace, df, covariates, hdi_prob, rope
        )
        
        # Calculate diagnostics
        rhat_max, ess_bulk_min, ess_tail_min, divergences = self._calculate_diagnostics(trace)
        
        return BayesianANCOVAResult(
            covariate_stats=covariate_stats,
            group_comparisons=group_comparisons,
            rhat_max=rhat_max,
            ess_bulk_min=ess_bulk_min,
            ess_tail_min=ess_tail_min,
            divergences=divergences,
            trace=trace,
            adjusted_means=None,
            adjusted_mean_diffs=None,
            credible_intervals_95=None,
        )

    def fit_postpre(
        self,
        pre_scores: DependentVariable,
        post_scores: DependentVariable,
        groups: npt.NDArray[np.str_],
        additional_covariates: Optional[Dict[str, Covariate]] = None,
        hdi_prob: float = 0.95,
        rope: Optional[Tuple[float, float]] = None,
    ) -> BayesianANCOVAResult:
        """
        Fit Bayesian ANCOVA model for post-pre experimental design.
        
        This analyzes change from pre to post using Bayesian inference,
        providing credible intervals for adjusted mean changes.
        
        Args:
            pre_scores: Pre-intervention scores
            post_scores: Post-intervention scores
            groups: Group labels for each observation
            additional_covariates: Optional additional covariates
            hdi_prob: Probability for HDI (default: 0.95)
            rope: Region of Practical Equivalence
            
        Returns:
            BayesianANCOVAResult with adjusted means and credible intervals
        """
        # Validate inputs
        if len(pre_scores) != len(post_scores) or len(pre_scores) != len(groups):
            raise ValueError("All input arrays must have the same length")
        
        # Calculate change scores
        change_scores = post_scores - pre_scores
        
        # Prepare covariates dictionary
        covariates: Dict[str, Covariate] = {
            "baseline": (pre_scores, "Q"),
            "group": (groups.astype(str), "G"),
        }
        
        if additional_covariates:
            covariates.update(additional_covariates)
        
        # Fit regular Bayesian ANCOVA
        result = self.fit(change_scores, covariates, hdi_prob, rope)
        
        # Calculate adjusted means with credible intervals
        df = self._prepare_dataframe(change_scores, covariates)
        adjusted_means = self._calculate_adjusted_means(
            result.trace, df, groups, pre_scores, hdi_prob
        )
        
        # Calculate pairwise differences
        adjusted_mean_diffs, credible_intervals = self._calculate_mean_differences(
            adjusted_means, result.trace, hdi_prob
        )
        
        # Update result with post-pre specific information
        result.adjusted_means = adjusted_means
        result.adjusted_mean_diffs = adjusted_mean_diffs
        result.credible_intervals_95 = credible_intervals
        
        return result

    def _validate_inputs(
        self,
        dependent_var: DependentVariable,
        covariates: Dict[str, Covariate],
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
            if cov_type not in ("Q", "C", "G"):
                raise ValueError(
                    f"Invalid covariate type '{cov_type}' for '{name}'. "
                    "Must be 'Q', 'C', or 'G'"
                )

    def _prepare_dataframe(
        self,
        dependent_var: DependentVariable,
        covariates: Dict[str, Covariate],
    ) -> pd.DataFrame:
        """Prepare pandas DataFrame for analysis."""
        data = {"y": dependent_var}
        
        for name, (cov_data, cov_type) in covariates.items():
            if cov_type in ("C", "G"):
                # Categorical variables - convert to codes
                cat = pd.Categorical(cov_data.astype(str))
                data[name] = cat.codes
                data[f"{name}_labels"] = cat
            else:
                # Quantitative variables - standardize
                data[name] = (cov_data - np.mean(cov_data)) / np.std(cov_data)
        
        return pd.DataFrame(data)

    def _build_and_sample_model(
        self,
        df: pd.DataFrame,
        covariates: Dict[str, Covariate],
    ) -> Tuple[Any, Any]:  # Tuple[pm.Model, az.InferenceData]
        """Build and sample from Bayesian model."""
        with pm.Model() as model:
            # Priors for intercept
            intercept = pm.Normal("intercept", mu=0, sigma=10)
            
            # Priors for each covariate
            mu = intercept
            for name, (_, cov_type) in covariates.items():
                if cov_type == "Q":
                    # Quantitative covariate - single coefficient
                    beta = pm.Normal(f"beta_{name}", mu=0, sigma=10)
                    mu += beta * df[name].values
                else:
                    # Categorical covariate - multiple coefficients
                    n_categories = int(df[name].max()) + 1
                    if n_categories > 1:
                        # Use sum-to-zero constraint for identifiability
                        beta_raw = pm.Normal(f"beta_{name}_raw", mu=0, sigma=10, shape=n_categories - 1)
                        beta_last = -pm.math.sum(beta_raw)
                        beta = pm.math.concatenate([beta_raw, [beta_last]])
                        mu += beta[df[name].values.astype(int)]
            
            # Prior for noise
            sigma = pm.HalfNormal("sigma", sigma=10)
            
            # Likelihood
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=df["y"].values)
            
            # Sample
            trace = pm.sample(
                draws=self.mcmc_samples,
                tune=self.mcmc_tune,
                chains=self.mcmc_chains,
                random_seed=self.random_seed,
                return_inferencedata=True,
                progressbar=False,
            )
        
        return model, trace

    def _extract_covariate_stats(
        self,
        trace: Any,  # az.InferenceData
        covariates: Dict[str, Covariate],
        hdi_prob: float,
    ) -> List[BayesianCovariateStats]:
        """Extract Bayesian statistics for each covariate."""
        stats_list = []
        
        for name, (_, cov_type) in covariates.items():
            param_name = f"beta_{name}"
            
            if cov_type == "Q":
                # Quantitative covariate
                if param_name in trace.posterior:
                    samples = trace.posterior[param_name].values.flatten()
                    hdi = az.hdi(trace, var_names=[param_name], hdi_prob=hdi_prob)
                    
                    stats_list.append(
                        BayesianCovariateStats(
                            name=name,
                            covariate_type=cov_type,
                            mean=float(np.mean(samples)),
                            std=float(np.std(samples)),
                            hdi_lower=float(hdi[param_name].values[0]),
                            hdi_upper=float(hdi[param_name].values[1]),
                            prob_positive=float(np.mean(samples > 0)),
                            prob_negative=float(np.mean(samples < 0)),
                        )
                    )
            else:
                # Categorical covariate - report first coefficient
                param_name_raw = f"beta_{name}_raw"
                if param_name_raw in trace.posterior:
                    samples = trace.posterior[param_name_raw].values[:, :, 0].flatten()
                    
                    # Calculate HDI manually for first coefficient
                    samples_sorted = np.sort(samples)
                    n = len(samples_sorted)
                    interval_size = int(np.ceil(hdi_prob * n))
                    n_intervals = n - interval_size
                    interval_width = samples_sorted[interval_size:] - samples_sorted[:n_intervals]
                    min_idx = int(np.argmin(interval_width))
                    hdi_lower = samples_sorted[min_idx]
                    hdi_upper = samples_sorted[min_idx + interval_size]
                    
                    stats_list.append(
                        BayesianCovariateStats(
                            name=name,
                            covariate_type=cov_type,
                            mean=float(np.mean(samples)),
                            std=float(np.std(samples)),
                            hdi_lower=float(hdi_lower),
                            hdi_upper=float(hdi_upper),
                            prob_positive=float(np.mean(samples > 0)),
                            prob_negative=float(np.mean(samples < 0)),
                        )
                    )
        
        return stats_list

    def _perform_group_comparisons(
        self,
        trace: Any,  # az.InferenceData
        df: pd.DataFrame,
        covariates: Dict[str, Covariate],
        hdi_prob: float,
        rope: Optional[Tuple[float, float]],
    ) -> Optional[List[BayesianGroupComparison]]:
        """Perform Bayesian group comparisons for G-type covariates."""
        comparisons = []
        
        # Find G-type covariates
        g_covariates = [
            name for name, (_, cov_type) in covariates.items() if cov_type == "G"
        ]
        
        if not g_covariates:
            return None
        
        for cov_name in g_covariates:
            param_name = f"beta_{cov_name}_raw"
            if param_name not in trace.posterior:
                continue
            
            # Get group labels
            unique_groups = df[f"{cov_name}_labels"].cat.categories
            n_groups = len(unique_groups)
            
            # Get posterior samples
            samples = trace.posterior[param_name].values
            
            # Reconstruct full coefficients including last (constrained) one
            samples_reshaped = samples.reshape(-1, samples.shape[-1])
            last_coef = -np.sum(samples_reshaped, axis=1, keepdims=True)
            full_samples = np.concatenate([samples_reshaped, last_coef], axis=1)
            
            # Compare all pairs
            for i in range(n_groups):
                for j in range(i + 1, n_groups):
                    diff_samples = full_samples[:, i] - full_samples[:, j]
                    
                    # Calculate HDI
                    diff_sorted = np.sort(diff_samples)
                    n = len(diff_sorted)
                    interval_size = int(np.ceil(hdi_prob * n))
                    n_intervals = n - interval_size
                    interval_width = diff_sorted[interval_size:] - diff_sorted[:n_intervals]
                    min_idx = int(np.argmin(interval_width))
                    hdi_lower = diff_sorted[min_idx]
                    hdi_upper = diff_sorted[min_idx + interval_size]
                    
                    # ROPE decision if rope is specified
                    rope_decision = None
                    if rope is not None:
                        prob_in_rope = float(np.mean(
                            (diff_samples > rope[0]) & (diff_samples < rope[1])
                        ))
                        if prob_in_rope > 0.95:
                            rope_decision = "accept"
                        elif prob_in_rope < 0.05:
                            rope_decision = "reject"
                        else:
                            rope_decision = "undecided"
                    
                    comparisons.append(
                        BayesianGroupComparison(
                            group1=str(unique_groups[i]),
                            group2=str(unique_groups[j]),
                            mean_diff=float(np.mean(diff_samples)),
                            hdi_lower=float(hdi_lower),
                            hdi_upper=float(hdi_upper),
                            prob_greater=float(np.mean(diff_samples > 0)),
                            rope_decision=rope_decision,
                        )
                    )
        
        return comparisons if comparisons else None

    def _calculate_diagnostics(
        self, trace: Any  # az.InferenceData
    ) -> Tuple[float, float, float, int]:
        """Calculate MCMC diagnostics."""
        # R-hat (should be < 1.01)
        rhat = az.rhat(trace)
        rhat_values = []
        for var in rhat.data_vars:
            rhat_values.extend(rhat[var].values.flatten())
        rhat_max = float(np.max(rhat_values))
        
        # ESS (Effective Sample Size)
        ess = az.ess(trace)
        ess_values = []
        for var in ess.data_vars:
            ess_values.extend(ess[var].values.flatten())
        ess_bulk_min = float(np.min(ess_values))
        ess_tail_min = ess_bulk_min  # Simplified
        
        # Divergences
        divergences = 0
        if hasattr(trace, "sample_stats") and "diverging" in trace.sample_stats:
            divergences = int(trace.sample_stats.diverging.values.sum())
        
        return rhat_max, ess_bulk_min, ess_tail_min, divergences

    def _calculate_adjusted_means(
        self,
        trace: Any,  # az.InferenceData
        df: pd.DataFrame,
        groups: npt.NDArray[np.str_],
        baseline: npt.NDArray[np.float64],
        hdi_prob: float,
    ) -> Dict[str, Tuple[float, float, float]]:
        """Calculate adjusted means with credible intervals."""
        adjusted_means = {}
        unique_groups = np.unique(groups)
        
        # Get intercept samples
        intercept_samples = trace.posterior["intercept"].values.flatten()
        
        # Get baseline coefficient samples (if exists)
        baseline_samples = None
        if "beta_baseline" in trace.posterior:
            baseline_samples = trace.posterior["beta_baseline"].values.flatten()
        
        # Mean baseline (standardized)
        mean_baseline_std = 0.0  # Standardized mean is 0
        
        for group_idx, group in enumerate(unique_groups):
            # Get group coefficient samples
            if "beta_group_raw" in trace.posterior:
                group_samples_raw = trace.posterior["beta_group_raw"].values
                group_samples_reshaped = group_samples_raw.reshape(-1, group_samples_raw.shape[-1])
                last_coef = -np.sum(group_samples_reshaped, axis=1)
                full_group_samples = np.concatenate(
                    [group_samples_reshaped.T, last_coef.reshape(1, -1)]
                ).T
                
                group_coef = full_group_samples[:, group_idx]
            else:
                group_coef = np.zeros_like(intercept_samples)
            
            # Calculate predicted change
            if baseline_samples is not None:
                predicted = intercept_samples + baseline_samples * mean_baseline_std + group_coef
            else:
                predicted = intercept_samples + group_coef
            
            # Calculate HDI
            pred_sorted = np.sort(predicted)
            n = len(pred_sorted)
            interval_size = int(np.ceil(hdi_prob * n))
            n_intervals = n - interval_size
            interval_width = pred_sorted[interval_size:] - pred_sorted[:n_intervals]
            min_idx = int(np.argmin(interval_width))
            hdi_lower = pred_sorted[min_idx]
            hdi_upper = pred_sorted[min_idx + interval_size]
            
            adjusted_means[str(group)] = (
                float(np.mean(predicted)),
                float(hdi_lower),
                float(hdi_upper),
            )
        
        return adjusted_means

    def _calculate_mean_differences(
        self,
        adjusted_means: Dict[str, Tuple[float, float, float]],
        trace: Any,  # az.InferenceData
        hdi_prob: float,
    ) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], Tuple[float, float]]]:
        """Calculate pairwise differences between adjusted means."""
        differences = {}
        intervals = {}
        
        groups = list(adjusted_means.keys())
        for i, group1 in enumerate(groups):
            for group2 in groups[i + 1 :]:
                mean1, _, _ = adjusted_means[group1]
                mean2, _, _ = adjusted_means[group2]
                diff = mean1 - mean2
                differences[(group1, group2)] = diff
                
                # Calculate credible interval for difference
                # Using approximation based on individual HDIs
                _, hdi_low1, hdi_high1 = adjusted_means[group1]
                _, hdi_low2, hdi_high2 = adjusted_means[group2]
                
                # Conservative estimate
                diff_low = hdi_low1 - hdi_high2
                diff_high = hdi_high1 - hdi_low2
                
                intervals[(group1, group2)] = (diff_low, diff_high)
        
        return differences, intervals
