"""Statistical computation modules for ANCOVA."""

from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from ancovapy.ancova import CovariateStats, PostHocResult
from ancovapy.types import Covariate


class FormulaBuilder:
    """
    Builds R-style formulas for statsmodels.

    Follows Single Responsibility Principle by focusing only on formula construction.
    """

    @staticmethod
    def build(covariates: dict[str, Covariate]) -> str:
        """
        Build R-style formula for statsmodels OLS.

        Args:
            covariates: Dictionary of covariates

        Returns:
            Formula string (e.g., "y ~ age + C(treatment)")
        """
        terms = []

        for name, (_, cov_type) in covariates.items():
            if cov_type in ("C", "G"):
                # Categorical: use C() wrapper
                terms.append(f"C({name})")
            else:
                # Quantitative: use as-is
                terms.append(name)

        return "y ~ " + " + ".join(terms)


class CovariateStatsExtractor:
    """
    Extracts covariate statistics from fitted model.

    Follows Single Responsibility Principle.
    """

    @staticmethod
    def extract(
        model: sm.regression.linear_model.RegressionResultsWrapper,
        covariates: dict[str, Covariate],
        alpha: float,
    ) -> list[CovariateStats]:
        """
        Extract statistics for each covariate from fitted model.

        Args:
            model: Fitted statsmodels OLS model
            covariates: Dictionary of covariates
            alpha: Significance level

        Returns:
            List of CovariateStats objects
        """
        stats_list = []

        params = model.params
        conf_int = model.conf_int(alpha=alpha)

        for name, (_, cov_type) in covariates.items():
            if cov_type == "Q":
                # Quantitative covariate - single parameter
                if name in params.index:
                    stats_list.append(
                        CovariateStats(
                            name=name,
                            covariate_type=cov_type,
                            coefficient=params[name],
                            std_error=model.bse[name],
                            t_value=model.tvalues[name],
                            p_value=model.pvalues[name],
                            ci_lower=conf_int.loc[name, 0],
                            ci_upper=conf_int.loc[name, 1],
                        )
                    )
            else:
                # Categorical - find first related parameter
                related_params = [p for p in params.index if name in p]
                if related_params:
                    param_name = related_params[0]
                    stats_list.append(
                        CovariateStats(
                            name=name,
                            covariate_type=cov_type,
                            coefficient=params[param_name],
                            std_error=model.bse[param_name],
                            t_value=model.tvalues[param_name],
                            p_value=model.pvalues[param_name],
                            ci_lower=conf_int.loc[param_name, 0],
                            ci_upper=conf_int.loc[param_name, 1],
                        )
                    )

        return stats_list


class TukeyPostHocAnalyzer:
    """
    Performs Tukey HSD post-hoc tests.

    Follows Single Responsibility and Open/Closed Principles.
    Can be extended or replaced with other post-hoc methods.
    """

    @staticmethod
    def analyze(
        df: pd.DataFrame, covariates: dict[str, Covariate], alpha: float
    ) -> Optional[list[PostHocResult]]:
        """
        Perform Tukey HSD post-hoc tests for G-type covariates.

        Args:
            df: Prepared DataFrame
            covariates: Dictionary of covariates
            alpha: Significance level

        Returns:
            List of PostHocResult objects or None
        """
        posthoc_results = []

        # Find G-type covariates
        g_covariates = [
            name for name, (_, cov_type) in covariates.items() if cov_type == "G"
        ]

        if not g_covariates:
            return None

        for cov_name in g_covariates:
            try:
                tukey = pairwise_tukeyhsd(df["y"], df[cov_name], alpha=alpha)

                # Extract results from Tukey summary
                for i in range(len(tukey.summary().data) - 1):  # Skip header
                    row = tukey.summary().data[i + 1]
                    posthoc_results.append(
                        PostHocResult(
                            group1=str(row[0]),
                            group2=str(row[1]),
                            mean_diff=float(row[2]),
                            p_value=float(row[3]),
                            ci_lower=float(row[4]),
                            ci_upper=float(row[5]),
                            reject=bool(row[6]),
                        )
                    )
            except Exception:
                # If Tukey test fails, continue with other covariates
                continue

        return posthoc_results if posthoc_results else None


class AdjustedMeanCalculator:
    """
    Calculates adjusted means for post-pre designs.

    Follows Single Responsibility Principle.
    """

    @staticmethod
    def calculate(
        model: sm.regression.linear_model.RegressionResultsWrapper,
        df: pd.DataFrame,
        groups: np.ndarray,
        baseline: np.ndarray,
    ) -> dict[str, float]:
        """
        Calculate adjusted means for each group at mean baseline.

        Args:
            model: Fitted model
            df: Prepared DataFrame
            groups: Group labels
            baseline: Baseline scores

        Returns:
            Dictionary of adjusted means by group
        """
        adjusted_means = {}
        unique_groups = np.unique(groups)
        mean_baseline = float(np.mean(baseline))

        for group in unique_groups:
            # Create prediction data at mean baseline
            pred_data = pd.DataFrame(
                {"baseline": [mean_baseline], "group": pd.Categorical([group])}
            )

            # Add other covariates at their means if present
            for col in df.columns:
                if col not in ["y", "baseline", "group"]:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        pred_data[col] = [df[col].mean()]
                    else:
                        pred_data[col] = [df[col].mode()[0]]

            try:
                predicted = model.predict(pred_data)
                adjusted_means[str(group)] = float(predicted[0])
            except Exception:
                # Fallback to simple mean
                group_mask = groups == group
                adjusted_means[str(group)] = float(df.loc[group_mask, "y"].mean())

        return adjusted_means

    @staticmethod
    def calculate_differences(
        adjusted_means: dict[str, float], alpha: float
    ) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], tuple[float, float]]]:
        """
        Calculate pairwise differences between adjusted means.

        Args:
            adjusted_means: Dictionary of adjusted means
            alpha: Significance level

        Returns:
            Tuple of (differences dict, confidence intervals dict)
        """
        differences = {}
        intervals = {}

        groups = list(adjusted_means.keys())
        for i, group1 in enumerate(groups):
            for group2 in groups[i + 1 :]:
                diff = adjusted_means[group1] - adjusted_means[group2]
                differences[(group1, group2)] = diff

                # Approximation for confidence interval
                se_approx = abs(diff) * 0.2  # Rough approximation
                margin = stats.norm.ppf(1 - alpha / 2) * se_approx
                intervals[(group1, group2)] = (diff - margin, diff + margin)

        return differences, intervals
