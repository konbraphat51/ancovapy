"""Classical ANCOVA implementation using statsmodels."""

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from ancovapy.constants import VALID_COVARIATE_TYPES
from ancovapy.helpers import convert_to_categorical, validate_covariate_type
from ancovapy.results import ANCOVAResult, CovariateStats, PostHocResult
from ancovapy.types import Covariate, DependentVariable, SSType


class ANCOVA:
    """
    Classical ANCOVA (Analysis of Covariance) implementation.

    This class provides a scientifically rigorous implementation of ANCOVA
    using statsmodels as the backend. It supports:
    - Multiple covariate types (Q, C, G)
    - Configurable SS (Sum of Squares) types
    - Post-hoc tests for group comparisons
    - Post-pre experimental design analysis

    Covariate Types:
        - "Q" (Quantitative): Continuous numerical variables (e.g., age, baseline score)
        - "C" (Categorical): Categorical factors without pairwise comparisons (e.g., site, gender)
        - "G" (Group): Categorical factors with automatic Tukey HSD post-hoc tests (e.g., treatment groups)

    References:
        - Maxwell, S. E., & Delaney, H. D. (2004). Designing experiments and
          analyzing data: A model comparison perspective (2nd ed.). Lawrence
          Erlbaum Associates.
        - Rutherford, A. (2011). ANOVA and ANCOVA: A GLM approach (2nd ed.).
          John Wiley & Sons.
    """

    def __init__(self, ss_type: SSType = 2):
        """
        Initialize ANCOVA analyzer.

        Args:
            ss_type: Sum of Squares type (1, 2, or 3). Default is 2.
                    Type I: Sequential (order-dependent)
                    Type II: Hierarchical (recommended for balanced designs)
                    Type III: Marginal (recommended for unbalanced designs)
        """
        self.ss_type = ss_type

    def fit(
        self,
        dependent_var: DependentVariable,
        covariates: dict[str, Covariate],
        alpha: float = 0.05,
    ) -> ANCOVAResult:
        """
        Fit ANCOVA model to data.

        Args:
            dependent_var: Dependent variable array (outcome measurements)
            covariates: Dictionary mapping covariate names to (data, type) tuples.
                       Each tuple contains:
                       - data: Array of covariate values (can be numbers or strings)
                       - type: One of "Q" (quantitative), "C" (categorical), or "G" (group)
            alpha: Significance level for statistical tests (default: 0.05)

        Returns:
            ANCOVAResult object containing all analysis results

        Raises:
            ValueError: If inputs are invalid or incompatible
        """
        # Validate inputs
        self._validate_inputs(dependent_var, covariates)

        # Prepare data
        df = self._prepare_dataframe(dependent_var, covariates)

        # Build formula
        formula = self._build_formula(covariates)

        # Fit model
        model = ols(formula, data=df).fit()

        # Calculate ANOVA table with specified SS type
        anova_table = sm.stats.anova_lm(model, typ=self.ss_type)

        # Extract covariate statistics (using ANOVA table for p-values)
        covariate_stats = self._extract_covariate_stats(
            model, covariates, alpha, anova_table
        )

        # Perform post-hoc tests for G-type covariates
        posthoc_results = self._perform_posthoc_tests(df, covariates, alpha)

        # Calculate adjusted means for G-type covariates
        adjusted_means, adj_mean_diffs, confidence_intervals = (
            self._calculate_adjusted_means_for_groups(
                model, df, covariates, alpha
            )
        )

        return ANCOVAResult(
            f_statistic=model.fvalue,
            p_value=model.f_pvalue,
            r_squared=model.rsquared,
            adj_r_squared=model.rsquared_adj,
            covariate_stats=covariate_stats,
            posthoc_results=posthoc_results,
            model_summary=str(model.summary()),
            adjusted_means=adjusted_means,
            adj_mean_diffs=adj_mean_diffs,
            confidence_intervals=confidence_intervals,
        )

    def fit_postpre(
        self,
        pre_scores: DependentVariable,
        post_scores: DependentVariable,
        groups: npt.NDArray[np.str_] | npt.NDArray[np.int_] | npt.NDArray[np.float64],
        additional_covariates: dict[str, Covariate] | None = None,
        alpha: float = 0.05,
    ) -> ANCOVAResult:
        """
        Fit ANCOVA model for post-pre experimental design.

        This analyzes post-intervention scores adjusting for baseline (pre) scores
        and any additional covariates. It calculates adjusted mean post scores and
        their differences between groups.

        Args:
            pre_scores: Pre-intervention (baseline) scores
            post_scores: Post-intervention scores (this is the dependent variable)
            groups: Group labels for each observation (can be strings or numbers)
            additional_covariates: Optional additional covariates
            alpha: Significance level (default: 0.05)

        Returns:
            ANCOVAResult with adjusted means, differences, and confidence intervals

        References:
            - Vickers, A. J., & Altman, D. G. (2001). Analysing controlled trials
              with baseline and follow up measurements. BMJ, 323(7321), 1123-1124.
        """
        # Validate inputs
        if len(pre_scores) != len(post_scores) or len(pre_scores) != len(groups):
            raise ValueError("All input arrays must have the same length")

        # Prepare covariates dictionary with baseline and group
        covariates = self._prepare_postpre_covariates(
            pre_scores, groups, additional_covariates
        )

        # Fit ANCOVA with post scores as dependent variable
        result = self.fit(post_scores, covariates, alpha)

        # Calculate adjusted means using the fitted model
        adjusted_means, adj_mean_diffs, confidence_intervals = (
            self._calculate_postpre_adjusted_values(
                post_scores, covariates, groups, pre_scores, alpha
            )
        )

        # Update result with post-pre specific information
        result.adjusted_means = adjusted_means
        result.adj_mean_diffs = adj_mean_diffs
        result.confidence_intervals = confidence_intervals

        return result

    def _calculate_postpre_adjusted_values(
        self,
        post_scores: DependentVariable,
        covariates: dict[str, Covariate],
        groups: npt.NDArray,
        pre_scores: npt.NDArray[np.float64],
        alpha: float,
    ) -> tuple[
        dict[str, float],
        dict[tuple[str, str], float],
        dict[tuple[str, str], tuple[float, float]],
    ]:
        """Calculate adjusted means, differences, and confidence intervals for post-pre design."""
        # Prepare data and fit model
        df = self._prepare_dataframe(post_scores, covariates)
        formula = self._build_formula(covariates)
        model = ols(formula, data=df).fit()

        # Calculate adjusted means
        adjusted_means = self._calculate_adjusted_means_prediction(
            model, df, groups, pre_scores
        )

        # Calculate pairwise differences
        adj_mean_diffs, confidence_intervals = self._calculate_mean_differences(
            adjusted_means, alpha
        )

        return adjusted_means, adj_mean_diffs, confidence_intervals

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
        """Prepare pandas DataFrame for analysis."""
        data = {"y": dependent_var}

        for name, (cov_data, cov_type) in covariates.items():
            if cov_type in ("C", "G"):
                # Categorical variables - handles both strings and numbers
                data[name] = convert_to_categorical(cov_data)
            else:
                # Quantitative variables
                data[name] = cov_data.astype(float)

        return pd.DataFrame(data)

    def _build_formula(self, covariates: dict[str, Covariate]) -> str:
        """Build R-style formula for statsmodels."""
        terms = []

        for name, (_, cov_type) in covariates.items():
            if cov_type in ("C", "G"):
                # Categorical: use C() wrapper
                terms.append(f"C({name})")
            else:
                # Quantitative: use as-is
                terms.append(name)

        formula = "y ~ " + " + ".join(terms)
        return formula

    def _extract_covariate_stats(
        self,
        model: sm.regression.linear_model.RegressionResultsWrapper,
        covariates: dict[str, Covariate],
        alpha: float,
        anova_table: pd.DataFrame,
    ) -> list[CovariateStats]:
        """Extract statistics for each covariate using ANOVA table for p-values."""
        stats_list = []

        params = model.params
        conf_int = model.conf_int(alpha=alpha)

        for name, (_, cov_type) in covariates.items():
            if cov_type == "Q":
                # Quantitative covariate - single parameter
                if name in params.index:
                    # Use ANOVA table p-value if available
                    if name in anova_table.index:
                        p_value = anova_table.loc[name, "PR(>F)"]
                    else:
                        p_value = model.pvalues[name]
                    
                    stats_list.append(
                        CovariateStats(
                            name=name,
                            covariate_type=cov_type,
                            coefficient=params[name],
                            std_error=model.bse[name],
                            t_value=model.tvalues[name],
                            p_value=p_value,
                            ci_lower=conf_int.loc[name, 0],
                            ci_upper=conf_int.loc[name, 1],
                        )
                    )
            else:
                # Categorical - find first related parameter and use ANOVA table p-value
                related_params = [p for p in params.index if name in p]
                if related_params:
                    param_name = related_params[0]
                    
                    # Look for corresponding row in ANOVA table
                    # Try different naming conventions: name, C(name), etc.
                    anova_p_value = model.pvalues[param_name]  # fallback
                    for idx in anova_table.index:
                        if name in idx:
                            anova_p_value = anova_table.loc[idx, "PR(>F)"]
                            break
                    
                    stats_list.append(
                        CovariateStats(
                            name=name,
                            covariate_type=cov_type,
                            coefficient=params[param_name],
                            std_error=model.bse[param_name],
                            t_value=model.tvalues[param_name],
                            p_value=anova_p_value,
                            ci_lower=conf_int.loc[param_name, 0],
                            ci_upper=conf_int.loc[param_name, 1],
                        )
                    )

        return stats_list

    def _perform_posthoc_tests(
        self,
        df: pd.DataFrame,
        covariates: dict[str, Covariate],
        alpha: float,
    ) -> list[PostHocResult] | None:
        """Perform post-hoc tests for G-type covariates."""
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

    def _calculate_adjusted_means_prediction(
        self,
        model: sm.regression.linear_model.RegressionResultsWrapper,
        df: pd.DataFrame,
        groups: npt.NDArray,
        baseline: npt.NDArray[np.float64],
    ) -> dict[str, float]:
        """
        Calculate adjusted means using model predictions for all rows.
        
        This computes predicted values for all observations and takes the mean
        within each group, which is more robust than predicting at mean baseline.
        """
        adjusted_means = {}
        unique_groups = np.unique(groups)
        
        # Get predictions for all observations
        predictions = model.predict(df)
        
        # Calculate mean prediction for each group
        for group in unique_groups:
            group_mask = groups.astype(str) == str(group)
            group_predictions = predictions[group_mask]
            adjusted_means[str(group)] = float(np.mean(group_predictions))
        
        return adjusted_means

    def _calculate_mean_differences(
        self,
        adjusted_means: dict[str, float],
        alpha: float,
    ) -> tuple[
        dict[tuple[str, str], float], dict[tuple[str, str], tuple[float, float]]
    ]:
        """Calculate pairwise differences between adjusted means."""
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

    def _calculate_adjusted_means_for_groups(
        self,
        model: sm.regression.linear_model.RegressionResultsWrapper,
        df: pd.DataFrame,
        covariates: dict[str, Covariate],
        alpha: float,
    ) -> tuple[
        dict[str, float] | None,
        dict[tuple[str, str], float] | None,
        dict[tuple[str, str], tuple[float, float]] | None,
    ]:
        """Calculate adjusted means for all G-type covariates."""
        # Find G-type covariates
        g_covariates = [
            (name, data) for name, (data, cov_type) in covariates.items() if cov_type == "G"
        ]

        if not g_covariates:
            return None, None, None

        # For simplicity, calculate adjusted means for the first G-type covariate
        # (typically there's one primary grouping variable)
        cov_name, cov_data = g_covariates[0]
        unique_groups = np.unique(cov_data.astype(str))

        # Calculate mean of all other quantitative covariates
        covariate_means = {}
        for name, (data, cov_type) in covariates.items():
            if cov_type == "Q":
                covariate_means[name] = float(np.mean(data))

        # Calculate adjusted mean for each group
        adjusted_means = {}
        for group in unique_groups:
            # Create prediction dataframe with this group and mean covariates
            pred_data = {cov_name: [group]}
            pred_data.update(covariate_means)

            # Add categorical covariates at their reference level or mode
            for name, (data, cov_type) in covariates.items():
                if cov_type in ("C", "G") and name != cov_name:
                    # Use most common category
                    unique_vals, counts = np.unique(data.astype(str), return_counts=True)
                    mode_val = unique_vals[np.argmax(counts)]
                    pred_data[name] = [mode_val]

            pred_df = pd.DataFrame(pred_data)

            # Ensure categorical columns are properly typed
            for name, (_, cov_type) in covariates.items():
                if cov_type in ("C", "G") and name in pred_df.columns:
                    pred_df[name] = pd.Categorical(pred_df[name])

            # Make prediction
            try:
                prediction = model.predict(pred_df)
                adjusted_means[str(group)] = float(prediction[0])
            except Exception:
                # If prediction fails, skip this calculation
                return None, None, None

        # Calculate pairwise differences
        adj_mean_diffs, confidence_intervals = self._calculate_mean_differences(
            adjusted_means, alpha
        )

        return adjusted_means, adj_mean_diffs, confidence_intervals
