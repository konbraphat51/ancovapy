from dataclasses import dataclass
from typing import Literal
import numpy as np

@dataclass
class AncovaResults:
    """Dataclass to report ANCOVA results."""

    @dataclass
    class ModelDescription:
        """Dataclass to describe the ANCOVA model used."""

        @dataclass
        class Term:
            """Dataclass to describe each term in the model."""
            name: str
            
        @dataclass
        class TermQuantitative(Term):
            mean: np.float64
            std: np.float64
            min: np.float64
            max: np.float64

        @dataclass
        class TermCategorical(Term):
            categories: dict[str, int]
            category_n: int
            control_category: str
            comparing: bool

        @dataclass
        class TermInteraction(Term):
            interacting_terms: list[str]

        model_type: Literal["standard ANCOVA"]
        ss_type: Literal[1, 2, 3]
        dependent_variable: TermQuantitative

        # term name -> Term
        independent_variables: dict[str, Term]
        
        alpha: np.float64
        """Significance level used for hypothesis testing."""

    @dataclass
    class AssumptionResults:
        """Dataclass to report assumption test results."""

        @dataclass
        class HomogeneityOfRegressionSlopes:
            """Dataclass to report homogeneity of regression slopes test results.
            
            if `passed==False`, use Heterogeneous Regression ANCOVA model.
            """

            @dataclass
            class InteractionTerm:
                """significance of interaction term in full model"""

                target_terms: list[str]
                f_statistic: np.float64
                df: int
                p_value: np.float64
                rejected: bool

            interaction_terms: list[InteractionTerm]

            passed: bool
            """True if all interaction terms are not significant."""

        @dataclass
        class IndependenceOfTreatment:
            """Dataclass to report independence of treatment test results."""

            f_statistic: np.float64
            df: int
            p_value: np.float64
            rejected: bool
            passed: bool

        @dataclass
        class NormalityOfResiduals:
            """Dataclass to report normality of residuals test results."""

            w_statistic: np.float64
            p_value: np.float64
            rejected: bool
            passed: bool

        @dataclass
        class IndependenceOfErrors:
            """Dataclass to report independence of errors test results."""

            durbin_watson_statistic: np.float64

        homogeneity_of_regression_slopes: HomogeneityOfRegressionSlopes
        independence_of_treatment: IndependenceOfTreatment
        normality_of_residuals: NormalityOfResiduals
        independence_of_errors: IndependenceOfErrors
        passed: bool
        """True if all assumptions are passed."""

    @dataclass
    class OmnibusResults:
        """Dataclass to report omnibus ANCOVA results."""

        @dataclass
        class TermResult:
            """Dataclass to report each term's ANCOVA result."""

            name: str
            f_statistic: np.float64
            df_between: int
            df_within: int
            p_value: np.float64
            rejected: bool
            mse: np.float64

        @dataclass
        class MeanAdjusted:
            """Dataclass to report adjusted mean."""

            category: str
            adjusted_mean: np.float64
            std: np.float64

        # term name -> TermResult
        term_results: dict[str, TermResult]
        
        # category name -> adjusted mean
        adjusted_means: dict[str, MeanAdjusted]


    description: ModelDescription
    assumption: AssumptionResults
    omnibus: OmnibusResults
