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
        independent_variables: list[Term]

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
                f_statistic: float
                df: int
                p_value: float
                rejected: bool

            interaction_terms: list[InteractionTerm]

            passed: bool
            """True if all interaction terms are not significant."""

        @dataclass
        class IndependenceOfTreatment:
            """Dataclass to report independence of treatment test results."""

            f_statistic: float
            df: int
            p_value: float
            rejected: bool
            passed: bool

        @dataclass
        class NormalityOfResiduals:
            """Dataclass to report normality of residuals test results."""

            w_statistic: float
            p_value: float
            rejected: bool
            passed: bool

        

        homogeneity_of_regression_slopes: HomogeneityOfRegressionSlopes
        independence_of_treatment: IndependenceOfTreatment
        normality_of_residuals: NormalityOfResiduals

    description: ModelDescription
    assumption: AssumptionResults
