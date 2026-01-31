from dataclasses import dataclass

@dataclass
class AncovaResults:
    """Dataclass to report ANCOVA results."""

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

        homogeneity_of_regression_slopes: HomogeneityOfRegressionSlopes

    assumption: AssumptionResults
