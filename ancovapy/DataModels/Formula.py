from ancovapy.DataModels.Variables import Variable, QuantitativeVariable, CategoricalVariable

class LinearFormula:
    """Domain class for linear formulas used for GLM modeling."""

    def __init__(
        self,
        dependent_variable: QuantitativeVariable,
        independent_variables: list[Variable],
    ) -> None:
        self.dependent_variable = dependent_variable
        self.independent_variables = independent_variables

    @property
    def independent_categorical_variables(self) -> list[CategoricalVariable]:
        return [
            var
            for var in self.independent_variables
            if isinstance(var, CategoricalVariable)
        ]
    
    @property
    def independent_quantitative_variables(self) -> list[QuantitativeVariable]:
        return [
            var
            for var in self.independent_variables
            if isinstance(var, QuantitativeVariable)
        ]
