# SOLID Principles Architecture

## Overview

The ancovapy library has been refactored to follow SOLID principles for better maintainability, testability, and extensibility.

## Architecture Components

### Base Classes (`ancovapy/base.py`)

Following the **Dependency Inversion Principle**, we define abstract base classes that higher-level modules depend on:

- `AnalysisResult`: Base class for all analysis results
- `DataValidator`: Abstract validator interface
- `DataPreparator`: Abstract data preparation interface
- `StatisticalModel`: Abstract statistical model interface
- `PostHocAnalyzer`: Abstract post-hoc analysis interface
- `ANCOVAAnalyzer`: Template for ANCOVA analyzers using dependency injection

### Utility Classes (`ancovapy/utils.py`)

Following **Single Responsibility Principle**, each class has one reason to change:

- `StandardDataValidator`: Validates ANCOVA input data
- `PandasDataPreparator`: Prepares data for statsmodels (classical ANCOVA)
- `BayesianDataPreparator`: Prepares and standardizes data for PyMC

### Statistical Modules (`ancovapy/statistics.py`)

Following **Single Responsibility** and **Open/Closed Principles**:

- `FormulaBuilder`: Builds R-style formulas for statsmodels
- `CovariateStatsExtractor`: Extracts statistics from fitted models
- `TukeyPostHocAnalyzer`: Performs Tukey HSD tests (can be extended/replaced)
- `AdjustedMeanCalculator`: Calculates adjusted means for post-pre designs

## SOLID Principles Applied

### 1. Single Responsibility Principle (SRP)

Each class has a single, well-defined responsibility:

- **FormulaBuilder**: Only builds formulas
- **CovariateStatsExtractor**: Only extracts statistics
- **DataValidator**: Only validates data
- **DataPreparator**: Only prepares data

**Benefits**:
- Easier to test each component independently
- Changes to formula building don't affect validation
- Clear separation of concerns

### 2. Open/Closed Principle (OCP)

Classes are open for extension but closed for modification:

- `PostHocAnalyzer` is an abstract base class
- `TukeyPostHocAnalyzer` implements Tukey HSD
- New post-hoc methods (e.g., Bonferroni, Holm) can be added without modifying existing code

**Example Extension**:
```python
class BonferroniPostHocAnalyzer(PostHocAnalyzer):
    def analyze(self, data, covariates):
        # Implement Bonferroni correction
        pass
```

### 3. Liskov Substitution Principle (LSP)

All implementations of abstract classes can be substituted:

- Any `DataValidator` can be used in `ANCOVAAnalyzer`
- Any `DataPreparator` can be used interchangeably
- Contracts defined by base classes are honored by implementations

### 4. Interface Segregation Principle (ISP)

Interfaces are specific and focused:

- `DataValidator` only requires `validate()` method
- `DataPreparator` only requires `prepare()` method
- Clients don't depend on methods they don't use

### 5. Dependency Inversion Principle (DIP)

High-level modules depend on abstractions, not concretions:

- `ANCOVAAnalyzer` depends on abstract `DataValidator`, not `StandardDataValidator`
- Concrete implementations are injected at runtime
- Easy to swap implementations for testing or different behaviors

## Benefits of This Architecture

### Testability

Each component can be tested independently:

```python
# Test validator in isolation
validator = StandardDataValidator()
validator.validate(test_data, test_covariates)

# Test with mock preparator
mock_preparator = Mock(spec=DataPreparator)
analyzer = ANCOVAAnalyzer(validator, mock_preparator, model)
```

### Extensibility

New statistical methods can be added easily:

```python
class RobustANCOVA(ANCOVAAnalyzer):
    """ANCOVA with robust standard errors."""
    
    def __init__(self):
        super().__init__(
            validator=StandardDataValidator(),
            preparator=PandasDataPreparator(),
            model=RobustModel(),  # New model implementation
        )
```

### Maintainability

- Clear separation of concerns makes code easier to understand
- Changes are localized to specific components
- Dependencies are explicit and manageable

## Backward Compatibility

The public API remains unchanged:

```python
# Original API still works
from ancovapy import ANCOVA

ancova = ANCOVA(ss_type=2)
result = ancova.fit(data, covariates)
```

Internal refactoring improves code quality without breaking existing code.

## Future Enhancements

The architecture supports future extensions:

1. **Alternative Post-Hoc Tests**:
   - Bonferroni correction
   - Holm-Bonferroni method
   - Games-Howell test for unequal variances

2. **Robust Methods**:
   - Heteroscedasticity-consistent standard errors
   - Bootstrap confidence intervals
   - Permutation tests

3. **Mixed Models**:
   - Random effects
   - Hierarchical models
   - Repeated measures

4. **Additional Validation**:
   - Assumption checking (homoscedasticity, normality)
   - Influence diagnostics
   - Multicollinearity detection

## References

- Martin, R. C. (2000). *Design principles and design patterns*. Object Mentor, 1(34), 597.
- Martin, R. C. (2017). *Clean architecture: A craftsman's guide to software structure and design*. Prentice Hall.
- Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). *Design patterns: Elements of reusable object-oriented software*. Addison-Wesley.
