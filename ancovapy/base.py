"""Base classes and interfaces for ANCOVA analysis."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from ancovapy.types import Covariate, DependentVariable


@dataclass
class AnalysisResult(ABC):
    """Base class for analysis results."""

    pass


class DataValidator(ABC):
    """Abstract base class for data validation."""

    @abstractmethod
    def validate(
        self, dependent_var: DependentVariable, covariates: dict[str, Covariate]
    ) -> None:
        """
        Validate input data.

        Args:
            dependent_var: Dependent variable array
            covariates: Dictionary of covariates

        Raises:
            ValueError: If validation fails
        """
        pass


class DataPreparator(ABC):
    """Abstract base class for data preparation."""

    @abstractmethod
    def prepare(
        self, dependent_var: DependentVariable, covariates: dict[str, Covariate]
    ) -> Any:
        """
        Prepare data for analysis.

        Args:
            dependent_var: Dependent variable array
            covariates: Dictionary of covariates

        Returns:
            Prepared data structure
        """
        pass


class StatisticalModel(ABC):
    """Abstract base class for statistical models."""

    @abstractmethod
    def fit(self, data: Any) -> AnalysisResult:
        """
        Fit the statistical model.

        Args:
            data: Prepared data

        Returns:
            Analysis result
        """
        pass


class PostHocAnalyzer(ABC):
    """Abstract base class for post-hoc analysis."""

    @abstractmethod
    def analyze(self, data: Any, covariates: dict[str, Covariate]) -> Optional[list[Any]]:
        """
        Perform post-hoc analysis.

        Args:
            data: Prepared data
            covariates: Dictionary of covariates

        Returns:
            Post-hoc results or None if not applicable
        """
        pass


class ANCOVAAnalyzer(ABC):
    """Abstract base class for ANCOVA analyzers."""

    def __init__(
        self,
        validator: DataValidator,
        preparator: DataPreparator,
        model: StatisticalModel,
        posthoc: Optional[PostHocAnalyzer] = None,
    ):
        """
        Initialize analyzer with dependencies.

        Args:
            validator: Data validator
            preparator: Data preparator
            model: Statistical model
            posthoc: Post-hoc analyzer (optional)
        """
        self.validator = validator
        self.preparator = preparator
        self.model = model
        self.posthoc = posthoc

    def fit(
        self, dependent_var: DependentVariable, covariates: dict[str, Covariate]
    ) -> AnalysisResult:
        """
        Fit ANCOVA model.

        Args:
            dependent_var: Dependent variable
            covariates: Dictionary of covariates

        Returns:
            Analysis result
        """
        # Validate
        self.validator.validate(dependent_var, covariates)

        # Prepare
        data = self.preparator.prepare(dependent_var, covariates)

        # Fit model
        result = self.model.fit(data)

        return result
