from collections.abc import Sequence
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal
import numpy as np
from numpy.typing import NDArray

@dataclass
class Variable(ABC):
    name: str
    type: Literal["quantitative", "categorical"]
    
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

class QuantitativeVariable(Variable):
    """Domain class for quantitative variables.

    Directly holds data as a numpy array of type np.float64.
    """

    data: NDArray[np.float64]

    def __init__(
        self,
        name: str,
        data: Sequence[Any],
    ):
        self.name = name
        self.data = self._transform_type(data)
        self.type = "quantitative"

    def _transform_type(
        self,
        data: Sequence[Any],
    ) -> NDArray[np.float64]:
        try:
            # TODO: implemennt type casting to np.float64
            raise NotImplementedError()

        except ValueError as e:
            raise ValueError(
                f"Data for quantitative variable '{self.name}' must be convertible to numpy.float64."
            ) from e

    def __len__(self) -> int:
        return len(self.data)

class CategoricalVariable(Variable):
    """Domain class for categorical variables.
    
    This holds categorical data as one-hot encoded numpy matrix
    """
    
    data: NDArray[np.int8]

    def __init__(
        self,
        name: str,
        data: Sequence[Any],
    ):
        self.name = name
        self.data = self._transform_type(data)
        self.type = "categorical"
    
    def _transform_type(
        self,
        data: Sequence[Any],
    ) -> NDArray[np.int8]:
        # TODO: implement one-hot encoding to np.int8 matrix
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.data)
    
    @property
    def categories(self) -> int:
        """Returns the number of categories in this categorical variable."""
        return self.data.shape[1]
