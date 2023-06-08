import numpy as np
from abc import abstractmethod
from typing import Any, Dict, Tuple


class BaseOperation():
    """
    Base class for each featurization operation
    """

    def __call__(
        self,
        signature_pair: Tuple[Dict[str, Any], Dict[str, Any]],
        field: str
    ) -> float:
        """
        Performs the operation with proper checking
        :param signature_pair: pair of signatures to be featurized
        :param field: field of signatures for featurization
        :return: calculated value
        """
        sig1, sig2 = signature_pair
        value1 = sig1.get(field, None)
        value2 = sig2.get(field, None)
        if value1 is None or value2 is None:
            return np.nan
        else:
            return self.calculate(values=(value1, value2))

    @abstractmethod
    def calculate(self, values: Tuple[Any]) -> float:
        """
        Performs the calculation of the operation
        :param values: tuple of values to be featurized
        :return: calculated value
        """
        pass

    @staticmethod
    def _check_type(values: Tuple[Any, Any], type: Any) -> None:
        """
        Check if both values have the desired type
        :param values: tuple of values to be featurized
        :param type: desired type
        """
        val1, val2 = values
        if not isinstance(val1, type) or not isinstance(val2, type):
            raise Exception(f'{values} are of undesired type')
