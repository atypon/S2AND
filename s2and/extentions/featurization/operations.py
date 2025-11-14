from typing import Any, Dict, Tuple

import numpy as np
import torch
from jarowinkler import jarowinkler_similarity
from Levenshtein import distance as levenshtein_distance
from torch.nn import functional as F

from s2and.extentions.featurization.base_operation import BaseOperation
from s2and.extentions.featurization.operations_registry import Registry


@Registry.register_operation(operation_name="equality")
class Equality(BaseOperation):
    """
    Subclass implementing the equality operation
    """

    def calculate(self, values: Tuple[Any]) -> float:
        val1, val2 = values
        return True if val1 == val2 else False


@Registry.register_operation(operation_name="levenshtein")
class Levenshtein(BaseOperation):
    """
    Subclass implementing the levenshtein distance operation
    """

    def calculate(self, values: Tuple[Any]) -> float:
        self._check_type(values, str)
        val1, val2 = values
        return levenshtein_distance(val1, val2)


@Registry.register_operation(operation_name="jaccard_similarity")
class JaccardSimilarity(BaseOperation):
    """
    Subclass implementing jaccard similarity operation
    """

    def calculate(self, values: Tuple[Any]) -> float:
        self._check_type(values, list)
        val1, val2 = values
        val1 = set(val1)
        val2 = set(val2)
        if len(val1) == 0 or len(val2) == 0:
            return np.nan
        if len(val1.union(val2)) > 0:
            return len(val1.intersection(val2)) / len(val1.union(val2))
        return 0


@Registry.register_operation(operation_name="cosine_similarity")
class CosineSimilarity(BaseOperation):
    """
    Subclass implementing cosine similarity operation
    """

    def calculate(self, values: Tuple[Any]) -> float:
        val1, val2 = values
        self._check_type(values, list)
        val1 = torch.Tensor(val1)
        val2 = torch.Tensor(val2)
        return F.cosine_similarity(val1, val2, dim=0).item()


@Registry.register_operation(operation_name="absolute_difference")
class AbsoluteDifference(BaseOperation):
    """
    Subclass implementing absolute difference operation
    """

    def calculate(self, values: Tuple[Any]) -> float:
        val1, val2 = values
        # Convert possible int or str to float
        val1 = float(val1)
        val2 = float(val2)
        return abs(val1 - val2)


@Registry.register_operation(operation_name="normalized_by_authors_absolute_difference")
class NormalizedByAuthorsAbsoluteDifference(BaseOperation):
    """
    Subclass implementing normalized absolute difference
    """

    def __call__(
        self,
        signature_pair: Tuple[Dict[str, Any], Dict[str, Any]],
        field: str,
        normalization_field: str,
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
        norm_factor_1 = sig1.get(normalization_field, None)
        norm_factor_2 = sig2.get(normalization_field, None)
        if (
            value1 is None
            or value2 is None
            or norm_factor_1 is None
            or norm_factor_2 is None
        ):
            return np.nan
        norm_factor_1 = len(norm_factor_1)
        norm_factor_2 = len(norm_factor_2)
        # Take into account zero devision
        if norm_factor_1 == 0 or norm_factor_2 == 0:
            return np.nan
        return self.calculate(
            values=(value1, value2), factors=(norm_factor_1, norm_factor_2)
        )

    def calculate(self, values: Tuple[Any], factors: Tuple[int, int]) -> float:
        """
        Takes input the normalized values by nummber of
        coauthors and performs the calculation
        """
        val1, val2 = values
        factor1, factor2 = factors
        # Convert possible int or str to float
        val1 = float(val1) / factor1
        val2 = float(val2) / factor2
        return abs(val1 - val2)


@Registry.register_operation(operation_name="random_noise")
class RandomNoise(BaseOperation):
    """
    Subclass for implementing random noise feature
    """

    def __call__(
        self,
        signature_pair: Tuple[Dict[str, Any], Dict[str, Any]],
        field: str,
    ) -> float:
        """
        Performs the operation with proper checking
        :param signature_pair: pair of signatures to be featurized
        :param field: field of signatures for featurization
        :return: calculated value
        """
        return self.calculate()

    def calculate(self) -> float:
        """
        Return random noise
        """
        return np.random.normal(loc=0, scale=0.5)


@Registry.register_operation(operation_name="jarowinkler")
class JaroWinkler(BaseOperation):
    """
    Subclass implementing Jaro Winkler similarity
    """

    def calculate(self, values: Tuple[str, str]) -> float:
        """
        Performs the calculation of the jarowinkler similarity
        :param values: tuple of values to be featurized
        :return: calculated value
        """
        self._check_type(values, str)
        val1, val2 = values
        return jarowinkler_similarity(val1, val2)


@Registry.register_operation(operation_name="jarowinkler_no_surname")
class JaroWinklerNoSurname(BaseOperation):
    """
    Subclass implementing Jaro Winkler similarity for full name field.
    This means that by default it subtracts the surname since it is always
    the same inside a block
    """

    def calculate(self, values: Tuple[str, str]) -> float:
        """
        Performs the calculation of the jarowinkler similarity
        after removing surname
        :param values: tuple of values to be featurized
        :return: calculated value
        """
        self._check_type(values, str)
        val1, val2 = values
        # Surname can be obtained from either val1 or val2
        surname = val1.split(" ")[-1]
        val1 = val1.replace(" " + surname, "")
        val2 = val2.replace(" " + surname, "")
        # cases like val1 = "k" and val2 = "k" return 0
        # cases like val1 = "k" and val2 = "a" return 1
        return jarowinkler_similarity(val1, val2)


@Registry.register_operation(operation_name="membership_check")
class MembershipCheck(BaseOperation):
    """
    Subclass implementing membership check operation
    """

    def __call__(
        self,
        signature_pair: Tuple[Dict[str, Any], Dict[str, Any]],
        field: str,
        membership_field: str,
    ) -> float:
        """
        Performs the operation with proper checking
        :param signature_pair: pair of signatures to be featurized
        :param field: field on which membership check is performed
        :param membership_field: field of signatures for membership check
        :return: calculated value
        """
        sig1, sig2 = signature_pair
        value1 = sig1.get(field, None)
        value2 = sig2.get(field, None)
        sig1_id = sig1.get(membership_field, None)
        sig2_id = sig2.get(membership_field, None)
        if value1 is None or value2 is None or sig1_id is None or sig2_id is None:
            return np.nan
        if len(value1) == 0 or len(value2) == 0:
            return np.nan
        return self.calculate(values=(value1, value2), ids=(sig1_id, sig2_id))

    def calculate(self, values: Tuple[Any], ids: Tuple[int, int]) -> float:
        """
        Takes input the normalized values by nummber of
        coauthors and performs the calculation
        """
        val1, val2 = values
        id1, id2 = ids
        if id1 in val2 or id2 in val1:
            return 1
        return 0
