import torch
from typing import Any, Tuple
from torch.nn import functional as F
from Levenshtein import distance as levenshtein_distance
from s2and.extentions.featurization.operations_registry import Registry
from s2and.extentions.featurization.base_operation import BaseOperation


@Registry.register_operation(operation_name='equality')
class Equality(BaseOperation):
    """
    Subclass implementing the equality operation
    """

    def calculate(self, values: Tuple[Any]) -> float:
        val1, val2 = values
        return 1 if val1 == val2 else 0


@Registry.register_operation(operation_name='levenshtein')
class Levenshtein(BaseOperation):
    """
    Subclass implementing the levenshtein distance operation
    """

    def calculate(self, values: Tuple[Any]) -> float:
        self._check_type(values, str)
        val1, val2 = values
        return levenshtein_distance(val1, val2)


@Registry.register_operation(operation_name='jaccard_similarity')
class JaccardSimilarity(BaseOperation):
    """
    Subclass implementing jaccard similarity operation
    """

    def calculate(self, values: Tuple[Any]) -> float:
        self._check_type(values, list)
        val1, val2 = values
        val1 = set(val1)
        val2 = set(val2)
        if len(val1.union(val2)) > 0:
            return len(val1.intersection(val2)) / len(val1.union(val2))
        return 0


@Registry.register_operation(operation_name='cosine_similarity')
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
