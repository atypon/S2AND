import numpy as np
import lightgbm as lgb
from typing import Union, List


class LightGBMWrapper():
    """
    Wrapper for lightgbm model, implementing predict_distance method
    necessary for Clusterer objects to work
    """

    def __init__(self, model: lgb.LGBMClassifier) -> None:
        """Initalizes class"""
        self.model = model

    def predict_distance(self, X: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """
        Predicts probability of featurized pair to originate from different entity
        """
        if isinstance(X, list):
            X = np.asarray(X)
        return self.model.predict_proba(X)[:, 0]
