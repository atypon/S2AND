import torch
import numpy as np
from typing import Dict, List, Tuple, Callable, Union
from Levenshtein import distance
from torch.nn import functional as F
from s2and.extentions.featurization.featurizer import Featurizer


def extract_feature(
    sig1: Dict[str, dict],
    sig2: Dict[str, dict],
    attribute: str
) -> float:
    """
    Returns 1 if attribute is equal in sig1 and sig2 otherwise 0.
    If attribute is missing, returns np.nan
    """
    if (sig1.get(attribute, None) is not None) and (sig2.get(attribute, None) is not None):
        if sig1[attribute] == sig2[attribute]:
            return 1
        return 0
    return np.nan


def cosine_sim(
    sig1: Dict[str, dict],
    sig2: Dict[str, dict]
) -> float:
    """
    Computes cosine similarity of paperVector field of signature.
    If field is missing, returns np.nan
    """
    v1 = sig1.get('paperVector', None)
    v2 = sig2.get('paperVector', None)
    if (v1 is not None) and (v2 is not None):
        v1 = torch.Tensor(v1)
        v2 = torch.Tensor(v2)
        return F.cosine_similarity(v1, v2, dim=0).item()
    return np.nan


def name_distance(
    sig1: Dict[str, dict],
    sig2: Dict[str, dict]
) -> float:
    """
    Computes levenshtein distance of s2AuthorName attribute.
    If attribute is missing, return np.nan
    """
    v1 = sig1.get('s2AuthorName', None)
    v2 = sig2.get('s2AuthorName', None)
    if (v1 is not None) and (v2 is not None):
        return distance(v1, v2)
    return np.nan


def jaccard(
    sig1: Dict[str, dict],
    sig2: Dict[str, dict],
    attribute: str
) -> float:
    """
    Computes jaccard similarity of list attribute between two signatures.
    If attribute is missing, returns np.nan
    """
    v1 = sig1.get(attribute, None)
    v2 = sig2.get(attribute, None)
    if (v1 is not None) and (v2 is not None):
        v1 = set(v1)
        v2 = set(v2)
        if len(v1.union(v2)) > 0:
            return len(v1.intersection(v2)) / len(v1.union(v2))
        return 0
    return np.nan


def featurizing_function(
    sig1: Dict[str, dict],
    sig2: Dict[str, dict]
) -> List[float]:
    """
    Calculates the feature vector of two given signature dicts
    """
    features = []
    features.append(cosine_sim(sig1, sig2))
    features.append(name_distance(sig1, sig2))
    features.append(jaccard(sig1, sig2, 'affiliationIds'))
    features.append(jaccard(sig1, sig2, 'conceptIdsLevel0'))
    features.append(jaccard(sig1, sig2, 'conceptIdsLevel1'))
    features.append(jaccard(sig1, sig2, 'oaCoAuthorNormNames'))

    # Ids for later use from the ensemble
    features.append(extract_feature(sig1, sig2, 'oaAuthorId'))
    features.append(extract_feature(sig1, sig2, 's2AuthorId'))
    features.append(extract_feature(sig1, sig2, 'orcId'))
    return features


def get_matrices(
    datasets: List[str],
    featurizing_function: Callable,
    remove_nan: bool = True,
    external_emb_dir: Union[str, None] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Featurize multiple datasets and return the combined matrix
    If no featurizing_function is given, the default will be used
    '''
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    for dataset_name in datasets:
        featurizer = Featurizer(dataset_name=dataset_name,
                                featurizing_function=featurizing_function,
                                embeddings_dir=external_emb_dir)
        X, y = featurizer.get_feature_matrix('train')
        X_train.append(X)
        y_train.append(y)
        X, y = featurizer.get_feature_matrix('val')
        X_val.append(X)
        y_val.append(y)
        X, y = featurizer.get_feature_matrix('test')
        X_test.append(X)
        y_test.append(y)
        print(f'Processed {dataset_name}')

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    X_val = np.vstack(X_val)
    y_val = np.concatenate(y_val)
    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)
    # Count nan per column for train set
    print('Nan values for each feature :')
    print(np.count_nonzero(np.isnan(X_train), axis=0))
    # Remove nan values
    if remove_nan:
        np.nan_to_num(X_train, copy=False)
        np.nan_to_num(X_test, copy=False)
        np.nan_to_num(X_val, copy=False)
    return X_train, y_train, X_val, y_val, X_test, y_test
