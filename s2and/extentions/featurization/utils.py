import numpy as np
from typing import List, Tuple, Union, Dict
from s2and.extentions.featurization.featurizer import Featurizer
from s2and import logger


def get_matrices(
    datasets: List[str],
    features: List[Dict[str, str]],
    remove_nan: bool = True,
    external_emb_dir: Union[str, None] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    '''
    Featurize multiple datasets and return the combined matrix
    :param datasets: list of s2and datasets to extract features from
    :param features: feature list as appeared in the configurations files
    :param remove_nan: whether to remove missing values or not
    :param external_emb_dir: directory of external embeddings
    :return: features matrices, label matrices X, y and missing values dict
    '''
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    for dataset_name in datasets:
        featurizer = Featurizer(
            dataset_name=dataset_name,
            features=features,
            embeddings_dir=external_emb_dir
        )
        X, y = featurizer.get_feature_matrix('train')
        X_train.append(X)
        y_train.append(y)
        X, y = featurizer.get_feature_matrix('val')
        X_val.append(X)
        y_val.append(y)
        X, y = featurizer.get_feature_matrix('test')
        X_test.append(X)
        y_test.append(y)
        logger.info(f'Processed {dataset_name}')

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    X_val = np.vstack(X_val)
    y_val = np.concatenate(y_val)
    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)
    # Count nan per column for train set
    nan_counts = [count for count in np.count_nonzero(np.isnan(X_train), axis=0)]
    # Remove nan values
    if remove_nan:
        np.nan_to_num(X_train, copy=False)
        np.nan_to_num(X_test, copy=False)
        np.nan_to_num(X_val, copy=False)
    return X_train, y_train, X_val, y_val, X_test, y_test, nan_counts
