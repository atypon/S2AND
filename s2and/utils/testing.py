import json
from math import isnan
from typing import List, Dict, Union
from s2and.extentions.classification_models import LightGBMWrapper
from s2and.extentions.featurization.featurizer import Featurizer


def create_test_file(
    dataset_name: str,
    features: List[Dict[str, Union[str, bool]]],
    model: LightGBMWrapper,
    embeddings_dir: str,
    num_pairs: int,
    test_file_path: str
) -> None:
    """
    Create test file for implementing unit test to check model validity
    :param dataset_name: s2and subdataset to create test file from
    :param features: feature list from config file
    :param model: the trained lightgbm model wrapper class
    :param embeddings_dir: path to external embeddings file
    """
    featurizer = Featurizer(
        dataset_name=dataset_name,
        features=features,
        embeddings_dir=embeddings_dir
    )
    train_sig, val_sig, test_sig = featurizer.dataset.split_cluster_signatures()
    train_pairs, _, _ = featurizer.dataset.split_pairs(train_sig, val_sig, test_sig)
    json_file = {
        'features': [f"{feature['operation']}({feature['field']})" for feature in features],
        'examples': []
    }
    # Fill examples of signature pairs until we reach the desired number
    for sig_id1, sig_id2, _ in train_pairs:
        if sig_id1 in featurizer.extended_signatures and sig_id2 in featurizer.extended_signatures:
            sig1 = featurizer.extended_signatures[sig_id1]
            sig2 = featurizer.extended_signatures[sig_id2]
            feature_vector = featurizer.featurize_pair(signature_pair=(sig1, sig2))
            distance = model.predict_distance([feature_vector]).item()
            json_file['examples'].append(
                {
                    'signature1': sig1,
                    'signature2': sig2,
                    'feature_vector': [
                        'NaN' if isnan(feature) else feature for feature in feature_vector
                    ],
                    'distance': distance
                }
            )
        if len(json_file['examples']) == num_pairs:
            break
    with open(test_file_path, 'w') as f:
        json.dump(json_file, f)
