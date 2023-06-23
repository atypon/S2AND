import joblib
import json
from math import isnan
from s2and.extentions.classification_models import LightGBMWrapper
from s2and.utils.configs import load_configurations
from s2and.extentions.featurization.featurizer import Featurizer


def main(cfg):

    featurizer = Featurizer(
        dataset_name=cfg.dataset,
        features=cfg.features,
        embeddings_dir=None
    )
    model: LightGBMWrapper = joblib.load(cfg.model_path)
    # Load signature pairs in featurize them
    train_sig, val_sig, test_sig = featurizer.dataset.split_cluster_signatures()
    train_pairs, _, _ = featurizer.dataset.split_pairs(train_sig, val_sig, test_sig)
    json_file = {
        'features': [f"{feature['operation']}({feature['field']})" for feature in cfg.features],
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
        if len(json_file['examples']) == cfg.num_pairs:
            break
    with open(cfg.result_path, 'w') as f:
        json.dump(json_file, f)


if __name__ == "__main__":

    cfg = load_configurations(path='configs/test_file_conf.yaml')
    main(cfg)
