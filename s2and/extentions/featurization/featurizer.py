import json
from os.path import join
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from s2and.extentions.featurization.operations import Registry
from s2and.extentions.utils import load_dataset, load_signatures


class Featurizer:
    """
    Class for creating numpy arrays containg the features for each signature pair
    """

    def __init__(
        self,
        dataset_name: str,
        features: List[Dict[str, str]],
        embeddings_dir: Union[None, str] = None,
    ) -> None:
        """
        Initializes Featurizer object, loading datasets and signatures.
        If parse_specter = true, it parses the provided specter embeddings
        for them to be used in the featurization process.
        :param dataset_name: Name of the dataset
        :param features: List of features to use
        :param embeddings_dir: Directory of the embeddings
        """
        self.dataset = load_dataset(dataset_name)
        self.extended_signatures = load_signatures(dataset_name)
        self.features = features
        if embeddings_dir is not None:
            with open(
                join(embeddings_dir, f"{dataset_name}_embeddings.json")
            ) as embeddings_file:
                self.paper_ids_to_emb = json.load(embeddings_file)
        else:
            self.paper_ids_to_emb = None

    def featurize_pairs(self, pairs: Tuple[Dict[str, dict]]) -> Tuple[np.ndarray]:
        """
        Given the list of pairs return the matrix of features and labels
        :param pairs: List of pairs to featurize
        :return: Tuple of features and labels
        """
        X = []
        y = []
        for pair in pairs:
            # Make sure the extended dataset contains the signatures
            #  of the pair, else do not featurize the pair
            if (
                pair[0] in self.extended_signatures
                and pair[1] in self.extended_signatures
            ):
                y.append(pair[2])
                sig1 = self.extended_signatures[pair[0]]
                sig2 = self.extended_signatures[pair[1]]
                if self.paper_ids_to_emb is not None:
                    sig1["external_vector"] = self.paper_ids_to_emb[
                        str(sig1["paper_id"])
                    ]
                    sig2["external_vector"] = self.paper_ids_to_emb[
                        str(sig2["paper_id"])
                    ]
                X.append(self.featurize_pair(signature_pair=(sig1, sig2)))
        return np.asarray(X), np.asarray(y)

    def get_feature_matrix(self, split: str) -> Tuple[np.ndarray]:
        """
        Get dataset's featurized pairs matrix by specifying the desired split
        :param split: Split to get the feature matrix for
        :return: Tuple of features and labels
        """
        train_sig, val_sig, test_sig = self.dataset.split_cluster_signatures()
        train_pairs, val_pairs, test_pairs = self.dataset.split_pairs(
            train_sig, val_sig, test_sig
        )
        if split == "train":
            return self.featurize_pairs(train_pairs)
        elif split == "val":
            return self.featurize_pairs(val_pairs)
        elif split == "test":
            return self.featurize_pairs(test_pairs)

    def featurize_pair(
        self, signature_pair: Tuple[Dict[str, Any]]
    ) -> List[Union[int, float]]:
        """
        Returns complete feature vector for the given signature pair
        :param signature_pair: pair of signatures to be featurized
        :return: feature vector
        """
        feature_vector = []
        for feature in self.features:
            operation_name = feature["operation"]
            field = feature["field"]
            operation = Registry.get_operation(operation_name=operation_name)
            if "operation_args" in feature:
                operation_args = feature["operation_args"]
                feature_vector.append(
                    operation(
                        signature_pair=signature_pair, field=field, **operation_args
                    )
                )
            else:
                feature_vector.append(
                    operation(signature_pair=signature_pair, field=field)
                )
        return feature_vector
