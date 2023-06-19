import json
import torch
import numpy as np
from collections import defaultdict
from os.path import join
from typing import Tuple, Any, Union, List, Dict
from tqdm import tqdm
from joblib import load
from s2and.data import ANDData
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from s2and.extentions.utils import load_signatures
from s2and.extentions.featurization.operations import Registry


class Clusterer():
    """
    Responsible for clustering the block_dict given as input
    to predict method.
    """

    def __init__(
        self,
        combined_classifier: str,
        dataset_name: str,
        features: List[Dict[str, str]],
        embeddings_dir: Union[str, None] = None,
        clusterer: str = 'dbscan'
    ) -> None:

        self.signatures = load_signatures(dataset_name)
        self.features = features
        if embeddings_dir is not None:
            self.embeddings_path = join(embeddings_dir, dataset_name,
                                        f'{dataset_name}_embeddings.json')
            with open(self.embeddings_path) as f:
                self.paper_ids_to_emb = json.load(f)
        else:
            self.paper_ids_to_emb = None

        self.model = load(combined_classifier)
        self.clusterer_name = clusterer

        if clusterer == 'dbscan':
            self.clusterer = DBSCAN(eps=0.4, min_samples=1, metric='precomputed', n_jobs=-1)
        elif clusterer == 'agglomerative':
            self.clusterer = AgglomerativeClustering(
                n_clusters=None,
                affinity='precomputed',
                distance_threshold=0.4,
                linkage='average'
            )

    @torch.inference_mode()
    def get_distance_matrix(self, block: List[str]) -> np.ndarray:

        n_signatures = len(block)
        d_matrix = np.zeros((n_signatures, n_signatures))

        features = []
        feature_idx = 0
        # Maps feature list potition to location in d_matrix
        # Used for placing results in correct dmatrix position
        mapper = {}
        for i in range(0, n_signatures):
            for j in range(0, n_signatures):
                if i > j:
                    continue
                # This is the same signature
                if i == j:
                    d_matrix[i, j] = 0
                # This is non the same signature
                else:
                    if block[i] in self.signatures and block[j] in self.signatures:
                        sig1 = self.signatures[block[i]]
                        sig2 = self.signatures[block[j]]
                        if self.paper_ids_to_emb is not None:
                            # Replace default embedding
                            sig1['vector'] = self.paper_ids_to_emb[str(sig1['paper_id'])]
                            sig2['vector'] = self.paper_ids_to_emb[str(sig2['paper_id'])]
                        features.append(self.__featurize_pair(signature_pair=(sig1, sig2)))
                        mapper[feature_idx] = (i, j)
                        feature_idx += 1
                    # In case we have no info for any of the two signatures
                    # we set their distance to 1
                    else:
                        d_matrix[i, j] = 1

        # Having featurized every sign combination, we calculate
        # distances in a batch-way to save time
        if len(features) > 0:
            distances = self.model.predict_distance(features)
            for i in range(distances.shape[0]):
                d_matrix[mapper[i]] = distances[i].item()

        return d_matrix + np.transpose(d_matrix)

    def get_dmatrix_dict(self, block_dict: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        Calculates distance matrices for each block in the block_dict
        """
        block_to_dmatrix = {}
        for block_name in tqdm(block_dict):
            block = block_dict[block_name]
            dmatrix = self.get_distance_matrix(block)
            block_to_dmatrix[block_name] = dmatrix
        return block_to_dmatrix

    def predict(
        self,
        block_dict: Dict[str, List[str]],
        block_to_dmatrix: Dict[str, np.ndarray] = None
    ) -> Dict[str, str]:
        """
        Predicts clusters for signatures in block_dict
        """
        sign_to_pred_clusters = {}
        for block_name in block_dict:
            block = block_dict[block_name]
            if block_to_dmatrix is None:
                dmatrix = self.get_distance_matrix(block)
            else:
                dmatrix = block_to_dmatrix[block_name]
            # Resolve issue with dmatrix of 1 element in agglomerative clustering
            if self.clusterer_name == 'agglomerative' and dmatrix.shape == (1, 1):
                clusters = [0]
            else:
                clusters = self.clusterer.fit_predict(dmatrix)

            max_identifier = max(clusters)
            for signature, cluster in zip(block, clusters):

                if cluster == -1:
                    max_identifier += 1
                    sign_to_pred_clusters[signature] = block_name + '_' + str(max_identifier)
                else:
                    sign_to_pred_clusters[signature] = block_name + '_' + str(cluster)
        return sign_to_pred_clusters

    def __featurize_pair(self, signature_pair: Tuple[Dict[str, Any]]) -> List[Union[int, float]]:
        """
        Returns complete feature vector for the given signature pair
        :param signature_pair: pair of signatures to be featurized
        :return: feature vector
        """
        feature_vector = []
        for feature in self.features:
            operation_name = feature['operation']
            field = feature['field']
            operation = Registry.get_operation(operation_name=operation_name)
            if 'operation_args' in feature:
                operation_args = feature['operation_args']
                feature_vector.append(
                    operation(signature_pair=signature_pair, field=field, **operation_args)
                )
            else:
                feature_vector.append(operation(signature_pair=signature_pair, field=field))


class DummyClusterer():
    """
    Dummy class used due to compatibilty issues with cluster_eval function
    which is provided by S2AND. Must implement predict method that maps
    cluster names to their signatures
    """

    def __init__(self, source) -> None:
        """
        Inits class by loading the clustering result files
        """
        if isinstance(source, str):
            with open(source) as f:
                self.sign_to_pred_cluster = json.load(f)
        else:
            self.sign_to_pred_cluster = source

    def predict(
        self,
        block_dict: Dict[str, List[str]],
        dataset: ANDData,
        use_s2_clusters: bool
    ) -> Dict[str, List[str]]:
        """
        Creates dict that maps cluster name to the signatures it owns
        """
        pred_clusters = defaultdict(list)
        uknown_id = 0
        for signatures in block_dict.values():
            for signature in signatures:
                if signature not in self.sign_to_pred_cluster:
                    pred_clusters[f'uknown_{uknown_id}'] = [signature]
                    uknown_id += 1
                else:
                    cluster = self.sign_to_pred_cluster[signature]
                    pred_clusters[cluster].append(signature)
        return pred_clusters, None
