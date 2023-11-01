import json
from typing import List, Dict, Union, Any

from hyperopt import hp
from hyperopt.pyll import scope
import numpy as np

from s2and.data import ANDData
from s2and.eval import cluster_eval
from s2and.extentions.clustering_models import Clusterer, DummyClusterer


class Objective():
    """
    Class that implements the optimization objective for the clustering
    step of the author name disambiguation process
    """

    def __init__(
        self,
        dataset_names: List[str],
        datasets: List[ANDData],
        clusterers: List[Clusterer],

    ) -> None:
        """
        Initiliazes class with important data
        :param dataset_name: list of s2and dataset names to be used
        :param dataset: list of loaded s2and datasets
        :param clusterers: list of loaded clusterers
        """
        self.dataset_names = dataset_names
        self.datasets = datasets
        self.clusterers = clusterers
        # These contain dicts that link block names to the actual block of sugnatures
        self.val_block_dicts = []
        self.test_block_dicts = []
        # These contain dicts that link block to its dmatrix
        self.val_dmatrices = []
        self.test_dmatrices = []
        for dataset, clusterer in zip(self.datasets, self.clusterers):
            _, val_block_dict, test_block_dict = dataset.split_blocks_helper(dataset.get_blocks())
            self.val_block_dicts.append(val_block_dict)
            self.test_block_dicts.append(test_block_dict)
            self.val_dmatrices.append(clusterer.get_dmatrix_dict(val_block_dict))
            self.test_dmatrices.append(clusterer.get_dmatrix_dict(test_block_dict))

    def __call__(self, params: Dict[str, Union[int, float]]) -> float:
        """
        Called by hyperopt to perform an optimization step
        :param params: parameter dictionary as provided by hyperopt
        :return: calculated score of the run
        """
        f1s = []
        # Weights are the size of each dataset. The will be used for proper score averaging
        weights = [len(clusterer.signatures.keys()) for clusterer in self.clusterers]
        # Calculate f1 for each dataset using its clusterer
        zipped_elements = zip(self.datasets, self.val_block_dicts, self.val_dmatrices,
                              self.clusterers)
        for dataset, val_block_dict, val_block_to_dmatrix, clusterer in zipped_elements:
            clusterer.clusterer.set_params(**params)
            sign_to_pred_clusters = clusterer.predict(val_block_dict, val_block_to_dmatrix)
            # Performing evaluation after parsing results from file
            dummy_clusterer = DummyClusterer(sign_to_pred_clusters)
            metrics, _ = cluster_eval(dataset, dummy_clusterer, split='val')
            f1s.append(metrics['B3 (P, R, F1)'][2])
        # hp module is minimizing so return minus f1 to get it maximized
        return -(np.asarray(f1s) * np.asarray(weights)).sum()/np.sum(weights)

    def evaluate_best(self, best_params: Dict[str, Union[int, float]]) -> Dict[str, Any]:
        """
        Evaluates clusterers on datasets for the given set of optimal params
        :best best_params: best parapemetrs are retrieved by hyperopt
        :return: dictionary containing metrics per dataset
        """
        zipped_elements = zip(self.test_block_dicts, self.test_dmatrices, self.clusterers,
                              self.dataset_names, self.datasets)
        results_per_dataset = {}
        for test_block_dict, test_block_to_dmatrix, clusterer, \
                dataset_name, dataset in zipped_elements:
            clusterer.clusterer.set_params(**best_params)
            sign_to_pred_clusters = clusterer.predict(test_block_dict, test_block_to_dmatrix)
            # Save clustered data
            with open(f'clustering_results/{dataset_name}.json', 'w') as f:
                json.dump(sign_to_pred_clusters, f)
            # Performing evaluation using DummyClusterer (used for compatibility with s2and)
            dummy_clusterer = DummyClusterer(sign_to_pred_clusters)
            metrics, _ = cluster_eval(dataset, dummy_clusterer, split='test')
            results_per_dataset[dataset_name] = metrics
        return results_per_dataset

    @staticmethod
    def get_search_space(clusterer: str) -> Dict[str, Any]:
        """
        Depending on clusterer, return the appropriate search space
        """
        if clusterer == 'agglomerative':
            search_space = {
                'distance_threshold': hp.uniform('distance_threshold', 0, 1),
            }
        elif clusterer == 'dbscan':
            search_space = {
                'eps': hp.uniform('eps', 0, 1),
                'min_samples': scope.int(hp.quniform('min_samples', 1, 5, q=1))
            }
        return search_space
