import mlflow
import json
import numpy as np
from typing import List, Dict, Union
from s2and.data import ANDData
from s2and.eval import cluster_eval
from s2and_ext.clustering_models import Clusterer, DummyClusterer

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
        """Initiliazes class with important data"""
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
        """Called by hyperopt to perform an optimization step"""
        f1s = []
        # Weights are the size of each dataset. The will be used for proper score averaging
        weights = [len(clusterer.signatures.keys()) for clusterer in self.clusterers]
        # Calculate f1 for each dataset using its clusterer
        zipped_elements = zip(self.datasets, self.val_block_dicts, self.val_dmatrices, self.clusterers)
        for dataset, val_block_dict, val_block_to_dmatrix, clusterer in zipped_elements:
            clusterer.clusterer.set_params(**params)
            sign_to_pred_clusters = clusterer.predict(val_block_dict, val_block_to_dmatrix)
            # Performing evaluation after parsing results from file
            dummy_clusterer = DummyClusterer(sign_to_pred_clusters)
            metrics, _ = cluster_eval(dataset, dummy_clusterer, split='val')
            f1s.append(metrics['B3 (P, R, F1)'][2])
        # hp module is minimizing so return minus f1 to get it maximized
        return -(np.asarray(f1s) * np.asarray(weights)).sum()/np.sum(weights)

    def evaluate_best(self, best_params: Dict[str, Union[int, float]]) -> None:
        """Evaluates clusterers on datasets for the given set of optimal params"""
        zipped_elements = zip(self.test_block_dicts, self.test_dmatrices, self.clusterers, self.dataset_names, self.datasets)
        for test_block_dict, test_block_to_dmatrix, clusterer, dataset_name, dataset in zipped_elements:
            clusterer.clusterer.set_params(**best_params)
            sign_to_pred_clusters = clusterer.predict(test_block_dict, test_block_to_dmatrix)
            with open(f'clustering_results/{dataset_name}.json', 'w') as f:
                json.dump(sign_to_pred_clusters, f)
            # Performing evaluation after parsing results from file
            dummy_clusterer = DummyClusterer(sign_to_pred_clusters)
            metrics, metrics_per_signature = cluster_eval(dataset, dummy_clusterer, split='test')
            mlflow_metrics = {
                f'{dataset_name} B3 P' : metrics['B3 (P, R, F1)'][0],
                f'{dataset_name} B3 R' : metrics['B3 (P, R, F1)'][1],
                f'{dataset_name} B3 F1' : metrics['B3 (P, R, F1)'][2],
            }
            mlflow.log_metrics(mlflow_metrics)
            print(dataset_name)
            print(metrics)