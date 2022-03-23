import json, yaml, argparse, numpy as np

from os.path import join
from hyperopt import fmin, tpe, hp
from hyperopt.pyll import scope

from s2and.eval import cluster_eval
from s2and_ext.my_utils import load_dataset
from s2and_ext.my_models import DummyClusterer, Clusterer
from s2and_ext.my_featurization import featurizing_function

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--conf',
        default='configs/clusterer_conf.yml',
        help='Path to configuration file'
    )
    args = parser.parse_args()
    with open(args.conf) as f:
        conf = yaml.safe_load(f)

    datasets_names = conf['datasets']
    datasets = [load_dataset(dataset_name) for dataset_name in datasets_names]
    clusterers = [Clusterer(combined_classifier=conf['model_source'],
                            dataset_name=dataset_name,
                            featurization_function=featurizing_function,
                            default_embeddings=conf['default_embeddings'],
                            embeddings_path=join(conf['external_embeddings_dir'], dataset_name, f'{dataset_name}_embeddings.json'),
                            clusterer='dbscan') for dataset_name in datasets_names]

    val_block_dicts = []
    test_block_dicts = []
    
    val_dmatrices = []
    test_dmatrices = []

    for dataset, clusterer in zip(datasets, clusterers):

        _, val_block_dict, test_block_dict = dataset.split_blocks_helper(dataset.get_blocks())

        val_block_dicts.append(val_block_dict)
        test_block_dicts.append(test_block_dict)

        val_dmatrices.append(clusterer.get_dmatrix_dict(val_block_dict))
        test_dmatrices.append(clusterer.get_dmatrix_dict(test_block_dict))

    
    # Define objective to find optimal hyperparms of DBSCAN
    def objective(params):

        f1s = []
        weights = [len(clusterer.signatures.keys()) for clusterer in clusterers]

        for dataset, val_block_dict, val_block_to_dmatrix, clusterer in zip(datasets, val_block_dicts, val_dmatrices, clusterers):
            clusterer.clusterer.set_params(**params)
            sign_to_pred_clusters = clusterer.predict(val_block_dict, val_block_to_dmatrix)

            # Performing evaluation after parsing results from file
            dummy_clusterer = DummyClusterer(sign_to_pred_clusters)
            metrics, _ = cluster_eval(dataset, dummy_clusterer, split='val')
            f1s.append(metrics['B3 (P, R, F1)'][2])
            # hp module is minimizing so return minus f1 to get it maximized

        return -(np.asarray(f1s) * np.asarray(weights)).sum()/np.sum(weights)
    
    search_space = {'eps' : hp.uniform('eps', 0, 1), 'min_samples' : scope.int(hp.quniform('min_samples', 1, 5, q=1))}
    #search_space = {'eps' : hp.uniform('eps', 0.0001, 0.9)}

    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=100
    )
    print(f'Best parameters found after optimization : {best}')

    for test_block_dict, test_block_to_dmatrix, clusterer, dataset_name, dataset in zip(test_block_dicts, test_dmatrices, clusterers, datasets_names, datasets):
        
        print(dataset_name)
        clusterer.clusterer.set_params(**best)
        sign_to_pred_clusters = clusterer.predict(test_block_dict, test_block_to_dmatrix)
        
        with open(f'clustering_results/{dataset_name}.json', 'w') as f:
            json.dump(sign_to_pred_clusters, f)

        # Performing evaluation after parsing results from file
        dummy_clusterer = DummyClusterer(sign_to_pred_clusters)
        metrics, metrics_per_signature = cluster_eval(dataset, dummy_clusterer, split='test')
        print(metrics)

