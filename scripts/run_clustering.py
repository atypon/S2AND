import mlflow
from os.path import join
from hyperopt import fmin, tpe, hp
from hyperopt.pyll import scope
from s2and.extentions.utils import load_dataset
from s2and.extentions.clustering_models import Clusterer
from s2and.extentions.featurization.utils import featurizing_function
from s2and.extentions.clustering_objective import Objective
from s2and.utils.configs import load_configurations
from s2and.utils.mlflow import get_or_create_experiment

if __name__ == "__main__":

    cfg = load_configurations('configs/clusterer_conf.yml')
    datasets = [load_dataset(dataset_name) for dataset_name in cfg.datasets]
    clusterers = [
        Clusterer(
            combined_classifier=cfg.model_source,
            dataset_name=dataset_name,
            featurization_function=featurizing_function,
            default_embeddings=cfg.default_embeddings,
            embeddings_path=join(
                cfg.external_embeddings_dir,
                dataset_name,
                f'{dataset_name}_embeddings.json'
            ),
            clusterer='dbscan'
        ) for dataset_name in cfg.datasets
    ]

    objective = Objective(dataset_names=cfg.datasets, datasets=datasets, clusterers=clusterers)
    search_space = {
        'eps': hp.uniform('eps', 0, 1),
        'min_samples': scope.int(hp.quniform('min_samples', 1, 5, q=1))
    }
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=100
    )
    print(f'Best parameters found after optimization : {best}')
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    experiment_id = get_or_create_experiment(name=cfg.mlflow.experiment_name)
    with mlflow.start_run(
        experiment_id=experiment_id,
        tags={'datasets': str(cfg.datasets)},
        run_name=cfg.mlflow.run_name
    ):
        mlflow.log_params(best)
        objective.evaluate_best(best_params=best)
