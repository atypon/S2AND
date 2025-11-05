import argparse

import mlflow
from mlflow.tracking.request_header.registry import _request_header_provider_registry

from s2and.extentions.pipeline import ANDPipeline
from s2and.utils.configs import load_configurations
from s2and.utils.mlflow import CustomHeaderProvider, get_or_create_experiment

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-config-file',
        help='path to configuration file',
        default='configs/and_configuration.yaml'
    )
    parser.add_argument(
        '--classification-step',
        help='run classification step',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--clustering-step',
        help='run clustering step',
        action='store_true',
        default=False
    )
    args = parser.parse_args()

    cfg = load_configurations(path=args.config_file)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    _request_header_provider_registry.register(CustomHeaderProvider)
    experiment_id = get_or_create_experiment(name=cfg.mlflow.experiment_name)
    and_pipeline = ANDPipeline(
        features=cfg.features,
        external_embeddings_dir=cfg.data.external_embeddings_dir,
        pairwise_model_path=cfg.pairwise_model.path,
        results_dir=cfg.results_dir
    )
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=cfg.mlflow.run_name
    ):
        if args.classification_step:
            and_pipeline.train_pairwise_classifier(
                datasets=cfg.pairwise_model.datasets,
                model_hyperparams=cfg.pairwise_model.hyperparams,
                onnx_path=cfg.pairwise_model.onnx_path,
                unit_test_pairs=cfg.pairwise_model.unit_test.num_pairs,
                unit_test_dataset=cfg.pairwise_model.unit_test.dataset,
                test_file_path=cfg.pairwise_model.unit_test.test_file_path,
            )
        if args.clustering_step:
            and_pipeline.optimize_clusterer(
                datasets=cfg.clustering.datasets,
                clusterer=cfg.clustering.algorithm
            )
