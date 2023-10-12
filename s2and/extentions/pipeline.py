from os.path import join
from typing import Any, List, Dict

from hyperopt import fmin, tpe
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import classification_report, f1_score

from s2and import logger
from s2and.eval import pairwise_eval
from s2and.extentions.classification_models import LightGBMWrapper
from s2and.extentions.clustering_models import Clusterer
from s2and.extentions.clustering_objective import Objective
from s2and.extentions.featurization.utils import get_matrices
from s2and.extentions.utils import load_dataset
from s2and.utils.testing import create_test_file
from s2and.utils.onnx_converter import ONNXConverter


class ANDPipeline():
    """
    Class implementing the complete AND pipeline
    """

    def __init__(
        self,
        features: List[Dict[str, Any]],
        external_embeddings_dir: str,
        pairwise_model_path: str,
        results_dir: str
    ) -> None:
        """
        Initialize AND pipeline
        :param features: list of features and operations to be used
        :param external_embeddings_dir: path to external embeddings
        :param pairwise_model_path: path to save pairwise model after training
                                    or to use for clustering step
        :param results_dir: directory to save results
        """
        self.features = features
        self.external_embeddings_dir = external_embeddings_dir
        self.pairwise_model_path = pairwise_model_path
        self.categorical_features = [
            idx for idx, feature in enumerate(features) if feature['categorical']
        ]
        self.feature_names = [f'{feature["operation"]}({feature["field"]})' for feature in features]
        self.results_dir = results_dir

    def train_pairwise_classifier(
        self,
        datasets: List[str],
        model_hyperparams: Dict[str, Any],
        onnx_path: str,
        unit_test_pairs: int,
        unit_test_dataset: str,
        test_file_path: str,
    ):
        """
        Perform complete training of the pairwise classifier model
        :param datasets: list of S2AND datasets to train the model on
        :param model_hyperaparms: hyperparams of pairwise classifier
        :param onnx_path: path to save converted model to onnx
        :param unit_test_pairs: number of signature pairs for test file
        :param unit_test_dataset: dataset to draw pairs from for test file
        :param test_file_path: path to save test file
        """
        logger.info('Starting pairwise-classifier training...')
        X_train, y_train, _, _, X_test, y_test = get_matrices(
            datasets=datasets,
            features=self.features,
            remove_nan=False,
            external_emb_dir=self.external_embeddings_dir
        )

        # Fit the classifier
        mlflow.lightgbm.autolog()
        model = lgb.LGBMClassifier(**model_hyperparams)
        model.fit(
            X=X_train,
            y=y_train,
            categorical_feature=self.categorical_features
        )

        # Save results and reports
        train_report = classification_report(y_train, model.predict(X_train))
        test_report = classification_report(y_test, model.predict(X_test))
        logger.info(f'\nTrain set evaluation\n{train_report}')
        logger.info(f'\nTest set evaluation\n{test_report}')
        with open(join(self.results_dir, 'train_report.txt'), 'w') as f:
            f.write(train_report)
        with open(join(self.results_dir, 'test_report.txt'), 'w') as f:
            f.write(test_report)
        mlflow.log_param('classifier-datasets', datasets)
        mlflow.log_metric('f1-macro', f1_score(y_test, model.predict(X_test), average='macro'))
        mlflow.log_artifact(join(self.results_dir, 'train_report.txt'))
        mlflow.log_artifact(join(self.results_dir, 'test_report.txt'))
        pairwise_eval(
            X=X_test,
            y=y_test,
            classifier=model,
            figs_path=self.results_dir,
            title='classifier',
            shap_feature_names=self.feature_names
        )
        mlflow.log_artifact(join(self.results_dir, 'classifier_pr.png'))
        mlflow.log_artifact(join(self.results_dir, 'classifier_roc.png'))
        mlflow.log_artifact(join(self.results_dir, 'classifier_shap.png'))
        mlflow.log_param('features', self.feature_names)
        
        # Plot the first 5 trees of lightgbm
        for tree_index in range(min(5, len(model.booster_.dump_model()['tree_info']))):
            plt.figure()
            lgb.plot_tree(model, tree_index=tree_index, show_info=['data_percentage'])
            plt.savefig(join(self.results_dir, f'model-tree-{tree_index}.png'), dpi=750)
            mlflow.log_artifact(join(self.results_dir, f'model-tree-{tree_index}.png'))

        # Save model as LightGBM wrapper that implements predict_distance method
        model = LightGBMWrapper(model)
        joblib.dump(model, self.pairwise_model_path)
        mlflow.log_artifact(self.pairwise_model_path)

        # Convert model to ONNX format
        logger.info('Converting to ONNX...')
        converter = ONNXConverter(
            model=model.model,
            features=self.features
        )
        converter.convert(destination_path=onnx_path)
        converter.test_conversion(
            destination_path=onnx_path,
            dataset=X_train
        )
        mlflow.log_artifact(onnx_path)

        # Create test file for the trained model
        logger.info('Creating test file...')
        create_test_file(
            dataset_name=unit_test_dataset,
            features=self.features,
            model=model,
            embeddings_dir=self.external_embeddings_dir,
            num_pairs=unit_test_pairs,
            test_file_path=test_file_path
        )
        mlflow.log_artifact(test_file_path)

    def optimize_clusterer(
        self,
        datasets: List[str],
        clusterer: str
    ) -> None:
        """
        Optimize and evaluate clustering step of the and procedure
        :param datasets: list of S2AND datasets to optimize the clusterer on
        :param clusterer: clusterer to be used either dbscan or agglomerative
        """
        logger.info('Starting clustering optimization...')
        loaded_datasets = [load_dataset(dataset_name) for dataset_name in datasets]
        clusterers = [
            Clusterer(
                combined_classifier=self.pairwise_model_path,
                dataset_name=dataset_name,
                features=self.features,
                embeddings_dir=self.external_embeddings_dir,
                clusterer=clusterer
            ) for dataset_name in datasets
        ]
        
        objective = Objective(dataset_names=datasets, datasets=loaded_datasets, clusterers=clusterers)
        search_space = objective.get_search_space(clusterer)
        # Run minimization with hyperopt to received best set of hyperparams
        best = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=100
        )
        
        # Fix linkage function as best returns index of bext linkage function, not the name
        if clusterer == 'agglomerative':
            best['linkage'] = ['complete', 'average', 'single'][best['linkage']]
        logger.info(f'\nBest parameters found after optimization : {best}')
        
        # Evaluate with optimal hyperparams and log results
        metrics = objective.evaluate_best(best_params=best)
        mlflow.log_param('clusterer', clusterer)
        mlflow.log_param('features', self.feature_names)
        mlflow.log_param('clusterer-datasets', datasets)
        mlflow.log_params(best)
        for dataset, results in metrics.items():
            logger.info(f'\n{dataset}')
            logger.info(f'\n{metrics}')
            mlflow.log_metric(f'{dataset} B3 P', results['B3 (P, R, F1)'][0])
            mlflow.log_metric(f'{dataset} B3 R', results['B3 (P, R, F1)'][1])
            mlflow.log_metric(f'{dataset} B3 F1', results['B3 (P, R, F1)'][2])
