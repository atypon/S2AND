import joblib
import mlflow
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Union, List
from os.path import join
from s2and.utils.configs import load_configurations
from s2and.utils.mlflow import get_or_create_experiment
from sklearn.metrics import classification_report, f1_score
from s2and.extentions.featurization.utils import get_matrices
from s2and.extentions.classification_models import LightGBMWrapper
from s2and.eval import pairwise_eval
from s2and import logger


def run_lightgbm_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    features: List[Dict[str, str]],
    results_folder: str,
    model_path: str,
    model_hyperparams: Dict[str, Union[str, float, int]],
    run_name: str
) -> None:
    """
    Train, evaluate and then same lightgbm model
    """
    mlflow.lightgbm.autolog()
    model = lgb.LGBMClassifier(**model_hyperparams)
    model.fit(X_train, y_train)
    train_report = classification_report(y_train, model.predict(X_train))
    test_report = classification_report(y_test, model.predict(X_test))
    logger.info(f'\nTrain set evaluation\n{train_report}')
    logger.info(f'\nTest set evaluation\n{test_report}')
    with open(join(results_folder, 'train_report.txt'), 'w') as f:
        f.write(train_report)
    with open(join(results_folder, 'test_report.txt'), 'w') as f:
        f.write(test_report)
    mlflow.log_metric('f1-macro', f1_score(y_test, model.predict(X_test), average='macro'))
    mlflow.log_artifact(join(results_folder, 'train_report.txt'))
    mlflow.log_artifact(join(results_folder, 'test_report.txt'))
    # Save model as LightGB wrapper that implements predict_distance method
    model = LightGBMWrapper(model)
    feature_names = [f'{feature["operation"]}({feature["field"]})' for feature in features]
    joblib.dump(model, model_path)
    pairwise_eval(
        X_test,
        y_test,
        model.model,
        figs_path=results_folder,
        title=run_name,
        shap_feature_names=feature_names
    )
    mlflow.log_artifact(join(results_folder, f'{run_name}_pr.png'))
    mlflow.log_artifact(join(results_folder, f'{run_name}_roc.png'))
    mlflow.log_artifact(join(results_folder, f'{run_name}_shap.png'))
    mlflow.log_param('features', feature_names)
    plt.figure()
    lgb.plot_tree(model.model)
    plt.savefig(join(results_folder, 'model_tree.png'), dpi=1000)
    mlflow.log_artifact(join(results_folder, 'model_tree.png'))


if __name__ == "__main__":

    cfg = load_configurations('configs/classifier_conf.yml')
    X_train, y_train, X_val, y_val, X_test, y_test = get_matrices(
        datasets=cfg.data.datasets,
        features=cfg.features,
        remove_nan=False,
        external_emb_dir=cfg.data.external_embeddings_dir
    )
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    experiment_id = get_or_create_experiment(name=cfg.mlflow.experiment_name)
    with mlflow.start_run(
        experiment_id=experiment_id,
        tags={'datasets': str(cfg.data.datasets)},
        run_name=cfg.mlflow.run_name
    ):
        run_lightgbm_experiment(
            X_train=X_train,
            y_train=y_train,
            X_test=X_val,
            y_test=y_val,
            features=cfg.features,
            results_folder=cfg.results.results_folder,
            model_path=cfg.results.model_path,
            model_hyperparams=cfg.model,
            run_name=cfg.mlflow.run_name
        )
