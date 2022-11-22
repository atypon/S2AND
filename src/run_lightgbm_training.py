import joblib
import mlflow
import lightgbm as lgb
import numpy as np
from typing import Dict, Union
from os.path import join
from utils.configs import load_configurations
from utils.mlflow import get_or_create_experiment
from sklearn.metrics import classification_report, f1_score
from s2and_ext.featurization.utils import get_matrices, featurizing_function
from s2and_ext.classification_models import LightGBMWrapper


def run_lightgbm_experiment(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    results_folder: str,
    model_path: str,
    model_hyperparams: Dict[str, Union[str, float, int]]
) -> None:
    """
    Train, evaluate and then same lightgbm model
    """
    mlflow.lightgbm.autolog()
    model = lgb.LGBMClassifier(**model_hyperparams)
    model.fit(X_train, y_train)
    train_report = classification_report(y_train, model.predict(X_train))
    test_report = classification_report(y_test, model.predict(X_test))
    print('\nTrain set evaluation')
    print(train_report)
    print('\nTest set evaluation')
    print(test_report)
    with open(join(results_folder, 'train_report.txt'), 'w') as f:
        f.write(train_report)
    with open(join(results_folder, 'test_report.txt'), 'w') as f:
        f.write(test_report)
    mlflow.log_metric('f1-macro', f1_score(y_test, model.predict(X_test), average='macro'))
    mlflow.log_artifact(join(results_folder, 'train_report.txt'))
    mlflow.log_artifact(join(results_folder, 'test_report.txt'))
    # Save model as LightGB wrapper that implements predict_distance method
    model = LightGBMWrapper(model)
    joblib.dump(model, model_path)


if __name__ == "__main__":

    cfg = load_configurations('configs/classifier_conf.yml')
    X_train, y_train, X_val, y_val, X_test, y_test = get_matrices(
        datasets=cfg.data.datasets, 
        featurizing_function=featurizing_function, 
        remove_nan=False,
        default_embeddings=cfg.data.default_embeddings,
        external_emb_dir=cfg.data.external_embeddings_dir
    )
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    experiment_id = get_or_create_experiment(name=cfg.mlflow.experiment_name)
    with mlflow.start_run(experiment_id=experiment_id, tags={'datasets': cfg.data.datasets}, run_name=cfg.mlflow.run_name):
        run_lightgbm_experiment(
            X_train=X_train, 
            y_train=y_train, 
            X_test=X_val, 
            y_test=y_val, 
            results_folder=cfg.results.results_folder,
            model_path=cfg.results.model_path,
            model_hyperparams=cfg.model
        )
