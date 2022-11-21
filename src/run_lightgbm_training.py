from pydoc import cli
import yaml, argparse, joblib, mlflow, lightgbm as lgb

from mlflow.tracking import MlflowClient
from sklearn.metrics import classification_report, f1_score
from s2and_ext.my_featurization import get_matrices, featurizing_function
from s2and_ext.my_models import LightGBMWrapper

def run_lightgbm_experiment(X_train, y_train, X_test, y_test):
    """Run training evaluation and save lightgbm model"""
    mlflow.lightgbm.autolog()
    model = lgb.LGBMClassifier(objective='binary',
                                tree_learner='data',
                                verbosity=-1,
                                n_jobs=15,
                                metric='auc',
                                random_state=42)

    model.fit(X_train, y_train)

    train_report = classification_report(y_train, model.predict(X_train))
    test_report = classification_report(y_test, model.predict(X_test))
    #print('\nTrain set evaluation')
    #print(train_report)
    #print('\nTest set evaluation')
    #print(test_report)

    with open('results/train_report.txt', 'w') as f:
        f.write(train_report)
    with open('results/test_report.txt', 'w') as f:
        f.write(test_report)

    mlflow.log_metric('f1-macro', f1_score(y_test, model.predict(X_test), average='macro'))
    mlflow.log_artifact('results/train_report.txt')
    mlflow.log_artifact('results/test_report.txt')

    model = LightGBMWrapper(model)
    joblib.dump(model, conf['model_path'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--conf',
        default='configs/classifier_conf.yml',
        help='Path to configuration file'
    )
    args = parser.parse_args()
    with open(args.conf) as f:
        conf = yaml.safe_load(f)

    X_train, y_train, X_val, y_val, X_test, y_test = get_matrices(datasets=conf['datasets'], 
                                                                  featurizing_function=featurizing_function, 
                                                                  remove_nan=False,
                                                                  default_embeddings=conf['default_embeddings'],
                                                                  external_emb_dir=conf['external_embeddings_dir'])
    mlflow.set_tracking_uri(conf['mlflow']['tracking_uri'])
    client = MlflowClient()
    

    experiment = client.get_experiment_by_name(conf['mlflow']['experiment_name'])
    if experiment is None:
        experiment_id = client.create_experiment(conf['mlflow']['experiment_name'])
    else:
        if dict(experiment)['lifecycle_stage'] == 'deleted':
            client.restore_experiment(dict(experiment)['experiment_id'])
        experiment_id = dict(experiment)['experiment_id']

    with mlflow.start_run(experiment_id=experiment_id, tags={'datasets': conf['datasets']}, run_name=conf['mlflow']['run_name']):
        run_lightgbm_experiment(X_train, y_train, X_val, y_val)
