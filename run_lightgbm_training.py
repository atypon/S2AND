import yaml, argparse, joblib, lightgbm as lgb

from sklearn.metrics import classification_report
from s2and_ext.my_featurization import get_matrices, featurizing_function
from s2and_ext.my_models import LightGBMWrapper


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
        
    model = lgb.LGBMClassifier(objective='binary',
                            tree_learner='data',
                            verbosity=-1,
                            n_jobs=15,
                            metric='auc',
                            random_state=42)

    model.fit(X_train, y_train)
   
    print('\nTrain set evaluation')
    print(classification_report(y_train, model.predict(X_train)))
    print('\nTest set evaluation')
    print(classification_report(y_val, model.predict(X_val)))

    model = LightGBMWrapper(model)
    joblib.dump(model, conf['model_path'])
