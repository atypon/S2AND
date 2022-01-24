import os, pickle, joblib
import lightgbm as lgb

from sklearn.metrics import classification_report
from s2and_ext.my_featurization import get_matrices, featurizing_function


datasets = ['aminer', 'pubmed', 'zbmath', 'kisti', 'arnetminer']

if os.path.isfile('cached/numpy_arrays.pickle'):
    with open('cached/numpy_arrays.pickle', 'rb') as f:
        X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(f)
else:
    X_train, y_train, X_val, y_val, X_test, y_test = get_matrices(datasets, featurizing_function, remove_nan=False)
    with open('cached/numpy_arrays.pickle', 'wb') as f:
        pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test), f)

from s2and_ext.my_models import LightGBMWrapper

model = lgb.LGBMClassifier(objective='binary',
                           tree_learner='data',
                           verbosity=-1,
                           n_jobs=15,
                           metric='auc',
                           random_state=42)


model.fit(X_train, y_train)

print(classification_report(y_train, model.predict(X_train)))
print(classification_report(y_val, model.predict(X_val)))

model = LightGBMWrapper(model)
joblib.dump(model, 'models/lightgbm.joblib')
