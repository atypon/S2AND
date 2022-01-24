import os, pickle, joblib
import numpy as np

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

from pytorch_tabnet.tab_model import TabNetClassifier
import torch 
from pytorch_tabnet.pretraining import TabNetPretrainer

np.nan_to_num(X_train, copy=False)
np.nan_to_num(X_val, copy=False)

# TabNetPretrainer
unsupervised_model = TabNetPretrainer(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type='entmax' # "sparsemax"
)

unsupervised_model.fit(
    X_train=X_train,
    eval_set=[X_val],
    pretraining_ratio=0.8,
)


clf = TabNetClassifier(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":10, # how to use learning rate scheduler
                      "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='sparsemax' # This will be overwritten if using pretrain model
)

clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_name=['train', 'valid'],
    eval_metric=['auc'],
    from_unsupervised=unsupervised_model)

print(classification_report(y_train, clf.predict(X_train)))
print(classification_report(y_val, clf.predict(X_val)))

from s2and_ext.my_models import TabNetWrapper

clf = TabNetWrapper(clf)
joblib.dump(clf, 'models/tabnet.joblib')
