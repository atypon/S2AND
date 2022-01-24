import os, pickle
import numpy as np

from joblib import dump, load

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from s2and_ext.my_utils import load_dataset, NumpyDataset, plot_loss
from s2and_ext.my_featurization import get_matrices, featurizing_function
from s2and_ext.my_models import EnsembleModel


datasets = ['aminer', 'pubmed', 'zbmath', 'kisti', 'arnetminer']

if os.path.isfile('cached/numpy_arrays.pickle'):
    with open('cached/numpy_arrays.pickle', 'rb') as f:
        X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(f)
else:
    X_train, y_train, X_val, y_val, X_test, y_test = get_matrices(datasets, featurizing_function)
    with open('cached/numpy_arrays.pickle', 'wb') as f:
        pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test), f)


model = EnsembleModel(hidden_layers=[4], n_features=6, n_classes=1,
                     shallow_nn_features = [0,1,2,3,4,5],
                     ensemble_features = [6,7,8])
model.fit(X_train, y_train, X_val, y_val, epochs=30, lr=1e-03)
