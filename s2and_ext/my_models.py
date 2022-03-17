import json, torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from typing import Callable, Union
from numpy import ndarray
from tqdm import tqdm
from joblib import dump, load

from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import classification_report

from s2and_ext.my_utils import load_signatures, plot_loss, NumpyDataset
from s2and_ext.my_featurization import Featurizer

class MLP(nn.Module):
    """
    Implements multilayer prerceptron
    """

    def __init__(self, hidden_layers, n_features, n_classes=1):

        super(MLP, self).__init__()
        self.n_classes = n_classes
        
        layers = []
        layers_in = [n_features] + hidden_layers
        layers_out = hidden_layers + [n_classes]

        for idx in range(0, len(hidden_layers)+1):
            layers.append(nn.Linear(layers_in[idx], layers_out[idx]))
            if idx != len(hidden_layers):
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X).squeeze(dim=-1)

    def predict(self, X):
        # Used for compatibility issues with sklearn
        return torch.sigmoid(self.forward(torch.Tensor(X))).detach().numpy()

    def fit(self, train_loader, val_loader, epochs=50, lr=1e-03):

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        train_loss, val_loss = [], []
        for epoch in tqdm(range(1, epochs+1)):
            epoch_loss = 0
            for idx, batch in enumerate(train_loader):
                x_batch, y_batch = batch
                optimizer.zero_grad()
                out = self.forward(x_batch)
                loss = criterion(out, y_batch.float())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_loss.append(epoch_loss/idx)

            epoch_loss = 0
            with torch.inference_mode():
                for idx, batch in enumerate(val_loader):
                    x_batch, y_batch = batch
                    out = self.forward(x_batch)
                    loss = criterion(out, y_batch.float())
                    epoch_loss += loss.item()
            val_loss.append(epoch_loss/idx)

        return train_loss, val_loss

class EnsembleModel():
    """
    Class for ensemble model consiting of two stacked layers MLP + Log.Regr.
    """

    def __init__(self, hidden_layers, n_features, n_classes,
                 shallow_nn_features:list, ensemble_features:list):
        
        self.shallow_nn = MLP(hidden_layers, n_features, n_classes)
        self.ensemble_layer = LogisticRegression(random_state=0)

        self.shallow_nn_features = shallow_nn_features
        self.ensemble_features = ensemble_features

    def fit(self, X_train, y_train, X_val, y_val, epochs=30, lr=1e-03):
        """
        Fits the stucked model
        """
        X_train_mlp = X_train[:,self.shallow_nn_features]
        X_val_mlp = X_val[:,self.shallow_nn_features]

        train_loader = DataLoader(NumpyDataset(X_train_mlp, y_train), batch_size=64, shuffle=True)
        val_loader = DataLoader(NumpyDataset(X_val_mlp, y_val), batch_size=32, shuffle=False)

        # Train shallow NN

        train_loss, val_loss = self.shallow_nn.fit(train_loader, val_loader, epochs=30, lr=1e-03)
        plot_loss(train_loss, val_loss, 'results/NNlosses')
        print(classification_report(y_val, np.rint(self.shallow_nn.predict(X_val_mlp))))
        torch.save(self.shallow_nn, 'models/shallowNN.pt')


        X_train_ens = np.concatenate((X_train[:,self.ensemble_features], np.expand_dims(self.shallow_nn.predict(X_train_mlp), axis=1)), axis=1)
        X_val_ens = np.concatenate((X_val[:,self.ensemble_features], np.expand_dims(self.shallow_nn.predict(X_val_mlp), axis=1)), axis=1)

        # Train ensemble layer

        self.ensemble_layer.fit(X_train_ens, y_train)
        print(classification_report(y_val, np.rint(self.ensemble_layer.predict(X_val_ens))))
        dump(self.ensemble_layer, f'models/ensemble.joblib')
        dump(self, f'models/combined.joblib')

    def predict_distance(self, X):
        """
        Predicts probability of featurized pair to originate from different entity
        """
        if isinstance(X, list):
            X = np.asarray(X)
        np.nan_to_num(X, copy=False)
        distance = self.shallow_nn.predict(torch.Tensor(X[:,self.shallow_nn_features]))
        ens_features = np.concatenate((X[:,self.ensemble_features], np.expand_dims(distance, axis=1)), axis=1)
        return self.ensemble_layer.predict_proba(ens_features)[:,0]

class LightGBMWrapper():
    """
    Wrapper for lightgbm model, implementing predict_distance method
    necessary for Clusterer objects to work
    """

    def __init__(self, model):

        self.model = model

    def predict_distance(self, X):
        """
        Predicts probability of featurized pair to originate from different entity
        """
        if isinstance(X,list):
            X = np.asarray(X)
        return self.model.predict_proba(X)[:,0]

class TabNetWrapper():
    """
    Wrapper for tabnet model, implementing predict_distance method
    necessary for Clusterer objects to work
    """

    def __init__(self, model):
        self.model = model

    def predict_distance(self, X):
        """
        Predicts probability of featurized pair to originate from different entity
        """
        if isinstance(X, list):
            X = np.asarray(X)
        np.nan_to_num(X, copy=False)
        return self.model.predict_proba(X)[:,0]

class Clusterer():
    """
    Responsible for clustering the block_dict given as input
    to predict method. 
    """

    def __init__(self, 
                combined_classifier : str,
                dataset_name : str, 
                featurization_function : Callable, 
                default_embeddings: bool = True,
                embeddings_path: Union[str, None] = None,
                clusterer: str = 'dbscan') -> None:

        self.signatures = load_signatures(dataset_name)
        self.featurization_function = featurization_function
        self.default_embeddings = default_embeddings
        self.embeddings_path = embeddings_path
        if not default_embeddings:
            with open(embeddings_path) as f:
                self.paper_ids_to_emb = json.load(f)

        self.model = load(combined_classifier)
        self.clusterer_name = clusterer

        if clusterer ==  'dbscan':
            self.clusterer = DBSCAN(eps=0.4, min_samples=1, metric='precomputed', n_jobs=-1)
        elif clusterer == 'agglomerative':
            self.clusterer = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                             distance_threshold=0.4, linkage='average')

    @torch.inference_mode()
    def get_distance_matrix(self, block : list, signatures : dict) -> ndarray:

        n_signatures = len(block)
        d_matrix = np.zeros((n_signatures, n_signatures))

        features = []
        feature_idx = 0
        # Maps feature list potition to location in d_matrix
        # Used for placing results in correct dmatrix position
        mapper = {}
        for i in range(0, n_signatures):
            for j in range(0, n_signatures):
                if i > j:
                    continue
                # This is the same signature
                if i == j:
                    d_matrix[i,j] = 0
                # This is non the same signature
                else:
                    if block[i] in signatures and block[j] in signatures:
                        sig1 = signatures[block[i]]
                        sig2 = signatures[block[j]]
                        if not self.default_embeddings:
                            # Replace default embedding 
                            sig1['paperVector'] = self.paper_ids_to_emb[str(sig1['paper_id'])]
                            sig2['paperVector'] = self.paper_ids_to_emb[str(sig2['paper_id'])]
                        features.append(self.featurization_function(sig1, sig2))
                        mapper[feature_idx] = (i,j)
                        feature_idx += 1
                    else:
                        d_matrix[i,j] = 1
        
        # Having featurized every sign combination, we calculate
        # distances in a batch-way to save time
        if len(features) > 0:
            distances = self.model.predict_distance(features)
            for i in range(distances.shape[0]):
                d_matrix[mapper[i]] = distances[i].item()

        return d_matrix + np.transpose(d_matrix)

    def get_dmatrix_dict(self, block_dict):

        block_to_dmatrix = {}
        for block_name in tqdm(block_dict):
            block = block_dict[block_name]
            dmatrix = self.get_distance_matrix(block, self.signatures)
            block_to_dmatrix[block_name] = dmatrix
        return block_to_dmatrix

    def predict(self, block_dict, block_to_dmatrix=None):

        sign_to_pred_clusters = {}
        for block_name in block_dict:
            block = block_dict[block_name]
            if block_to_dmatrix is None:
                dmatrix = self.get_distance_matrix(block, self.signatures)
            else:
                dmatrix = block_to_dmatrix[block_name]
            # Resolve issue with dmatrix of 1 element in agglomerative clustering
            if self.clusterer_name == 'agglomerative' and dmatrix.shape == (1,1):
                clusters=[0]
            else:
                clusters = self.clusterer.fit_predict(dmatrix)

            max_identifier = max(clusters)
            for signature, cluster in zip(block, clusters):
                
                if cluster == -1:
                    max_identifier += 1
                    sign_to_pred_clusters[signature] = block_name + '_' + str(max_identifier)
                else:
                    sign_to_pred_clusters[signature] = block_name + '_' + str(cluster)  
        return sign_to_pred_clusters

class DummyClusterer():

    def __init__(self, source) -> None:
        
        if isinstance(source, str):
            with open(source) as f:
                self.sign_to_pred_cluster = json.load(f)
        else:
            self.sign_to_pred_cluster = source
            
    def predict(self, block_dict, dataset, use_s2_clusters) -> dict:
        
        # dataset and use_s2_clusters are placed only for compatibility reasons
        
        pred_clusters = {}
        uknown_id = 0
        for signatures in block_dict.values():
            for signature in signatures:
                if signature not in self.sign_to_pred_cluster:
                    pred_clusters[f'uknown_{uknown_id}'] = [signature]
                    uknown_id += 1
                else:
                    cluster = self.sign_to_pred_cluster[signature]
                    if cluster not in pred_clusters:
                        pred_clusters[cluster] = [signature]
                    else:
                        pred_clusters[cluster].append(signature)
        
        return pred_clusters, None 