import json, torch, copy
import lightgbm as lgb

import numpy as np
import torch.nn as nn
import torch.optim as optim

from typing import Callable, Dict, List, Tuple, Union
from tqdm import tqdm
from joblib import dump, load

from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import classification_report

from s2and_ext.my_utils import load_signatures, plot_loss, NumpyDataset

from torch.utils.data import DataLoader
from pytorch_tabnet.tab_model import TabNetClassifier


class MLP(nn.Module):

    """
    Implements simple multilayer perceptron

    """

    def __init__(self,
                 hidden_layers : List[int],
                 n_features : List[int],
                 n_classes : int =1) -> None :
        
        """
        Constructs the MLP

        """

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

    def forward(self, X : torch.Tensor) -> torch.Tensor :
        
        """
        Model's forward pass
        """

        return self.model(X).squeeze(dim=-1)

    def predict(self, X : np.ndarray) -> np.ndarray:

        """
        Predict method accepts and returns numpy array and also applies
        activation function to output layer

        """

        # Used for compatibility issues with sklearn
        return torch.sigmoid(self.forward(torch.Tensor(X))).detach().numpy()

    def fit(self, 
            train_loader : DataLoader, 
            val_loader : DataLoader, 
            epochs : int = 50,
            lr : float =1e-03) -> Tuple[List[float]]:

        """
        Trains the model

        """

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
    Implements model stacking of MLP + logistic regression of features 
    regarding equality of ids that can be considered an another model

    """

    def __init__(self, 
                 hidden_layers : List[int], 
                 n_features : List[int],
                 n_classes : int,
                 shallow_nn_features : List[int],
                 ensemble_features : List[int]) -> None:
        
        """
        Initializes ensemble model

        """

        self.shallow_nn = MLP(hidden_layers, n_features, n_classes)
        self.ensemble_layer = LogisticRegression(random_state=0)

        self.shallow_nn_features = shallow_nn_features
        self.ensemble_features = ensemble_features

    def fit(self, 
            X_train : np.ndarray, 
            y_train : np.ndarray,
            X_val : np.ndarray, 
            y_val : np.ndarray, 
            epochs : int = 30,
            lr : float = 1e-03) -> None:

        """
        Fits both the MLP and then the logistic regression layer

        """

        X_train_mlp = X_train[:,self.shallow_nn_features]
        X_val_mlp = X_val[:,self.shallow_nn_features]

        train_loader = DataLoader(NumpyDataset(X_train_mlp, y_train), batch_size=64, shuffle=True)
        val_loader = DataLoader(NumpyDataset(X_val_mlp, y_val), batch_size=32, shuffle=False)

        # Train shallow NN

        train_loss, val_loss = self.shallow_nn.fit(train_loader, val_loader, epochs=epochs, lr=lr)
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

    def predict_distance(self, 
                         X : Union[np.ndarray, List[List[float]]]) -> np.ndarray :

        """
        Predicts distance (probability(0))
        
        """

        if isinstance(X, list):
            X = np.asarray(X)
        np.nan_to_num(X, copy=False)
        distance = self.shallow_nn.predict(torch.Tensor(X[:,self.shallow_nn_features]))
        ens_features = np.concatenate((X[:,self.ensemble_features], np.expand_dims(distance, axis=1)), axis=1)
        return self.ensemble_layer.predict_proba(ens_features)[:,0]


class LightGBMWrapper():

    """
    Wrapper for LGBMClassifier that implements predict_distance method

    """

    def __init__(self, model : lgb.LGBMClassifier) -> None:
        
        """
        Initilizes wrapper

        """

        self.model = model

    def predict_distance(self, X : Union[np.ndarray, List[List[float]]]) -> np.ndarray:

        """
        Predicts distance (probability(0))

        """

        if isinstance(X,list):
            X = np.asarray(X)
        return self.model.predict_proba(X)[:,0]


class TabNetWrapper():

    """
    Wrapper for TabNet that implements predict_distance method

    """

    def __init__(self, model : TabNetClassifier) -> None:

        """
        Initilize wrapper
        """

        self.model = model

    def predict_distance(self, X : Union[np.ndarray, List[List[float]]]) -> np.ndarray:

        """
        Predicts distance (probability(0))

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
                 featurization_fun : Callable, 
                 clusterer : str = 'dbscan') -> None:

        """
        Initilizes clusterer object

        """
        self.signatures = load_signatures(dataset_name)
        self.featurization_fun = featurization_fun
        self.model = load(combined_classifier)
        self.clusterer_name = clusterer

        if clusterer ==  'dbscan':
            self.clusterer = DBSCAN(eps=0.4, min_samples=1, metric='precomputed', n_jobs=-1)
        elif clusterer == 'agglomerative':
            self.clusterer = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                             distance_threshold=0.4, linkage='average')

    def get_distance_matrix(self, 
                            block : List[str], 
                            signatures : Dict[str, Dict[str, Union[list, float, str]]]) -> np.ndarray:

        """
        Given a list of signatures (block) and the signatures metadata in 
        signatures dict, calculates the block distance matrix

        """

        n_signatures = len(block)
        d_matrix = np.zeros((n_signatures, n_signatures))

        features = []
        feature_idx = 0
        # Maps feature list potition to location in d_matrix
        # Used for placing results in correct dmatrix position
        mapper = {}
        with torch.inference_mode():
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
                            features.append(self.featurization_fun(sig1, sig2))
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

    def get_dmatrix_dict(self, block_dict : Dict[str, List[str]]) -> Dict[str, np.ndarray]:

        """
        Given dictionary of blocks (block_name : block) returns dict of distance matrices
        {block_name : distance_matrix}

        """

        block_to_dmatrix = {}
        for block_name in tqdm(block_dict):
            block = block_dict[block_name]
            dmatrix = self.get_distance_matrix(block, self.signatures)
            block_to_dmatrix[block_name] = dmatrix
        return block_to_dmatrix

    def predict(self, 
                block_dict : Dict[str, List[str]], 
                block_to_dmatrix : Union[None, Dict[str, np.ndarray]] = None) -> Dict[str,str]:

        """
        Runs clustering on distance matrices and produces final prediction

        """

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

    """
    Used for compatibility reasons for S2AND's cluster_eval function

    """

    def __init__(self, source : Union[str, Dict[str, str]]) -> None:
            
        if isinstance(source, str):
            with open(source) as f:
                self.sign_to_pred_cluster = json.load(f)
        else:
            self.sign_to_pred_cluster = source
            
    def predict(self, 
                block_dict : Dict[str, List[str]], 
                dataset, 
                use_s2_clusters) -> Dict[str, List[str]]:
        
        """
        Given the block dict, returns the predicted clusters dict
        cluster_id : signatures llist
        
        """

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