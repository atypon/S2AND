import numpy as np

from Levenshtein import distance
from scipy import spatial
from s2and_ext.my_utils import load_dataset, load_signatures
from torch.nn import functional as F
import torch, pickle


def extract_feature(sig1, sig2, attribute):
    if (sig1.get(attribute, None) is not None) and (sig2.get(attribute, None) is not None):
        if sig1[attribute] == sig2[attribute]:
            return 1
        return 0
    return np.nan

def cosine_sim(sig1, sig2):
    v1 = sig1.get('paperVector', None)
    v2 = sig2.get('paperVector', None)
    if (v1 is not None) and (v2 is not None):
        #return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        #return 1-spatial.distance.cosine(v1,v2)
        return F.cosine_similarity(torch.Tensor(v1), torch.Tensor(v2), dim=0).item()
    return np.nan

def name_distance(sig1, sig2):
    v1 = sig1.get('s2AuthorName', None)
    v2 = sig2.get('s2AuthorName', None)
    if (v1 is not None) and (v2 is not None):
        return distance(v1,v2)
    return np.nan

def jaccard(sig1, sig2, attribute):
    v1 = sig1.get(attribute, None)
    v2 = sig2.get(attribute, None)
    if (v1 is not None) and (v2 is not None):
        v1 = set(v1)
        v2 = set(v2)
        if len(v1.union(v2)) > 0:
            return len(v1.intersection(v2)) / len(v1.union(v2))
        return 0
    return np.nan

def featurizing_function(sig1, sig2):

    features = []
    features.append(cosine_sim(sig1, sig2))
    features.append(name_distance(sig1, sig2))
    features.append(jaccard(sig1, sig2, 'affiliationIds'))
    features.append(jaccard(sig1, sig2, 'fosIdsLevel0'))
    features.append(jaccard(sig1, sig2, 'fosIdsLevel1'))
    features.append(jaccard(sig1, sig2, 'authorNormNames'))

    # Ids for later use from the ensemble
    features.append(extract_feature(sig1, sig2, 'magAuthorId'))
    features.append(extract_feature(sig1, sig2, 's2AuthorId'))
    features.append(extract_feature(sig1, sig2, 'orcId'))
    return features


class Featurizer():
    
    '''
    Class for creating numpy arrays containg the features for each signature pair

    '''

    def __init__(self, dataset_name, featurizing_function, parse_specter=False):
        self.parse_specter = parse_specter
        self.dataset = load_dataset(dataset_name)
        self.extended_signatures = load_signatures(dataset_name)
        self.featurizing_function = featurizing_function

        if parse_specter:
            with open(f'data/{dataset_name}/{dataset_name}_specter.pickle', 'rb') as f:
                vectors, ids = pickle.load(f)
            
            self.paper_ids_to_specter = {}
            for vector, paper_id in zip(vectors, ids):
                self.paper_ids_to_specter[paper_id] = vector.tolist()

    def featurize_pairs(self, pairs):
        
        '''
        Given the list of pairs return the matrix of features
        '''  
        
        X = []
        y = []

        for i, pair in enumerate(pairs):
            # Make sure the extended dataset contains the signatures
            #  of the pair, else do not featurize the pair
            if pair[0] in self.extended_signatures and pair[1] in self.extended_signatures:
                y.append(pair[2])
                sig1 = self.extended_signatures[pair[0]]
                sig2 = self.extended_signatures[pair[1]]
                if self.parse_specter:
                    sig1['paperVector'] = self.paper_ids_to_specter[str(sig1['paper_id'])]
                    sig2['paperVector'] = self.paper_ids_to_specter[str(sig2['paper_id'])]
                X.append(self.featurizing_function(sig1, sig2))

        return np.asarray(X), np.asarray(y)

    def get_feature_matrix(self, split):

        '''
        Get dataset's featurized pairs matrix by specifying the desired split
        '''
        
        train_sig, val_sig, test_sig = self.dataset.split_cluster_signatures()
        train_pairs, val_pairs, test_pairs = self.dataset.split_pairs(train_sig, val_sig, test_sig)
        
        if split == 'train':
            return self.featurize_pairs(train_pairs)
        elif split == 'val':
            return self.featurize_pairs(val_pairs)
        elif split == 'test':
            return self.featurize_pairs(test_pairs)

def get_matrices(datasets, featurizing_function, remove_nan=True):

    '''
    Featurize multiple datasets and return the combined matrix
    If no featurizing_function is given, the default will be used
    '''

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    for dataset_name in datasets:

        featurizer = Featurizer(dataset_name, featurizing_function)
        
        X, y = featurizer.get_feature_matrix('train')
        X_train.append(X)
        y_train.append(y)

        X, y = featurizer.get_feature_matrix('val')
        X_val.append(X)
        y_val.append(y)
        
        X, y = featurizer.get_feature_matrix('test')
        X_test.append(X)
        y_test.append(y)
        print(f'Processed {dataset_name}')

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)

    X_val = np.vstack(X_val)
    y_val = np.concatenate(y_val)

    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)


    # Count nan per column for train set
    print('Nan values for each feature :')
    print(np.count_nonzero(np.isnan(X_train), axis=0))

    # Remove nan values
    if remove_nan:
        np.nan_to_num(X_train, copy=False)
        np.nan_to_num(X_test, copy=False)
        np.nan_to_num(X_val, copy=False)

    return X_train, y_train, X_val, y_val, X_test, y_test