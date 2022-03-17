import json, torch, numpy as np
from typing import Callable, Dict, List, Tuple, Union

from Levenshtein import distance
from s2and_ext.my_utils import load_dataset, load_signatures
from torch.nn import functional as F


def extract_feature(sig1 : Dict[str, dict], 
                    sig2 : Dict[str, dict],
                    attribute : str) -> float :
    """
    Returns 1 if attribute is equal in sig1 and sig2 otherwise 0.
    If attribute is missing, returns np.nan
    """
    if (sig1.get(attribute, None) is not None) and (sig2.get(attribute, None) is not None):
        if sig1[attribute] == sig2[attribute]:
            return 1
        return 0
    return np.nan

def cosine_sim(sig1 : Dict[str, dict], 
               sig2 : Dict[str, dict]) -> float :
    """
    Computes cosine similarity of paperVector field of signature.
    If field is missing, returns np.nan
    """
    v1 = sig1.get('paperVector', None)
    v2 = sig2.get('paperVector', None)
    if (v1 is not None) and (v2 is not None):
        return F.cosine_similarity(torch.Tensor(v1), torch.Tensor(v2), dim=0).item()
    return np.nan

def name_distance(sig1 : Dict[str, dict],
                  sig2 : Dict[str, dict]) -> float :
    """
    Computes levenshtein distance of s2AuthorName attribute.
    If attribute is missing, return np.nan
    """
    v1 = sig1.get('s2AuthorName', None)
    v2 = sig2.get('s2AuthorName', None)
    if (v1 is not None) and (v2 is not None):
        return distance(v1,v2)
    return np.nan

def jaccard(sig1 : Dict[str, dict], 
            sig2 : Dict[str, dict],
            attribute : str) -> float :
    """
    Computes jaccard similarity of list attribute between two signatures.
    If attribute is missing, returns np.nan
    """
    v1 = sig1.get(attribute, None)
    v2 = sig2.get(attribute, None)
    if (v1 is not None) and (v2 is not None):
        v1 = set(v1)
        v2 = set(v2)
        if len(v1.union(v2)) > 0:
            return len(v1.intersection(v2)) / len(v1.union(v2))
        return 0
    return np.nan

def featurizing_function(sig1 : Dict[str, dict],
                         sig2 : Dict[str, dict]) -> float:
    """
    Calculates the feature vector of two given signature dicts
    """
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

    def __init__(self, 
                 dataset_name : str, 
                 featurizing_function : Callable, 
                 default_embeddings : bool = True,
                 embeddings_path : Union[None, str] = None) -> None :
        """
        Initializes Featurizer object, loading datasets and signatures.
        If parse_specter = true, it parses the provided specter embeddings
        for them to be used in the featurization process.
        """
        self.default_embeddings = default_embeddings
        self.dataset = load_dataset(dataset_name)
        self.extended_signatures = load_signatures(dataset_name)
        self.featurizing_function = featurizing_function

        if not default_embeddings:
            with open(embeddings_path) as f:
                self.paper_ids_to_emb = json.load(f)

    def featurize_pairs(self, pairs : Tuple[Dict[str, dict]]) -> Tuple[np.ndarray]:
        '''
        Given the list of pairs return the matrix of features and labels
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
                if not self.default_embeddings:
                    sig1['paperVector'] = self.paper_ids_to_emb[str(sig1['paper_id'])]
                    sig2['paperVector'] = self.paper_ids_to_emb[str(sig2['paper_id'])]
                X.append(self.featurizing_function(sig1, sig2))

        return np.asarray(X), np.asarray(y)

    def get_feature_matrix(self, split : str) -> Tuple[np.ndarray]:
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

def get_matrices(datasets : List[str], 
                 featurizing_function : Callable, 
                 remove_nan : bool =True,
                 default_embeddings: bool = True,
                 external_emb_dir: str = None) -> Tuple[np.ndarray]:
    '''
    Featurize multiple datasets and return the combined matrix
    If no featurizing_function is given, the default will be used
    '''
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    for dataset_name in datasets:

        embeddings_path = f'{external_emb_dir}/{dataset_name}/{dataset_name}_embeddings.json'
        featurizer = Featurizer(dataset_name=dataset_name, 
                                featurizing_function=featurizing_function,
                                default_embeddings=default_embeddings,
                                embeddings_path=embeddings_path)
        
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
