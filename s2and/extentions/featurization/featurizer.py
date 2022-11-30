import json 
import numpy as np
from typing import Callable, Dict, Tuple, Union
from s2and.extentions.utils import load_dataset, load_signatures

class Featurizer():
    '''
    Class for creating numpy arrays containg the features for each signature pair
    '''

    def __init__(
        self, 
        dataset_name: str, 
        featurizing_function: Callable, 
        default_embeddings: bool = True,
        embeddings_path: Union[None, str] = None
    ) -> None:
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

    def featurize_pairs(self, pairs: Tuple[Dict[str, dict]]) -> Tuple[np.ndarray]:
        '''
        Given the list of pairs return the matrix of features and labels
        '''  
        X = []
        y = []
        for pair in pairs:
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

    def get_feature_matrix(self, split: str) -> Tuple[np.ndarray]:
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