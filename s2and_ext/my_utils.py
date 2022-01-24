import json, os, pickle, torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from typing import List, Dict
from os.path import join
from s2and.data import ANDData


def load_dataset(dataset_name : str):

    parent_dir = f"data/{dataset_name}/"
    if not os.path.isfile(f'pickled_datasets/{dataset_name}.pickle'):
        dataset = ANDData(
            signatures=join(parent_dir, f"{dataset_name}_signatures.json"),
            papers=join(parent_dir, f"{dataset_name}_papers.json"),
            mode="train",
            specter_embeddings=join(parent_dir, f"{dataset_name}_specter.pickle"),
            clusters=join(parent_dir, f"{dataset_name}_clusters.json"),
            block_type="s2",
            train_pairs_size=100000,
            val_pairs_size=10000,
            test_pairs_size=10000,
            name=dataset_name,
            n_jobs=4,
            preprocess=False
        )
        with open(f'pickled_datasets/{dataset_name}.pickle', 'wb') as f:
            pickle.dump(dataset, f)
    else:
        with open(f'pickled_datasets/{dataset_name}.pickle', 'rb') as f:
            dataset = pickle.load(f)
            print('Loaded dataset from pickle...')
    return dataset

def load_signatures(dataset : str) -> dict:

    with open(f'extended_data/{dataset}/{dataset+"-signatures.json"}') as f:
        content = f.read().split('\n')
        content.remove('')
        content = list(map(json.loads, content))

    return {entry['signature_id'] : entry for entry in content}

def get_block_dict(signatures : dict) -> dict:

    block_dict = {}
    for item in signatures.values():
        if item['block'] not in block_dict:
            block_dict[item['block']] = [item['signature_id']]
        else:
            block_dict[item['block']].append(item['signature_id'])
    return block_dict

def custom_distance(sig1, sig2):
    
    score = 0
    sum = 0 

    if (sig1.get('magAuthorId', None) is not None) and (sig2.get('magAuthorId', None) is not None):
        sum += 1
        if sig1['magAuthorId'] == sig2['magAuthorId']:
            score += 1
    if (sig1.get('s2AuthorId', None) is not None) and (sig2.get('s2AuthorId', None) is not None):
        sum += 1
        if sig1['s2AuthorId'] == sig2['magAuthorId']:
            score += 1
    if (sig1.get('s2AuthorShorNormName', None) is not None) and (sig2.get('s2AuthorShorNormName', None) is not None):
        sum += 0.5
        if sig1['s2AuthorShorNormName'] == sig2['s2AuthorShorNormName']:
            score += 0.5
    if (sig1.get('s2AuthorNormName', None) is not None) and (sig2.get('s2AuthorNormName', None) is not None):
        sum += 0.5
        if sig1['s2AuthorNormName'] == sig2['s2AuthorNormName']:
            score += 0.5
    if (sig1.get('fosIdsLevel0', None) is not None) and (sig2.get('fosIdsLevel0', None) is not None):
        sum += 0.5
        if set(sig1['fosIdsLevel0']) ==  set(sig2['fosIdsLevel0']):
            score += 0.5
    if (sig1.get('fosIdsLevel1', None) is not None) and (sig2.get('fosIdsLevel1', None) is not None):
        sum += 0.5
        if set(sig1['fosIdsLevel1']) ==  set(sig2['fosIdsLevel1']):
            score += 0.5
    if (sig1.get('affiliationIds', None) is not None) and (sig2.get('affiliationIds', None) is not None):
        sum += 0.5
        if set(sig1['affiliationIds']) ==  set(sig2['affiliationIds']):
            score += 0.5

    return 1 - score/sum

class NumpyDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return torch.Tensor(self.X[idx,:]), self.y[idx]

def plot_loss(train_loss, val_loss, path):

    plt.figure()
    plt.plot(range(1,len(train_loss)+1), train_loss)
    plt.plot(range(1,len(val_loss)+1), val_loss)
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train loss', 'Validation loss'])
    plt.savefig(path)