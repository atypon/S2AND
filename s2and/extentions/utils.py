import json
import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from typing import List, Dict
from os.path import join
from s2and.data import ANDData


def load_dataset(dataset_name: str) -> ANDData:

    parent_dir = join('data', dataset_name)
    if not os.path.isfile(join('pickled_datasets', f'{dataset_name}.pickle')):
        dataset = ANDData(
            signatures=join(parent_dir, f"{dataset_name}_signatures.json"),
            papers=join(parent_dir, f"{dataset_name}_papers.json"),
            mode="train",
            specter_embeddings=None,
            clusters=join(parent_dir, f"{dataset_name}_clusters.json"),
            block_type="s2",
            train_pairs_size=100000,
            val_pairs_size=10000,
            test_pairs_size=10000,
            name=dataset_name,
            n_jobs=4,
            preprocess=False
        )
        with open(join('pickled_datasets', f'{dataset_name}.pickle'), 'wb') as f:
            pickle.dump(dataset, f)
    else:
        with open(join('pickled_datasets', f'{dataset_name}.pickle'), 'rb') as f:
            dataset = pickle.load(f)
            print('Loaded dataset from pickle...')
    return dataset


def load_signatures(dataset: str) -> Dict[str, dict]:
    """
    Load sigatures data and store them in dict
    """
    with open(join('extended_data', f'{dataset}-signatures.json')) as f:
        content = f.read().split('\n')
        if '' in content:
            content.remove('')
        content = list(map(json.loads, content))

    return {entry['signature_id']: entry for entry in content}


def get_block_dict(signatures: Dict[str, dict]) -> Dict[str, List[str]]:
    """
    Given dict of signatures, return the dict of blocks containing
    block name : list of siagnatures in this block
    """
    block_dict = {}
    for item in signatures.values():
        if item['block'] not in block_dict:
            block_dict[item['block']] = [item['signature_id']]
        else:
            block_dict[item['block']].append(item['signature_id'])
    return block_dict


class NumpyDataset(Dataset):
    """
    Custom Pytorch Dataset class
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.Tensor(self.X[idx, :]), self.y[idx]


def plot_loss(train_loss: List[float],
              val_loss: List[float],
              path) -> None:
    """
    User for plotting the loss of NN model
    """
    plt.figure()
    plt.plot(range(1, len(train_loss)+1), train_loss)
    plt.plot(range(1, len(val_loss)+1), val_loss)
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train loss', 'Validation loss'])
    plt.savefig(path)
