import tensorflow as tf
import numpy as np
from config import config
import pandas as pd

def add_noise(data, percent_noise):
    length = data.shape[0]
    size = int(length * percent_noise)
    #randomly choose indices
    chosen_idx = np.random.choice(length, replace = True, size = size)
    #then + noise
    noise = np.random.normal(size = size, scale = 6)
    data.iloc[chosen_idx] = (data.iloc[chosen_idx].T + noise).T
    return data

class ContrastiveGenerator(keras.utils.Sequence):
    def __init__(self, data, mode="train"):
        '''
        data: dictionary with format {'vctA': 1D numpy array, 'vctB': 1D numpy array}
        '''
        self.dim = config['dim']
        self.bs = config['batch_size']
        self.data = data

    def __len__(self):
        return len(self.data) // self.bs)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch = self.data[index * self.bs:(index+1) * self.bs]
        X = []
        for b in batch:
            A, B = b['vctA'], b['vctB']
            A1 = add_noise(A, percent_noise=1)
            A2 = add_noise(A, percent_noise=1)
            B1 = add_noise(B, percent_noise=1)
            B2 = add_noise(B, percent_noise=1)
            X.append[]

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.data)


def get_constrastive_data():
    data_A = pd.read_csv(config['data_A'], sep=',', header=0, index_col=0)
    data_B = pd.read_csv(config['data_B'], sep=',', header=0, index_col=0)
    data = [{'A':A, 'B':B} for A, B in zip(list(data_A), list(data_B))]
    idx = 0.2*len(data)
    train_data, val_data = data[idx:], data[:idx]
    train_gen = ContrastiveGenerator(train_data, mode='train')
    val_gen = ContrastiveGenerator(val_data, mode='val')
    return train_gen. val_gen
