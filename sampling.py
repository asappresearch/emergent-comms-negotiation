import torch
import numpy as np


def sample_items(batch_size, num_values=6, seq_len=3):
    """
    num_values 6 will give possible values: 0,1,2,3,4,5
    """
    pool = torch.from_numpy(np.random.choice(num_values, (batch_size, seq_len), replace=True))
    return pool


def sample_utility(batch_size, num_values=6, seq_len=3):
    u = torch.zeros(seq_len).long()
    while u.sum() == 0:
        u = torch.from_numpy(np.random.choice(num_values, (batch_size, seq_len), replace=True))
    return u


def sample_N(batch_size):
    N = np.random.poisson(7, batch_size)
    N = np.maximum(4, N)
    N = np.minimum(10, N)
    N = torch.from_numpy(N)
    return N
