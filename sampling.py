import torch
import numpy as np


def sample_items(batch_size):
    pool = torch.from_numpy(np.random.choice(6, (batch_size, 3), replace=True))
    return pool


def sample_utility(batch_size):
    u = torch.zeros(3).long()
    while u.sum() == 0:
        u = torch.from_numpy(np.random.choice(11, (batch_size, 3), replace=True))
    return u


def sample_N(batch_size):
    N = np.random.poisson(7, batch_size)
    N = np.maximum(4, N)
    N = np.minimum(10, N)
    N = torch.from_numpy(N)
    return N
