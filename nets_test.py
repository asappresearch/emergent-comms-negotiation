"""
This doesnt run with py.test, runs like `python nets_test.py [args ... ]`

Thats because this is going to run training, not very unit-testy :)

We want to check things like:
- does the context encoder work ok?
- can the termination policy learn?

We're going to try this with very stupid numbers first. If it doesnt work with stupid numbers, theres a problem :)
"""
import argparse
import json
import datetime
import time
import os
from os import path

import yaml
import torch
from torch import nn, autograd, optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

import nets


def test_context():
    """
    So, we will have two contexts:
    - previous proposal 0,0,0
    - previous proposal 3,3,3

    let's check the encodings are different :)
    Then, we will traing the terminator to terminate on the second, and not on the first

    Then we will repeat with some other numbers
    """
    train = [
        {'proposal': [0,0,0], 'term': 0},
        {'proposal': [3,3,3], 'term': 1}
    ]
    """
    also, lets use a small embedding, say 20, for speed
    and lets have only 4 possible values

    note: we should alos try with cuda and not cuda
    """
    embedding_size = 20

    proposalencoder = nets.NumberSequenceEncoder(num_values=4, embedding_size=embedding_size)
    combiner = nets.CombinedNet(num_sources=1, embedding_size=embedding_size)
    term_policy = nets.TermPolicy(embedding_size=embedding_size)

    params = set(proposalencoder.parameters()) | set(combiner.parameters()) | set(term_policy.parameters())
    opt = optim.Adam(lr=0.001, params=params)

    episode = 0
    N = len(train)
    prop_K = len(train[0]['proposal'])
    num_episodes = 10000
    batch_X = torch.LongTensor(N, prop_K).fill_(0)
    batch_term = torch.ByteTensor(N).fill_(0)
    for n in range(N):
        batch_X[n] = torch.LongTensor(train[n]['proposal'])
        batch_term[n] = train[n]['term']
    print('batch_X', batch_X)
    print('batch_term', batch_term)
    baseline = 0
    while True:
        pred_enc = proposalencoder(Variable(batch_X))
        combined = combiner(pred_enc)
        term_probs, term_node, term_a, entropy, argmax_matches = term_policy(combined, testing=False)
        reward = (term_a.view(-1) == batch_term).float()

        opt.zero_grad()
        term_node.reinforce(reward.view(-1, 1) - baseline)
        autograd.backward([term_node], [None])
        opt.step()

        baseline = 0.7 * baseline + 0.3 * reward.mean()

        # print('reward', reward)
        # print('term_probs.data', term_probs.data)
        # print('term_a', term_a)
        # print('entropy.data', entropy.data[0])
        # print('argmax_matches', argmax_matches)
        # asdf
        # print(term_a.view(-1) == batch_term)
        num_right = (term_a.view(-1) == batch_term).int().sum()
        if episode % 100 == 0:
            print('episode', episode, 'num_right', num_right, 'baseline', baseline)

        episode += 1
        if episode >= num_episodes:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parsers = parser.add_subparsers()

    parser_ = parsers.add_parser('test-context')
    parser_.set_defaults(func=test_context)

    args = parser.parse_args()
    func = args.func
    del args.__dict__['func']
    func(**args.__dict__)
