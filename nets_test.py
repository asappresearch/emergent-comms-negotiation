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
import sampling


def test_context(term_entropy_reg, embedding_size, num_values, batch_size):
    """
    So, we will have two contexts:
    - previous proposal 0,0,0
    - previous proposal 3,3,3

    let's check the encodings are different :)
    Then, we will traing the terminator to terminate on the second, and not on the first

    Then we will repeat with some other numbers
    """
    train = []
    N = batch_size
    seen_proposals = set()
    for n in range(N):
        seen = True
        while seen:
            proposal = sampling.sample_items(batch_size=1, num_values=num_values).view(-1)
            hash = proposal[0] * 100 + proposal[1] * 10 + proposal[0]
            seen = hash in seen_proposals
            seen_proposals.add(seen)
        term = np.random.choice(2, 1).item()
        train.append({'proposal': proposal, 'term': term})


    """
    also, lets use a small embedding, say 20, for speed
    and lets have only 4 possible values

    note: we should alos try with cuda and not cuda
    """

    proposalencoder = nets.NumberSequenceEncoder(num_values=num_values, embedding_size=embedding_size)
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
    baseline = 0
    while True:
        pred_enc = proposalencoder(Variable(batch_X))
        combined = combiner(pred_enc)
        term_probs, term_node, term_a, entropy, argmax_matches = term_policy(combined, testing=False)
        reward = (term_a.view(-1) == batch_term).float()

        opt.zero_grad()
        term_node.reinforce(reward.view(-1, 1) - baseline)
        autograd.backward([term_node, - term_entropy_reg * entropy], [None, None])
        opt.step()

        baseline = 0.7 * baseline + 0.3 * reward.mean()

        num_right = (term_a.view(-1) == batch_term).int().sum()
        if episode % 100 == 0:
            term_probs, term_node, term_a, entropy, argmax_matches = term_policy(combined, testing=True)
            reward_val = (term_a.view(-1) == batch_term).float()
            print('episode', episode, 'num_right', num_right, 'baseline', baseline, 'reward_val', reward_val.float().mean())

        episode += 1
        if episode >= num_episodes:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parsers = parser.add_subparsers()

    parser_ = parsers.add_parser('test-context')
    parser_.add_argument('--term-entropy-reg', type=float, default=0.05)
    parser_.add_argument('--embedding-size', type=int, default=100)
    parser_.add_argument('--num-values', type=int, default=6)
    parser_.add_argument('--batch-size', type=int, default=128)
    parser_.set_defaults(func=test_context)

    args = parser.parse_args()
    func = args.func
    del args.__dict__['func']
    func(**args.__dict__)
