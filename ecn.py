import json
import time
import argparse
import os
import datetime
from os import path
import numpy as np
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
import torch.nn.functional as F


def sample_items():
    pool = torch.from_numpy(np.random.choice(6, 3, replace=True))
    return pool


def sample_utility():
    u = torch.zeros(3).long()
    while u.sum() == 0:
        u = torch.from_numpy(np.random.choice(11, 3, replace=True))
    return u


def sample_N():
    N = np.random.poisson(7)
    N = max(4, N)
    N = min(10, N)
    return N


# In hindsight, all three of these classes are identical, and could be
# merged :)
class ContextNet(nn.Module):
    def __init__(self, embedding_size=100):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(11, embedding_size)
        self.lstm = nn.GRU(
            input_size=embedding_size,
            hidden_size=embedding_size,
            num_layers=1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(-1, 1, self.embedding_size)
        state = Variable(torch.zeros(1, 1, self.embedding_size))
        x, state = self.lstm(x, state)
        return state[0].view(1, -1)


class UtteranceNet(nn.Module):
    def __init__(self, embedding_size=100):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(11, embedding_size)
        # using GRU, since means the hidden state exactly matches the embedding size,
        # dont need to think about how to handle the presence of both cell and
        # hidden states
        self.lstm = nn.GRU(
            input_size=embedding_size,
            hidden_size=embedding_size,
            num_layers=1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(-1, 1, self.embedding_size)
        state = Variable(torch.zeros(1, 1, self.embedding_size))
        x, state = self.lstm(x, state)
        return state[0].view(1, -1)


class ProposalNet(nn.Module):
    def __init__(self, embedding_size=100):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(11, embedding_size)
        self.lstm = nn.GRU(
            input_size=embedding_size,
            hidden_size=embedding_size,
            num_layers=1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(-1, 1, self.embedding_size)
        state = Variable(torch.zeros(1, 1, self.embedding_size))
        x, state = self.lstm(x, state)
        return state[0].view(1, -1)


class CombinedNet(nn.Module):
    def __init__(self, embedding_size=100):
        super().__init__()
        self.embedding_size = embedding_size
        self.h1 = nn.Linear(embedding_size * 3, embedding_size)

    def forward(self, x):
        x = self.h1(x)
        x = F.relu(x)
        return x


class TermPolicy(nn.Module):
    def __init__(self, embedding_size=100):
        super().__init__()
        self.h1 = nn.Linear(embedding_size, 1)

    def forward(self, x):
        x = self.h1(x)
        x = F.sigmoid(x)
        # print('term x.size()', x.size())
        out_node = torch.bernoulli(x)
        return out_node


class UtterancePolicy(nn.Module):
    def __init__(self, embedding_size=100, num_tokens=11, max_len=6):
        super().__init__()
        # use this to make onehot
        self.onehot = torch.eye(num_tokens)
        self.num_tokens = num_tokens
        self.max_len = max_len
        self.lstm = nn.GRU(
            input_size=num_tokens,
            hidden_size=embedding_size,
            num_layers=1
        )
        self.h1 = nn.Linear(embedding_size, num_tokens)

    def forward(self, h_t):
        state = h_t.view(1, 1, -1)
        # use first token as the initial dummy token
        last_token = 0
        tokens = []
        # use last token of vocab as end-of-utterance
        while last_token != self.num_tokens - 1 and len(tokens) < self.max_len:
            token_onehot = self.onehot[last_token].view(1, 1, -1)
            out, state = self.lstm(Variable(token_onehot), state)
            out = self.h1(out)
            out = F.softmax(out)
            # print('utt out.size()', out.size())
            # print('out.requires_grad', out.requires_grad)
            token_node = torch.multinomial(out.view(1, -1))
            # print('token_node.requires_grad', token_node.requires_grad)
            tokens.append(token_node)
            last_token = token_node.data[0][0]
        return tokens


class ProposalPolicy(nn.Module):
    def __init__(self, embedding_size=100, num_counts=6):
        super().__init__()
        self.num_counts = num_counts
        self.embedding_size = embedding_size
        self.h1 = nn.Linear(embedding_size, num_counts)

    def forward(self, x):
        x = self.h1(x)
        x = F.softmax(x)
        out_node = torch.multinomial(x)
        return out_node


class AgentModel(nn.Module):
    def __init__(self, enable_comms, enable_proposal):
        super().__init__()
        self.enable_comms = enable_comms
        self.enable_proposal = enable_proposal
        self.context_net = ContextNet()
        self.utterance_net = UtteranceNet()
        self.proposal_net = ProposalNet()
        self.proposal_net.embedding = self.context_net.embedding

        self.combined_net = CombinedNet()

        self.term_policy = TermPolicy()
        self.utterance_policy = UtterancePolicy()
        self.proposal_policies = []
        for i in range(3):
            proposal_policy = ProposalPolicy()
            self.proposal_policies.append(proposal_policy)
            # do this so it registers its parameters:
            self.__setattr__('policy%s' % i, proposal_policy)

    def forward(self, context, m_prev, p_prev):
        c_h = self.context_net(context)
        m_h = self.utterance_net(m_prev)
        p_h = self.proposal_net(p_prev)

        h_t = torch.cat([c_h, m_h, p_h], -1)
        h_t = self.combined_net(h_t)

        term_node = self.term_policy(h_t)
        utterance_token_nodes = []
        if self.enable_comms:
            utterance_token_nodes = self.utterance_policy(h_t)
        proposal_nodes = []
        if self.enable_proposal:
            for proposal_policy in self.proposal_policies:
                proposal_node = proposal_policy(h_t)
                proposal_nodes.append(proposal_node)
        return term_node, utterance_token_nodes, proposal_nodes


class Agent(object):
    """
    holds model, optimizer, etc
    """
    def __init__(self, enable_comms, enable_proposal):
        self.enable_comms = enable_comms
        self.enable_proposal = enable_proposal
        self.model = AgentModel(
            enable_comms=enable_comms,
            enable_proposal=enable_proposal)
        self.opt = optim.Adam(params=self.model.parameters())


def run_episode(
        prosocial,
        agent_models,
        render=False):
    N = sample_N()
    pool = sample_items()
    utilities = torch.zeros(2, 3).long()
    utilities[0] = sample_utility()
    utilities[1] = sample_utility()
    last_proposal = torch.zeros(3).long()
    m_prev = torch.zeros(1).long()
    p_prev = torch.zeros(3).long()
    nodes_by_agent = [[], []]
    if render:
        print('  N=%s' % N, end='')
        print(' pool: %s,%s,%s' % (pool[0], pool[1], pool[2]), end='')
        print(' util:', end='')
        for i in range(2):
            print(' %s,%s,%s' % (utilities[i][0], utilities[i][2], utilities[i][2]), end='')
        print('')
    terminated_ok = False
    for t in range(N):
        agent = 1 if t % 2 else 0
        utility = utilities[agent]
        c = torch.cat([pool, utility]).view(1, -1)
        agent_model = agent_models[agent]
        term_node, utterance_nodes, proposal_nodes = agent_model(
            context=Variable(c),
            m_prev=Variable(m_prev.view(1, -1)),
            p_prev=Variable(p_prev.view(1, -1))
        )
        nodes_by_agent[agent].append(term_node)
        nodes_by_agent[agent] += utterance_nodes
        nodes_by_agent[agent] += proposal_nodes
        if render:
            print('  t %s term=%s' % (t, term_node.data[0][0]), end='')
            print(' prop %s,%s,%s' % (
                proposal_nodes[0].data[0][0],
                proposal_nodes[1].data[0][0],
                proposal_nodes[2].data[0][0]
            ))
        if term_node.data[0][0]:
            terminated_ok = True
            break
        else:
            last_proposal = torch.LongTensor([
                proposal_nodes[0].data[0][0],
                proposal_nodes[1].data[0][0],
                proposal_nodes[2].data[0][0],
            ])
    rewards = [0, 0]
    # so, lets say agent is 1, means the previous proposal was
    # by agent 0
    proposing_agent = 1 - agent
    # cap last proposal by pool size
    exceeded_pool = False
    # if render:
    #     print(last_proposal, torch.min(pool, last_proposal))
    if t == 0:
        terminated_ok = False
    if (last_proposal - torch.min(pool, last_proposal)).sum() != 0:
        exceeded_pool = True
    if render:
        print('term ok %s exceeded_pool %s' % (terminated_ok, exceeded_pool))
    if not exceeded_pool and terminated_ok:
        rewards[proposing_agent] = utilities[proposing_agent].dot(last_proposal)
        rewards[agent] = utilities[agent].dot(pool - last_proposal)
        if prosocial:
            total_actual_reward = np.sum(rewards)
            max_utility = torch.max(*utilities)
            total_possible_reward = max_utility.dot(pool)
            # if render:
            #     print('rewards', rewards)
            #     print('utilities', utilities)
            #     print('max_utility', max_utility)
            #     print('total_actual_reward %s total_possible_reward %s' (
            #         total_actual_reward, total_possible_reward))
            scaled_reward = 0
            if total_possible_reward != 0:
                scaled_reward = total_actual_reward / total_possible_reward
            rewards[0] = scaled_reward
            rewards[1] = scaled_reward
        else:
            for i in range(2):
                max_possible = utilities[i].dot(pool)
                if max_possible != 0:
                    rewards[i] /= max_possible
    if render:
        print('  reward: %.1f,%.1f' % (rewards[0], rewards[1]))
    return nodes_by_agent, rewards


def run(enable_proposal, enable_comms, seed, prosocial, logfile, model_file):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    episode = 0
    start_time = time.time()
    agent_models = []
    agent_opts = []
    for i in range(2):
        agent_models.append(AgentModel(
            enable_comms=enable_comms,
            enable_proposal=enable_proposal))
        agent_opts.append(optim.Adam(params=agent_models[i].parameters()))
    # print(agent_models[0].state_dict())
    # asdfasdf
    if path.isfile(model_file):
        with open(model_file, 'rb') as f:
            state = torch.load(f)
        for i in range(2):
            agent_models[i].load_state_dict(state['agent%s' % i]['model_state'])
            agent_opts[i].load_state_dict(state['agent%s' % i]['opt_state'])
        episode = state['episode']
        # create a kind of 'virtual' start_time
        start_time = time.time() - state['elapsed_time']
        print('loaded model')
            # agent_opts.append(optim.Adam(params=agent_models[i].parameters()))
    last_print = time.time()
    rewards_sum = [0, 0]
    count_sum = 0
    for d in ['logs', 'model_saves']:
        if not path.isdir(d):
            os.makedirs(d)
    f_log = open(logfile, 'w')
    f_log.write('meta: %s\n' % json.dumps({
        'enable_proposal': enable_proposal,
        'enable_comms': enable_comms,
        'prosocial': prosocial,
        'seed': seed
    }))
    last_save = time.time()
    baseline = 0
    while True:
        render = time.time() - last_print >= 3.0
        nodes_by_agent, rewards = run_episode(
            agent_models=agent_models,
            prosocial=prosocial,
            render=render)
        for i in range(2):
            if len(nodes_by_agent[i]) == 0:
                continue
            rewards_sum[i] += rewards[i]
            reward = rewards[i]
            for node in nodes_by_agent[i]:
                node.reinforce(reward - baseline)
            agent_opts[i].zero_grad()
            autograd.backward(nodes_by_agent[i], [None] * len(nodes_by_agent[i]))
            agent_opts[i].step()
        baseline = 0.7 * baseline + 0.3 * (np.mean(rewards))
        count_sum += 1
        if render:
            print('episode %s avg rewards %.1f %.1f b=%.1f' % (
                episode, rewards_sum[0] / count_sum, rewards_sum[1] / count_sum, baseline))
            f_log.write(json.dumps({
                'episode': episode,
                'avg_reward_0': rewards_sum[0] / count_sum,
                'avg_reward_1': rewards_sum[1] / count_sum,
                'elapsed': time.time() - start_time
            }) + '\n')
            f_log.flush()
            last_print = time.time()
            rewards_sum = [0, 0]
            count_sum = 0
        if time.time() - last_save >= 5.0:
            state = {}
            for i in range(2):
                state['agent%s' % i] = {}
                state['agent%s' % i]['model_state'] = agent_models[i].state_dict()
                state['agent%s' % i]['opt_state'] = agent_opts[i].state_dict()
            state['episode'] = episode
            state['elapsed_time'] = time.time() - start_time
            with open(model_file, 'wb') as f:
                torch.save(state, f)
            print('saved model')
            last_save = time.time()

        episode += 1
    f_log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, default='model_saves/model.dat')
    parser.add_argument('--seed', type=int, help='optional')
    parser.add_argument('--disable-proposal', action='store_true')
    parser.add_argument('--disable-comms', action='store_true')
    parser.add_argument('--disable-prosocial', action='store_true')
    parser.add_argument('--logfile', type=str, default='logs/log_%Y%m%d_%H%M%S.log')
    args = parser.parse_args()
    args.enable_comms = not args.disable_comms
    args.enable_proposal = not args.disable_proposal
    args.prosocial = not args.disable_prosocial
    args.logfile = datetime.datetime.strftime(datetime.datetime.now(), args.logfile)
    del args.__dict__['disable_comms']
    del args.__dict__['disable_proposal']
    del args.__dict__['disable_prosocial']
    run(**args.__dict__)
