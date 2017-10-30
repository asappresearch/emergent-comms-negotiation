import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import argparse


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
            token_node = torch.multinomial(out)
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
        agent_opts):
    N = sample_N()
    pool = sample_items()
    utilities = torch.zeros(2, 3).long()
    utilities[0] = sample_utility()
    utilities[1] = sample_utility()
    last_proposal = torch.zeros(3).long()
    m_prev = torch.zeros(1).long()
    p_prev = torch.zeros(3).long()
    nodes_by_agent = [[], []]
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
        if term_node.data[0][0]:
            break
        else:
            last_proposal = torch.LongTensor([
                proposal_nodes[0].data[0][0],
                proposal_nodes[1].data[0][0],
                proposal_nodes[2].data[0][0],
            ])
    rewards = [None, None]
    # so, lets say agent is 1, means the previous proposal was
    # by agent 0
    proposing_agent = 1 - agent
    rewards[proposing_agent] = utilities[proposing_agent].dot(last_proposal)
    rewards[agent] = utilities[agent].dot(pool - last_proposal)
    if prosocial:
        total_actual_reward = np.sum(rewards)
        max_utility = torch.max(*utilities)
        total_possible_reward = max_utility.dot(pool)
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


def run(enable_proposal, enable_comms, seed, prosocial):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    episode = 0
    agent_models = []
    agent_opts = []
    for i in range(2):
        agent_models.append(AgentModel(
            enable_comms=enable_comms,
            enable_proposal=enable_proposal))
        agent_opts.append(optim.Adam(params=agent_models[i].parameters()))
    while True:
        run_episode(
            agent_models=agent_models,
            agent_opts=agent_opts,
            prosocial=prosocial)
        episode += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='optional')
    parser.add_argument('--disable-proposal', action='store_true')
    parser.add_argument('--disable-comms', action='store_true')
    parser.add_argument('--disable-prosocial', action='store_true')
    args = parser.parse_args()
    args.enable_comms = not args.disable_comms
    args.enable_proposal = not args.disable_proposal
    args.prosocial = not args.disable_prosocial
    del args.__dict__['disable_comms']
    del args.__dict__['disable_proposal']
    del args.__dict__['disable_prosocial']
    run(**args.__dict__)
