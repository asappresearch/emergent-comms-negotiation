import torch
from torch import nn, autograd
from torch.autograd import Variable
import torch.nn.functional as F


# In hindsight, all three of the next three classes are identical, and could be
# merged :)
class ContextNet(nn.Module):
    def __init__(self, embedding_size=100):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(11, embedding_size)
        self.lstm = nn.LSTMCell(
            input_size=embedding_size,
            hidden_size=embedding_size)
        self.zero_state = None

    def forward(self, x):
        batch_size = x.size()[0]
        seq_len = x.size()[1]
        x = x.transpose(0, 1)
        x = self.embedding(x)
        if x.is_cuda:
            state = (
                    Variable(torch.cuda.FloatTensor(batch_size, self.embedding_size).fill_(0)),
                    Variable(torch.cuda.FloatTensor(batch_size, self.embedding_size).fill_(0))
                )
        else:
            state = (
                    Variable(torch.zeros(batch_size, self.embedding_size)),
                    Variable(torch.zeros(batch_size, self.embedding_size))
                )

        for s in range(seq_len):
            state = self.lstm(x[s], state)
        return state[0]


class UtteranceNet(nn.Module):
    def __init__(self, embedding_size=100):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(11, embedding_size)
        self.lstm = nn.LSTMCell(
            input_size=embedding_size,
            hidden_size=embedding_size)

    def forward(self, x):
        batch_size = x.size()[0]
        seq_len = x.size()[1]
        x = x.transpose(0, 1)
        x = self.embedding(x)
        if x.is_cuda:
            state = (
                    Variable(torch.cuda.FloatTensor(batch_size, self.embedding_size).fill_(0)),
                    Variable(torch.cuda.FloatTensor(batch_size, self.embedding_size).fill_(0))
                )
        else:
            state = (
                    Variable(torch.zeros(batch_size, self.embedding_size)),
                    Variable(torch.zeros(batch_size, self.embedding_size))
                )
        for s in range(seq_len):
            state = self.lstm(x[s], state)
        return state[0]


class ProposalNet(nn.Module):
    def __init__(self, embedding_size=100):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(11, embedding_size)
        self.lstm = nn.LSTMCell(
            input_size=embedding_size,
            hidden_size=embedding_size)

    def forward(self, x):
        batch_size = x.size()[0]
        seq_len = x.size()[1]
        x = x.transpose(0, 1)
        x = self.embedding(x)
        if x.is_cuda:
            state = (
                    Variable(torch.cuda.FloatTensor(batch_size, self.embedding_size).fill_(0)),
                    Variable(torch.cuda.FloatTensor(batch_size, self.embedding_size).fill_(0))
                )
        else:
            state = (
                    Variable(torch.zeros(batch_size, self.embedding_size)),
                    Variable(torch.zeros(batch_size, self.embedding_size))
                )
        for s in range(seq_len):
            state = self.lstm(x[s], state)
        return state[0]


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

    def forward(self, x, eps=1e-8):
        x = self.h1(x)
        x = F.sigmoid(x)
        out_node = torch.bernoulli(x)
        x = x + eps
        entropy = - (x * x.log()).sum(1).sum()
        return out_node, entropy


class UtterancePolicy(nn.Module):
    def __init__(self, embedding_size=100, num_tokens=10, max_len=6):
        super().__init__()
        # use this to make onehot
        self.embedding_size = embedding_size
        self.onehot = torch.eye(num_tokens)
        self.num_tokens = num_tokens
        self.max_len = max_len
        self.lstm = nn.LSTM(
            input_size=num_tokens,
            hidden_size=embedding_size,
            num_layers=1
        )
        self.h1 = nn.Linear(embedding_size, num_tokens)

    def forward(self, h_t):
        batch_size = h_t.size()[0]

        state = (
            h_t.view(1, batch_size, self.embedding_size),
            Variable(torch.zeros(1, batch_size, self.embedding_size))
        )

        # use first token as the initial dummy token
        last_token = torch.zeros(batch_size).long()
        tokens = []
        while len(tokens) < self.max_len:
            token_onehot = self.onehot[last_token]
            token_onehot = token_onehot.view(1, batch_size, self.num_tokens)
            out, state = self.lstm(Variable(token_onehot), state)
            out = self.h1(out)
            out = F.softmax(out)
            token_node = torch.multinomial(out.view(batch_size, self.num_tokens))
            tokens.append(token_node)
            last_token = token_node.data.view(batch_size)
        return tokens


class ProposalPolicy(nn.Module):
    def __init__(self, embedding_size=100, num_counts=6):
        super().__init__()
        self.num_counts = num_counts
        self.embedding_size = embedding_size
        self.h1 = nn.Linear(embedding_size, num_counts)

    def forward(self, x, eps=1e-8):
        x1 = self.h1(x)
        x = F.softmax(x1)
        out_node = torch.multinomial(x)
        x = x + eps
        entropy = (- x * x.log()).sum(1).sum()
        return out_node, entropy


class AgentModel(nn.Module):
    def __init__(
            self, enable_comms, enable_proposal,
            term_entropy_reg,
            proposal_entropy_reg,
            embedding_size=100):
        super().__init__()
        self.term_entropy_reg = term_entropy_reg
        self.proposal_entropy_reg = proposal_entropy_reg
        self.embedding_size = embedding_size
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

    def forward(self, context, m_prev, prev_proposal):
        batch_size = context.size()[0]
        # print('batch_size', batch_size)
        c_h = self.context_net(context)
        if self.enable_comms:
            m_h = self.utterance_net(m_prev)
        else:
            if context.is_cuda:
                m_h = Variable(torch.cuda.FloatTensor(batch_size, self.embedding_size).fill_(0))
            else:
                m_h = Variable(torch.zeros(batch_size, self.embedding_size))
        p_h = self.proposal_net(prev_proposal)

        h_t = torch.cat([c_h, m_h, p_h], -1)
        h_t = self.combined_net(h_t)

        entropy_loss = 0
        term_node, entropy = self.term_policy(h_t)
        entropy_loss -= entropy * self.term_entropy_reg
        utterance_token_nodes = []
        if self.enable_comms:
            utterance_token_nodes = self.utterance_policy(h_t)
        proposal_nodes = []
        for proposal_policy in self.proposal_policies:
            proposal_node, _entropy = proposal_policy(h_t)
            proposal_nodes.append(proposal_node)
            entropy_loss -= self.proposal_entropy_reg * _entropy
        return term_node, utterance_token_nodes, proposal_nodes, entropy_loss
