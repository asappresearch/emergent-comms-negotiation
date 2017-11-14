import torch
from torch import nn, autograd
from torch.autograd import Variable
import torch.nn.functional as F


class NumberSequenceEncoder(nn.Module):
    def __init__(self, num_values, embedding_size=100):
        """
        eg for values 0,1,2,3,4,5, num_values will be: 6
        for 0,1,..,9 num_values will be: 10
        """
        super().__init__()
        self.embedding_size = embedding_size
        self.num_values = num_values
        self.embedding = nn.Embedding(num_values, embedding_size)
        self.lstm = nn.LSTMCell(
            input_size=embedding_size,
            hidden_size=embedding_size)
        self.zero_state = None

    def forward(self, x):
        batch_size = x.size()[0]
        seq_len = x.size()[1]
        x = x.transpose(0, 1)
        x = self.embedding(x)
        type_constr = torch.cuda if x.is_cuda else torch
        state = (
                Variable(type_constr.FloatTensor(batch_size, self.embedding_size).fill_(0)),
                Variable(type_constr.FloatTensor(batch_size, self.embedding_size).fill_(0))
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

    def forward(self, x, testing, eps=1e-8):
        x = self.h1(x)
        x = F.sigmoid(x)
        matches_argmax_count = 0

        if not testing:
            out_node = torch.bernoulli(x)
        else:
            _, out_node = x.max(1)
            out_node = out_node.view(-1, 1).float()

        _, argmax_res = x.data.max(1)
        matches_argmax = argmax_res == out_node.data.long().view(-1)
        matches_argmax_count = matches_argmax.int().sum()
        x = x + eps
        entropy = - (x * x.log()).sum(1).sum()
        return out_node, out_node.data.byte(), entropy, matches_argmax_count


class UtterancePolicy(nn.Module):
    def __init__(self, embedding_size=100, num_tokens=10, max_len=6):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_tokens = num_tokens
        self.max_len = max_len
        self.embedding = nn.Embedding(num_tokens, embedding_size)
        self.lstm = nn.LSTMCell(
            input_size=embedding_size,
            hidden_size=embedding_size
        )
        self.h1 = nn.Linear(embedding_size, num_tokens)

    def forward(self, h_t, testing, eps=1e-8):
        batch_size = h_t.size()[0]

        type_constr = torch.cuda if h_t.is_cuda else torch
        h =h_t
        c = Variable(type_constr.FloatTensor(batch_size, self.embedding_size).fill_(0))

        matches_argmax_count = 0
        last_token = type_constr.LongTensor(batch_size).fill_(0)
        utterance_nodes = []
        type_constr = torch.cuda if h_t.is_cuda else torch
        utterance = type_constr.LongTensor(batch_size, self.max_len).fill_(0)
        entropy = 0
        matches_argmax_count = 0
        stochastic_draws = 0
        for i in range(6):
            embedded = self.embedding(Variable(last_token))
            h, c = self.lstm(embedded, (h, c))
            out = self.h1(h)
            out = F.softmax(out)

            if not testing:
                token_node = torch.multinomial(out)
            else:
                _, token_node = out.max(1)
                token_node = token_node.view(-1, 1)

            _, argmax_res = out.data.max(1)
            matches_argmax = argmax_res == token_node.data.long().view(-1)
            matches_argmax_count += matches_argmax.int().sum()
            stochastic_draws += batch_size

            utterance_nodes.append(token_node)
            last_token = token_node.data.view(batch_size)
            utterance[:, i] = last_token
            out = out + eps
            entropy -= (out * out.log()).sum(1).sum()
        return utterance_nodes, utterance, entropy, matches_argmax_count, stochastic_draws


class ProposalPolicy(nn.Module):
    def __init__(self, embedding_size=100, num_counts=6, num_items=3):
        super().__init__()
        self.num_counts = num_counts
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.fcs = []
        for i in range(num_items):
            fc = nn.Linear(embedding_size, num_counts)
            self.fcs.append(fc)
            self.__setattr__('h1_%s' % i, fc)

    def forward(self, x, testing, eps=1e-8):
        batch_size = x.size()[0]
        nodes = []
        entropy = 0
        matches_argmax_count = 0
        type_constr = torch.cuda if x.is_cuda else torch
        matches_argmax_count = 0
        stochastic_draws = 0
        proposal = type_constr.LongTensor(batch_size, self.num_items).fill_(0)
        for i in range(self.num_items):
            x1 = self.fcs[i](x)
            x2 = F.softmax(x1)
            if not testing:
                node = torch.multinomial(x2)
            else:
                _, node = x2.max(1)
                node = node.view(-1, 1)

            _, argmax_res = x2.data.max(1)
            matches_argmax = argmax_res == node.data.long().view(-1)
            matches_argmax_count += matches_argmax.int().sum()
            stochastic_draws += batch_size

            nodes.append(node)
            x2 = x2 + eps
            entropy += (- x2 * x2.log()).sum(1).sum()
            proposal[:, i] = node.data

        return nodes, proposal, entropy, matches_argmax_count, stochastic_draws


class AgentModel(nn.Module):
    def __init__(
            self, enable_comms, enable_proposal,
            term_entropy_reg,
            utterance_entropy_reg,
            proposal_entropy_reg,
            embedding_size=100):
        super().__init__()
        self.term_entropy_reg = term_entropy_reg
        self.utterance_entropy_reg = utterance_entropy_reg
        self.proposal_entropy_reg = proposal_entropy_reg
        self.embedding_size = embedding_size
        self.enable_comms = enable_comms
        self.enable_proposal = enable_proposal
        self.context_net = NumberSequenceEncoder(num_values=6)
        self.utterance_net = NumberSequenceEncoder(num_values=10)
        self.proposal_net = NumberSequenceEncoder(num_values=6)
        self.proposal_net.embedding = self.context_net.embedding

        self.combined_net = CombinedNet()

        self.term_policy = TermPolicy()
        self.utterance_policy = UtterancePolicy()
        self.proposal_policy = ProposalPolicy()

    def forward(self, pool, utility, m_prev, prev_proposal, testing):
        """
        setting testing to True disables stochasticity: always picks the argmax
        cannot use this when training
        """
        batch_size = pool.size()[0]
        context = torch.cat([pool, utility], 1)
        c_h = self.context_net(context)
        type_constr = torch.cuda if context.is_cuda else torch
        if self.enable_comms:
            m_h = self.utterance_net(m_prev)
        else:
            m_h = Variable(type_constr.FloatTensor(batch_size, self.embedding_size).fill_(0))
        p_h = self.proposal_net(prev_proposal)

        h_t = torch.cat([c_h, m_h, p_h], -1)
        h_t = self.combined_net(h_t)

        entropy_loss = 0
        nodes = []

        term_node, term_a, entropy, term_matches_argmax_count = self.term_policy(h_t, testing=testing)
        nodes.append(term_node)
        entropy_loss -= entropy * self.term_entropy_reg

        utterance = None
        if self.enable_comms:
            utterance_nodes, utterance, utterance_entropy, utt_matches_argmax_count, utt_stochastic_draws = self.utterance_policy(
                h_t, testing=testing)
            nodes += utterance_nodes
            entropy_loss -= self.utterance_entropy_reg * utterance_entropy
        else:
            utterance = type_constr.LongTensor(batch_size, 6).zero_()  # hard-coding 6 here is a bit hacky...

        proposal_nodes, proposal, proposal_entropy, prop_matches_argmax_count, prop_stochastic_draws = self.proposal_policy(
            h_t, testing=testing)
        nodes += proposal_nodes
        entropy_loss -= self.proposal_entropy_reg * proposal_entropy

        return nodes, term_a, utterance, proposal, entropy_loss, \
            term_matches_argmax_count, utt_matches_argmax_count, utt_stochastic_draws, prop_matches_argmax_count, prop_stochastic_draws
