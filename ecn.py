"""
Note that this needs pytorch built from master, so that it works on cuda 9/v100.
"""
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


# In hindsight, all three of these classes are identical, and could be
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
        m = torch.distributions.Bernoulli(x)
        a = m.sample()
        eligibility = m.log_prob(a)
        a = a.data
        x = x + eps
        entropy = - (x * x.log()).sum(1).sum()
        return eligibility, a, entropy


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
            token_node = torch.distributions.Multinomial(out.view(batch_size, self.num_tokens)).sample()
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
        m = torch.distributions.Multinomial(x)
        a = m.sample()
        eligibility = m.log_prob(a)
        a = a.data
        x = x + eps
        entropy = (- x * x.log()).sum(1).sum()
        return eligibility, a, entropy


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
        term_e, term_a, entropy = self.term_policy(h_t)
        entropy_loss -= entropy * self.term_entropy_reg
        utterance_token_nodes = []
        if self.enable_comms:
            utterance_token_nodes = self.utterance_policy(h_t)
        proposal_as = []
        proposal_es = []
        for proposal_policy in self.proposal_policies:
            proposal_e, proposal_a, _entropy = proposal_policy(h_t)
            proposal_es.append(proposal_e)
            proposal_as.append(proposal_a)
            entropy_loss -= self.proposal_entropy_reg * _entropy
        return term_e, term_a, utterance_token_nodes, proposal_es, proposal_as, entropy_loss


def run_episode(
        enable_cuda,
        enable_comms,
        enable_proposal,
        prosocial,
        agent_models,
        batch_size,
        render=False):
    # following take not much memory, not fluffed up yet:
    N = sample_N(batch_size).int()
    pool = sample_items(batch_size)
    utilities = torch.zeros(batch_size, 2, 3).long()
    utilities[:, 0] = sample_utility(batch_size)
    utilities[:, 1] = sample_utility(batch_size)
    last_proposal = torch.zeros(batch_size, 3).long()
    m_prev = torch.zeros(batch_size, 6).long()

    if enable_cuda:
        N = N.cuda()
        pool = pool.cuda()
        utilities = utilities.cuda()
        last_proposal = last_proposal.cuda()
        m_prev = m_prev.cuda()

    games = []
    actions_by_timestep = []
    alive_masks = []
    for b in range(batch_size):
        games.append({'rewards': [0, 0]})
    alive_games = games.copy()

    if render:
        print('  N=%s' % N[0])
        print('  pool=%s,%s,%s' % (pool[0][0], pool[0][1], pool[0][2]))
        for i in range(2):
            print('  util[%s] %s,%s,%s' % (i, utilities[0][i][0], utilities[0][i][1], utilities[0][i][2]))
    b_0_present = True
    entropy_loss_by_agent = [Variable(torch.zeros(1)), Variable(torch.zeros(1))]
    if enable_cuda:
        entropy_loss_by_agent[0] = entropy_loss_by_agent[0].cuda()
        entropy_loss_by_agent[1] = entropy_loss_by_agent[1].cuda()
    for t in range(10):
        agent = 0 if t % 2 == 0 else 1
        batch_size = len(alive_games)
        utility = utilities[:, agent]

        c = torch.cat([pool, utility], 1)
        agent_model = agent_models[agent]
        term_e, term_a, utterance_nodes, proposal_es, proposal_as, _entropy_loss = agent_model(
            context=Variable(c),
            m_prev=Variable(m_prev),
            prev_proposal=Variable(last_proposal)
        )
        entropy_loss_by_agent[agent] += _entropy_loss
        if enable_comms:
            for i in range(6):
                m_prev[:, i] = utterance_nodes[i].data

        this_proposal = torch.zeros(batch_size, 3).long()
        if enable_cuda:
            this_proposal = this_proposal.cuda()
        for p in range(3):
            this_proposal[:, p] = proposal_as[p]

        actions_t = []
        actions_t.append(term_e)
        if enable_comms:
            actions_t += utterance_nodes
        if enable_proposal:
            actions_t += proposal_es
        if render and b_0_present:
            speaker = 'A' if agent == 0 else 'B'
            print('  %s t=%s' % (speaker, term_a[0][0]), end='')
            print(' u=' + ''.join([str(s) for s in m_prev[0].view(-1).tolist()]), end='')
            # print('proposal_as[0].size()', proposal_as[0].size())
            print(' p=%s,%s,%s' % (
                proposal_as[0][0],
                proposal_as[1][0],
                proposal_as[2][0]
            ), end='')
            print('')
        actions_by_timestep.append(actions_t)

        # calcualate rewards for any that just finished
        reward_eligible_mask = term_a.view(batch_size).clone().byte()
        if t == 0:
            # on first timestep theres no actual proposal yet, so score zero if terminate
            reward_eligible_mask.fill_(0)
        if reward_eligible_mask.max() > 0:
            exceeded_pool, _ = ((last_proposal - pool) > 0).max(1)
            if exceeded_pool.max() > 0:
                reward_eligible_mask[exceeded_pool.nonzero().long().view(-1)] = 0

        if reward_eligible_mask.max() > 0:
            proposer = 1 - agent
            accepter = agent
            proposal = torch.zeros(batch_size, 2, 3).long()
            proposal[:, proposer] = last_proposal
            proposal[:, accepter] = pool - last_proposal
            max_utility, _ = utilities.max(1)

            reward_eligible_idxes = reward_eligible_mask.nonzero().long().view(-1)
            for b in reward_eligible_idxes:
                rewards = [0, 0]
                for i in range(2):
                    rewards[i] = utilities[b, i].cpu().dot(proposal[b, i].cpu())

                if prosocial:
                    total_actual_reward = np.sum(rewards)
                    total_possible_reward = max_utility[b].cpu().dot(pool[b].cpu())
                    scaled_reward = 0
                    if total_possible_reward != 0:
                        scaled_reward = total_actual_reward / total_possible_reward
                    rewards = [scaled_reward, scaled_reward]
                    if render and b_0_present and b == 0:
                        print('  steps=%s reward=%.2f' % (t + 1, scaled_reward))
                else:
                    for i in range(2):
                        max_possible = utilities[b, i].cpu().dot(pool.cpu())
                        if max_possible != 0:
                            rewards[i] /= max_possible

                alive_games[b]['rewards'] = rewards

        still_alive_mask = 1 - term_a.view(batch_size).clone().byte()
        finished_N = t >= N
        if enable_cuda:
            finished_N = finished_N.cuda()
        still_alive_mask[finished_N] = 0
        alive_masks.append(still_alive_mask)

        dead_idxes = (1 - still_alive_mask).nonzero().long().view(-1)
        for b in dead_idxes:
            alive_games[b]['steps'] = t + 1

        if still_alive_mask.max() == 0:
            break

        # filter the state through the still alive mask:
        still_alive_idxes = still_alive_mask.nonzero().long().view(-1)
        if enable_cuda:
            still_alive_idxes = still_alive_idxes.cuda()
        pool = pool[still_alive_idxes]
        last_proposal = this_proposal[still_alive_idxes]
        utilities = utilities[still_alive_idxes]
        m_prev = m_prev[still_alive_idxes]
        N = N[still_alive_idxes]
        if still_alive_mask[0] == 0:
            b_0_present = False

        new_alive_games = []
        for i in still_alive_idxes:
            new_alive_games.append(alive_games[i])
        alive_games = new_alive_games

    for g in games:
        if 'steps' not in g:
            g['steps'] = 10

    return actions_by_timestep, [g['rewards'] for g in games], [g['steps'] for g in games], alive_masks, entropy_loss_by_agent


def run(enable_proposal, enable_comms, seed, prosocial, logfile, model_file, batch_size,
        term_entropy_reg, proposal_entropy_reg, enable_cuda):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    episode = 0
    start_time = time.time()
    agent_models = []
    agent_opts = []
    for i in range(2):
        model = AgentModel(
            enable_comms=enable_comms,
            enable_proposal=enable_proposal,
            term_entropy_reg=term_entropy_reg,
            proposal_entropy_reg=proposal_entropy_reg
        )
        if enable_cuda:
            model = model.cuda()
        agent_models.append(model)
        agent_opts.append(optim.Adam(params=agent_models[i].parameters()))
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
    last_print = time.time()
    rewards_sum = torch.zeros(2)
    steps_sum = 0
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
        # render = True
        actions, rewards, steps, alive_masks, entropy_loss_by_agent = run_episode(
            enable_cuda=enable_cuda,
            enable_comms=enable_comms,
            enable_proposal=enable_proposal,
            agent_models=agent_models,
            prosocial=prosocial,
            batch_size=batch_size,
            render=render)

        for i in range(2):
            agent_opts[i].zero_grad()
        # nodes_by_agent = [[], []]
        alive_rewards = torch.zeros(batch_size, 2)
        all_rewards = torch.zeros(batch_size, 2)
        for i in range(2):
            # note to self: just use .clone() or something...
            all_rewards[:, i] = torch.FloatTensor([r[i] for r in rewards])
            alive_rewards[:, i] = torch.FloatTensor([r[i] for r in rewards])
        if enable_cuda:
            all_rewards = all_rewards.cuda()
            alive_rewards = alive_rewards.cuda()
        alive_rewards -= baseline
        T = len(actions)
        loss_by_agent = [Variable(torch.zeros(1)), Variable(torch.zeros(1))]
        if enable_cuda:
            loss_by_agent = [Variable(torch.zeros(1).cuda()), Variable(torch.zeros(1).cuda())]
        for t in range(T):
            _batch_size = alive_rewards.size()[0]
            agent = 0 if t % 2 == 0 else 1
            if len(actions[t]) > 0:
                for action in actions[t]:
                    # print('type(action)', type(action))
                    # print('type(action.data)', type(action.data))
                    # print('type(alive_rewards[:, agent].contiguous().view(_batch_size, 1))', type(alive_rewards[:, agent].contiguous().view(_batch_size, 1)))
                    loss_by_agent[agent] -= (action * Variable(alive_rewards[:, agent].contiguous().view(_batch_size, 1))).sum()
                    # action.reinforce(alive_rewards[:, agent].contiguous().view(_batch_size, 1))
            # nodes_by_agent[agent] += actions[t]
            mask = alive_masks[t]
            if mask.max() == 0:
                break
            alive_rewards = alive_rewards[mask.nonzero().long().view(-1)]
        for i in range(2):
            loss_by_agent[i] += entropy_loss_by_agent[i]
            # if len(nodes_by_agent[i]) > 0:
                # autograd.backward([entropy_loss_by_agent[i]] + nodes_by_agent[i], [None] + len(nodes_by_agent[i]) * [None])
            loss_by_agent[i].backward()
            agent_opts[i].step()

        rewards_sum += all_rewards.sum(0).cpu()
        steps_sum += np.sum(steps)
        baseline = 0.7 * baseline + 0.3 * all_rewards.mean()
        count_sum += batch_size

        if render:
            time_since_last = time.time() - last_print
            print('episode %s avg rewards %.3f %.3f b=%.3f games/sec %s avg steps %.4f' % (
                episode,
                rewards_sum[0] / count_sum,
                rewards_sum[1] / count_sum,
                baseline,
                int(count_sum / time_since_last),
                steps_sum / count_sum
            ))
            f_log.write(json.dumps({
                'episode': episode,
                'avg_reward_0': rewards_sum[0] / count_sum,
                'avg_reward_1': rewards_sum[1] / count_sum,
                'avg_steps': steps_sum / count_sum,
                'games_sec': count_sum / time_since_last,
                'elapsed': time.time() - start_time
            }) + '\n')
            f_log.flush()
            last_print = time.time()
            steps_sum = 0
            rewards_sum = torch.zeros(2)
            count_sum = 0
        if time.time() - last_save >= 30.0:
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
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, help='optional')
    parser.add_argument('--term-entropy-reg', type=float, default=0.05)
    parser.add_argument('--proposal-entropy-reg', type=float, default=0.05)
    parser.add_argument('--disable-proposal', action='store_true')
    parser.add_argument('--disable-comms', action='store_true')
    parser.add_argument('--disable-prosocial', action='store_true')
    parser.add_argument('--enable-cuda', action='store_true')
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
