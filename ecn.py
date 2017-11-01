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
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=embedding_size,
            num_layers=1)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.embedding(x)
        x = x.view(-1, batch_size, self.embedding_size)
        state = (
            Variable(torch.zeros(1, batch_size, self.embedding_size)),
            Variable(torch.zeros(1, batch_size, self.embedding_size))
        )
        x, state = self.lstm(x, state)
        return state[0].view(batch_size, self.embedding_size)


class UtteranceNet(nn.Module):
    def __init__(self, embedding_size=100):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(11, embedding_size)
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=embedding_size,
            num_layers=1)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.embedding(x)
        x = x.view(-1, batch_size, self.embedding_size)
        state = (
            Variable(torch.zeros(1, 1, self.embedding_size)),
            Variable(torch.zeros(1, 1, self.embedding_size)))
        x, state = self.lstm(x, state)
        return state[0].view(batch_size, self.embedding_size)


class ProposalNet(nn.Module):
    def __init__(self, embedding_size=100):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(11, embedding_size)
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=embedding_size,
            num_layers=1)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.embedding(x)
        x = x.view(-1, batch_size, self.embedding_size)
        state = (
            Variable(torch.zeros(1, batch_size, self.embedding_size)),
            Variable(torch.zeros(1, batch_size, self.embedding_size)))
        x, state = self.lstm(x, state)
        return state[0].view(batch_size, self.embedding_size)


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
        # print('x[0]', x[0])
        out_node = torch.bernoulli(x)
        return out_node


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

    def forward(self, x):
        x = self.h1(x)
        x = F.softmax(x)
        out_node = torch.multinomial(x)
        return out_node


class AgentModel(nn.Module):
    def __init__(self, enable_comms, enable_proposal, embedding_size=100):
        super().__init__()
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

    def forward(self, context, m_prev, p_prev):
        batch_size = context.size()[0]
        # print('batch_size', batch_size)
        # c_h = self.context_net(context)
        # m_h = self.utterance_net(m_prev)
        # p_h = self.proposal_net(p_prev)

        # h_t = torch.cat([c_h, m_h, p_h], -1)
        h_t = Variable(torch.zeros(batch_size, self.embedding_size * 3))
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
        batch_size,
        render=False):
    # batch_size = 5
    # print('WARNIGN DEBUG CODE PRESENT')

    # following take not much memory, not fluffed up yet:
    N = sample_N(batch_size).int()
    pool = sample_items(batch_size)
    utilities = torch.zeros(batch_size, 2, 3).long()
    utilities[:, 0] = sample_utility(batch_size)
    utilities[:, 1] = sample_utility(batch_size)
    last_proposal = torch.zeros(batch_size, 3).long()
    m_prev = torch.zeros(batch_size, 6).long()
    p_prev = torch.zeros(batch_size, 3).long()

    games = []
    actions_by_timestep = []
    alive_masks = []
    for b in range(batch_size):
        games.append({'rewards': [0, 0], 'num_steps': 0})
    alive_games = games.copy()

    if render:
        print('  N[0]=%s' % N[0], end='')
        print(' pool[0]: %s,%s,%s' % (pool[0][0], pool[0][1], pool[0][2]), end='')
        print(' util[0]:', end='')
        for i in range(2):
            print(' %s,%s,%s' % (utilities[0][i][0], utilities[0][i][1], utilities[0][i][2]), end='')
        print('')
    b_0_present = True
    for t in range(10):
        agent = 0 if t % 2 == 0 else 1
        batch_size = len(alive_games)
        if render:
            print('batch_size', batch_size)
        utility = utilities[:, agent]

        c = torch.cat([pool, utility], 1)
        agent_model = agent_models[agent]
        term_node, utterance_nodes, proposal_nodes = agent_model(
            context=Variable(c),
            m_prev=Variable(m_prev),
            p_prev=Variable(p_prev)
        )

        actions_t = []
        actions_t.append(term_node)
        actions_t += utterance_nodes
        actions_t += proposal_nodes
        # if render:
        if render and b_0_present:
            print('  t %s term=%s' % (t, term_node.data[0][0]), end='')
            print(' thisprop %s,%s,%s' % (
                proposal_nodes[0].data[0][0],
                proposal_nodes[1].data[0][0],
                proposal_nodes[2].data[0][0]
            ))
        actions_by_timestep.append(actions_t)

        # calcualate rewards for any that just finished
        reward_eligible_mask = term_node.data.view(batch_size).clone().byte()
        if t == 0:
            # on first timestep theres no actual proposal yet, so score zero if terminate
            reward_eligible_mask.fill_(0)
        reward_eligible_idxes = reward_eligible_mask.nonzero().long().view(-1)
        if reward_eligible_mask.max() > 0:
            # things we need to do:
            # - eliminate any that provided invalid proposals (exceeded pool)
            # - calculate score for each agent
            # - calculcate max score for each agent
            # - normalize score
            exceeded_pool, _ = ((last_proposal - pool) > 0).max(1)
            if exceeded_pool.max() > 0:
                reward_eligible_mask[exceeded_pool.nonzero().long().view(-1)] = 0

        reward_eligible_idxes = reward_eligible_mask.nonzero().long().view(-1)
        if reward_eligible_mask.max() > 0:
            proposer = 1 - agent
            accepter = agent
            proposal = torch.zeros(batch_size, 2, 3).long()
            proposal[:, proposer] = last_proposal
            proposal[:, accepter] = pool - last_proposal
            max_utility, _ = utilities.max(1)

            for b in reward_eligible_idxes:
                rewards = [0, 0]
                for i in range(2):
                    rewards[i] = utilities[b, i].dot(proposal[b, i])
                if render and b_0_present and b == 0:
                    print('rewards', rewards)

                if prosocial:
                    total_actual_reward = np.sum(rewards)
                    total_possible_reward = max_utility[b].dot(pool[b])
                    scaled_reward = 0
                    if total_possible_reward != 0:
                        scaled_reward = total_actual_reward / total_possible_reward
                    rewards = [scaled_reward, scaled_reward]
                    if render and b_0_present and b == 0:
                        print('tot act %.1f tot pos %.1f scal %.1f' % (total_actual_reward, total_possible_reward, scaled_reward))
                else:
                    for i in range(2):
                        max_possible = utilities[b, i].dot(pool)
                        if max_possible != 0:
                            rewards[i] /= max_possible

                alive_games[b]['rewards'] = rewards
                if render and b_0_present and b == 0:
                    print('  rewards', rewards)

        if render and b_0_present:
            print('  term[0]', term_node.data.view(batch_size)[0])
        still_alive_mask = 1 - term_node.data.view(batch_size).clone().byte()
        finished_N = t >= N
        still_alive_mask[finished_N] = 0
        alive_masks.append(still_alive_mask)

        dead_idxes = (1 - still_alive_mask).nonzero().long().view(-1)
        for b in dead_idxes:
            alive_games[b]['steps'] = t + 1

        if still_alive_mask.max() == 0:
            break

        this_proposal = torch.LongTensor(batch_size, 3)
        for p in range(3):
            this_proposal[:, p] = proposal_nodes[p].data
        last_proposal = this_proposal

        # filter the state through the still alive mask:
        still_alive_idxes = still_alive_mask.nonzero().long().view(-1)
        pool = pool[still_alive_idxes]
        last_proposal = last_proposal[still_alive_idxes]
        utilities = utilities[still_alive_idxes]
        m_prev = m_prev[still_alive_idxes]
        p_prev = p_prev[still_alive_idxes]
        N = N[still_alive_idxes]
        if still_alive_mask[0] == 0:
            b_0_present = False

        new_alive_games = []
        for i in still_alive_idxes:
            new_alive_games.append(alive_games[i])
        alive_games = new_alive_games

    if render:
        print('  num steps ', games[0]['steps'], 'rewards:', games[0]['rewards'])
    return actions_by_timestep, [g['rewards'] for g in games], alive_masks


def run(enable_proposal, enable_comms, seed, prosocial, logfile, model_file, batch_size):
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
        actions, rewards, alive_masks = run_episode(
            agent_models=agent_models,
            prosocial=prosocial,
            batch_size=batch_size,
            render=render)

        for i in range(2):
            agent_opts[i].zero_grad()
        nodes_by_agent = [[], []]
        alive_rewards = torch.zeros(batch_size, 2)
        all_rewards = torch.zeros(batch_size, 2)
        for i in range(2):
            # note to self: just use .clone() or something...
            all_rewards[:, i] = torch.FloatTensor([r[i] for r in rewards])
            alive_rewards[:, i] = torch.FloatTensor([r[i] for r in rewards])
        alive_rewards -= baseline
        T = len(rewards)
        for t in range(T):
            batch_size = alive_rewards.size()[0]
            agent = 0 if t % 2 == 0 else 1
            for action in actions[t]:
                action.reinforce(alive_rewards[:, agent].contiguous().view(batch_size, 1))
            nodes_by_agent[agent] += actions[t]
            mask = alive_masks[t]
            # if enable_cuda:
            #     mask = mask.cuda()
            if mask.max() == 0:
                break
            alive_rewards = alive_rewards[mask.nonzero().long().view(-1)]
        for i in range(2):
            if len(nodes_by_agent[i]) > 0:
                autograd.backward(nodes_by_agent[i], len(nodes_by_agent[i]) * [None])
                agent_opts[i].step()

        rewards_sum += all_rewards.mean(0)
        baseline = 0.7 * baseline + 0.3 * all_rewards.mean()

        count_sum += batch_size
        if render:
            print('episode %s avg rewards %.2f %.2f b=%.2f' % (
                episode, rewards_sum[0] / count_sum, rewards_sum[1] / count_sum, baseline))
            f_log.write(json.dumps({
                'episode': episode,
                'avg_reward_0': rewards_sum[0] / count_sum,
                'avg_reward_1': rewards_sum[1] / count_sum,
                'elapsed': time.time() - start_time
            }) + '\n')
            f_log.flush()
            last_print = time.time()
            rewards_sum = torch.zeros(2)
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

        # asdfd
        episode += 1
    f_log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, default='model_saves/model.dat')
    parser.add_argument('--batch-size', type=int, default=128)
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
