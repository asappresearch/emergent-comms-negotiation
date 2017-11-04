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

import nets
import sampling


def render_action(t, agent, s, proposal_nodes, term_node):
    speaker = 'A' if agent == 0 else 'B'
    utility = s.utilities[:, agent]
    print('  ', end='')
    if speaker == 'B':
        print('                                   ', end='')
    if term_node.data[0][0]:
        print(' ACC' )
    else:
        print(' ' + ''.join([str(v) for v in s.m_prev[0].view(-1).tolist()]), end='')
        print(' %s:%s/%s %s:%s/%s %s:%s/%s' % (
            utility[0][0], proposal_nodes[0].data[0][0], s.pool[0][0],
            utility[0][1], proposal_nodes[1].data[0][0], s.pool[0][1],
            utility[0][2], proposal_nodes[2].data[0][0], s.pool[0][2],
        ), end='')
        print('')
        if t + 1 == s.N[0]:
            print('  [out of time]')


class State(object):
    def __init__(self, batch_size):
        self.N = sampling.sample_N(batch_size).int()
        self.pool = sampling.sample_items(batch_size)
        self.utilities = torch.zeros(batch_size, 2, 3).long()
        self.utilities[:, 0] = sampling.sample_utility(batch_size)
        self.utilities[:, 1] = sampling.sample_utility(batch_size)
        self.last_proposal = torch.zeros(batch_size, 3).long()
        self.m_prev = torch.zeros(batch_size, 6).long()

    def cuda(self):
        self.N = self.N.cuda()
        self.pool = self.pool.cuda()
        self.utilities = self.utilities.cuda()
        self.last_proposal = self.last_proposal.cuda()
        self.m_prev = self.m_prev.cuda()

    def sieve_(self, still_alive_idxes):
        self.N = self.N[still_alive_idxes]
        self.pool = self.pool[still_alive_idxes]
        self.utilities = self.utilities[still_alive_idxes]
        self.last_proposal = self.last_proposal[still_alive_idxes]
        self.m_prev = self.m_prev[still_alive_idxes]


def run_episode(
        enable_cuda,
        enable_comms,
        enable_proposal,
        prosocial,
        agent_models,
        batch_size,
        render=False):
    s = State(batch_size=batch_size)

    if enable_cuda:
        s.cuda()

    games = []
    actions_by_timestep = []
    alive_masks = []
    for b in range(batch_size):
        games.append({'rewards': [0, 0]})
    alive_games = games.copy()

    b_0_present = True  # is the first row of batch present? strictly for rendering purposes
    entropy_loss_by_agent = [Variable(torch.zeros(1)), Variable(torch.zeros(1))]
    if enable_cuda:
        entropy_loss_by_agent[0] = entropy_loss_by_agent[0].cuda()
        entropy_loss_by_agent[1] = entropy_loss_by_agent[1].cuda()
    if render:
        print('  ')
    for t in range(10):
        agent = 0 if t % 2 == 0 else 1
        batch_size = len(alive_games)
        utility = s.utilities[:, agent]

        c = torch.cat([s.pool, utility], 1)
        agent_model = agent_models[agent]
        term_node, utterance_nodes, proposal_nodes, _entropy_loss = agent_model(
            context=Variable(c),
            m_prev=Variable(s.m_prev),
            prev_proposal=Variable(s.last_proposal)
        )
        entropy_loss_by_agent[agent] += _entropy_loss
        if enable_comms:
            for i in range(6):
                s.m_prev[:, i] = utterance_nodes[i].data

        this_proposal = torch.zeros(batch_size, 3).long()
        if enable_cuda:
            this_proposal = this_proposal.cuda()
        for p in range(3):
            this_proposal[:, p] = proposal_nodes[p].data

        actions_t = []
        actions_t.append(term_node)
        if enable_comms:
            actions_t += utterance_nodes
        if enable_proposal:
            actions_t += proposal_nodes
        actions_by_timestep.append(actions_t)

        if render and b_0_present:
            render_action(
                t=t,
                agent=agent,
                s=s,
                term_node=term_node,
                proposal_nodes=proposal_nodes
            )

        # calcualate rewards for any that just finished
        reward_eligible_mask = term_node.data.view(batch_size).clone().byte()
        if t == 0:
            # on first timestep theres no actual proposal yet, so score zero if terminate
            reward_eligible_mask.fill_(0)
        if reward_eligible_mask.max() > 0:
            exceeded_pool, _ = ((s.last_proposal - s.pool) > 0).max(1)
            if exceeded_pool.max() > 0:
                reward_eligible_mask[exceeded_pool.nonzero().long().view(-1)] = 0

        if reward_eligible_mask.max() > 0:
            proposer = 1 - agent
            accepter = agent
            proposal = torch.zeros(batch_size, 2, 3).long()
            proposal[:, proposer] = s.last_proposal
            proposal[:, accepter] = s.pool - s.last_proposal
            max_utility, _ = s.utilities.max(1)

            reward_eligible_idxes = reward_eligible_mask.nonzero().long().view(-1)
            for b in reward_eligible_idxes:
                rewards = [0, 0]
                for i in range(2):
                    rewards[i] = s.utilities[b, i].cpu().dot(proposal[b, i].cpu())

                if prosocial:
                    total_actual_reward = np.sum(rewards)
                    total_possible_reward = max_utility[b].cpu().dot(s.pool[b].cpu())
                    scaled_reward = 0
                    if total_possible_reward != 0:
                        scaled_reward = total_actual_reward / total_possible_reward
                    rewards = [scaled_reward, scaled_reward]
                else:
                    for i in range(2):
                        max_possible = s.utilities[b, i].cpu().dot(s.pool.cpu())
                        if max_possible != 0:
                            rewards[i] /= max_possible

                alive_games[b]['rewards'] = rewards

        still_alive_mask = 1 - term_node.data.view(batch_size).clone().byte()
        # to think about off-by-one stuff, so let's say N is 3
        # and t is 2, then we should finish
        # so conditions is t + 1 >= N
        finished_N = t + 1 >= s.N
        if enable_cuda:
            finished_N = finished_N.cuda()
        still_alive_mask[finished_N] = 0
        alive_masks.append(still_alive_mask)

        dead_idxes = (1 - still_alive_mask).nonzero().long().view(-1)
        for b in dead_idxes:
            alive_games[b]['steps'] = t + 1

        if still_alive_mask.max() == 0:
            break
        if still_alive_mask[0] == 0:
            b_0_present = False

        still_alive_idxes = still_alive_mask.nonzero().long().view(-1)
        if enable_cuda:
            still_alive_idxes = still_alive_idxes.cuda()
        # filter the state through the still alive mask:
        s.last_proposal = this_proposal
        s.sieve_(still_alive_idxes)
        alive_games = [alive_games[b] for b in still_alive_idxes]

    for g in games:
        if 'steps' not in g:
            g['steps'] = 10

    if render:
        print('  r: %.2f' % np.mean(g['rewards']))
        print('  ')

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
        model = nets.AgentModel(
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
        nodes_by_agent = [[], []]
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
        for t in range(T):
            _batch_size = alive_rewards.size()[0]
            agent = 0 if t % 2 == 0 else 1
            if len(actions[t]) > 0:
                for action in actions[t]:
                    action.reinforce(alive_rewards[:, agent].contiguous().view(_batch_size, 1))
            nodes_by_agent[agent] += actions[t]
            mask = alive_masks[t]
            if mask.max() == 0:
                break
            alive_rewards = alive_rewards[mask.nonzero().long().view(-1)]
        for i in range(2):
            if len(nodes_by_agent[i]) > 0:
                autograd.backward([entropy_loss_by_agent[i]] + nodes_by_agent[i], [None] + len(nodes_by_agent[i]) * [None])
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
