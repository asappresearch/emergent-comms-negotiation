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
import alive_sieve


def render_action(t, agent, s, prop, term):
    speaker = 'A' if agent == 0 else 'B'
    utility = s.utilities[:, agent]
    print('  ', end='')
    if speaker == 'B':
        print('                                   ', end='')
    if term[0][0]:
        print(' ACC' )
    else:
        print(' ' + ''.join([str(v) for v in s.m_prev[0].view(-1).tolist()]), end='')
        print(' %s:%s/%s %s:%s/%s %s:%s/%s' % (
            utility[0][0], prop[0][0], s.pool[0][0],
            utility[0][1], prop[0][1], s.pool[0][1],
            utility[0][2], prop[0][2], s.pool[0][2],
        ), end='')
        print('')
        if t + 1 == s.N[0]:
            print('  [out of time]')


def save_model(model_file, agent_models, agent_opts, start_time, episode):
    state = {}
    for i in range(2):
        state['agent%s' % i] = {}
        state['agent%s' % i]['model_state'] = agent_models[i].state_dict()
        state['agent%s' % i]['opt_state'] = agent_opts[i].state_dict()
    state['episode'] = episode
    state['elapsed_time'] = time.time() - start_time
    with open(model_file, 'wb') as f:
        torch.save(state, f)


def load_model(model_file, agent_models, agent_opts):
    with open(model_file, 'rb') as f:
        state = torch.load(f)
    for i in range(2):
        agent_models[i].load_state_dict(state['agent%s' % i]['model_state'])
        agent_opts[i].load_state_dict(state['agent%s' % i]['opt_state'])
    episode = state['episode']
    # create a kind of 'virtual' start_time
    start_time = time.time() - state['elapsed_time']
    return episode, start_time


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


def calc_rewards(t, prosocial, s, term, agent):
    # calcualate rewards for any that just finished

    assert prosocial, 'not tested for not prosocial currently'

    batch_size = term.size()[0]
    utility = s.utilities[:, agent]
    type_constr = torch.cuda if s.pool.is_cuda else torch
    rewards_batch = type_constr.FloatTensor(batch_size, 2).fill_(0)
    if t == 0:
        # on first timestep theres no actual proposal yet, so score zero if terminate
        return rewards_batch

    reward_eligible_mask = term.view(batch_size).clone().byte()
    if reward_eligible_mask.max() == 0:
        # if none of them accepted proposal, by terminating
        return rewards_batch

    exceeded_pool, _ = ((s.last_proposal - s.pool) > 0).max(1)
    if exceeded_pool.max() > 0:
        reward_eligible_mask[exceeded_pool.nonzero().long().view(-1)] = 0
        if reward_eligible_mask.max() == 0:
            # all eligible ones exceeded pool
            return rewards_batch

    proposer = 1 - agent
    accepter = agent
    proposal = torch.zeros(batch_size, 2, 3).long()
    proposal[:, proposer] = s.last_proposal
    proposal[:, accepter] = s.pool - s.last_proposal
    max_utility, _ = s.utilities.max(1)

    reward_eligible_idxes = reward_eligible_mask.nonzero().long().view(-1)
    for b in reward_eligible_idxes:
        rewards = torch.FloatTensor(2).fill_(0)
        for i in range(2):
            rewards[i] = s.utilities[b, i].cpu().dot(proposal[b, i].cpu())

        if prosocial:
            total_actual_reward = rewards.sum()
            total_possible_reward = max_utility[b].cpu().dot(s.pool[b].cpu())
            scaled_reward = 0
            if total_possible_reward != 0:
                scaled_reward = total_actual_reward / total_possible_reward
            rewards.fill_(scaled_reward)
        else:
            for i in range(2):
                max_possible = s.utilities[b, i].cpu().dot(s.pool.cpu())
                if max_possible != 0:
                    rewards[i] /= max_possible

        # alive_games[b]['rewards'] = rewards
        rewards_batch[b] = rewards
    return rewards_batch


def run_episode(
        enable_cuda,
        enable_comms,
        enable_proposal,
        prosocial,
        agent_models,
        batch_size,
        render=False):

    type_constr = torch.cuda if enable_cuda else torch
    s = State(batch_size=batch_size)
    if enable_cuda:
        s.cuda()

    sieve = alive_sieve.AliveSieve(batch_size=batch_size, enable_cuda=enable_cuda)
    actions_by_timestep = []
    alive_masks = []

    # next two tensofrs wont be sieved, they will stay same size throughout
    # entire batch, we will update them using sieve.out_idxes[...]
    rewards = type_constr.FloatTensor(batch_size, 2).fill_(0)
    num_steps = type_constr.LongTensor(batch_size).fill_(10)

    entropy_loss_by_agent = [
        Variable(type_constr.FloatTensor(1).fill_(0)),
        Variable(type_constr.FloatTensor(1).fill_(0))
    ]
    if render:
        print('  ')
    for t in range(10):
        agent = 0 if t % 2 == 0 else 1

        agent_model = agent_models[agent]
        nodes, term_a, s.m_prev, this_proposal, _entropy_loss = agent_model(
            pool=Variable(s.pool),
            utility=Variable(s.utilities[:, agent]),
            m_prev=Variable(s.m_prev),
            prev_proposal=Variable(s.last_proposal)
        )
        entropy_loss_by_agent[agent] += _entropy_loss
        actions_by_timestep.append(nodes)

        if render and sieve.out_idxes[0] == 0:
            render_action(
                t=t,
                agent=agent,
                s=s,
                term=term_a,
                prop=this_proposal
            )

        new_rewards = calc_rewards(
            t=t,
            s=s,
            prosocial=prosocial,
            agent=agent,
            term=term_a
        )
        rewards[sieve.out_idxes] = new_rewards
        s.last_proposal = this_proposal

        sieve.mark_dead(term_a)
        sieve.mark_dead(t + 1 >= s.N)
        alive_masks.append(sieve.alive_mask.clone())
        sieve.set_dead_global(num_steps, t + 1)
        if sieve.all_dead():
            break

        s.sieve_(sieve.alive_idxes)
        sieve.self_sieve_()

    if render:
        print('  r: %.2f' % rewards[0].mean())
        print('  ')

    return actions_by_timestep, rewards, num_steps, alive_masks, entropy_loss_by_agent


def run(enable_proposal, enable_comms, seed, prosocial, logfile, model_file, batch_size,
        term_entropy_reg, utterance_entropy_reg, proposal_entropy_reg, enable_cuda):
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
            utterance_entropy_reg=utterance_entropy_reg,
            proposal_entropy_reg=proposal_entropy_reg
        )
        if enable_cuda:
            model = model.cuda()
        agent_models.append(model)
        agent_opts.append(optim.Adam(params=agent_models[i].parameters()))
    if path.isfile(model_file):
        episode, start_time = load_model(
            model_file=model_file,
            agent_models=agent_models,
            agent_opts=agent_opts)
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
        alive_rewards = rewards - baseline
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

        rewards_sum += rewards.sum(0).cpu()
        steps_sum += steps.sum()
        baseline = 0.7 * baseline + 0.3 * rewards.mean()
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
            save_model(
                model_file=model_file,
                agent_models=agent_models,
                agent_opts=agent_opts,
                start_time=start_time,
                episode=episode)
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
    parser.add_argument('--utterance-entropy-reg', type=float, default=0.001)
    parser.add_argument('--proposal-entropy-reg', type=float, default=0.05)
    parser.add_argument('--disable-proposal', action='store_true')
    parser.add_argument('--disable-comms', action='store_true')
    parser.add_argument('--disable-prosocial', action='store_true')
    parser.add_argument('--enable-cuda', action='store_true')
    parser.add_argument('--name', type=str, default='', help='used for logfile naming')
    parser.add_argument('--logfile', type=str, default='logs/log_%Y%m%d_%H%M%S{name}.log')
    args = parser.parse_args()
    args.enable_comms = not args.disable_comms
    args.enable_proposal = not args.disable_proposal
    args.prosocial = not args.disable_prosocial
    args.logfile = args.logfile.format(**args.__dict__)
    args.logfile = datetime.datetime.strftime(datetime.datetime.now(), args.logfile)
    del args.__dict__['disable_comms']
    del args.__dict__['disable_proposal']
    del args.__dict__['disable_prosocial']
    del args.__dict__['name']
    run(**args.__dict__)
