import torch


def calc_rewards(t, prosocial, s, term):
    # calcualate rewards for any that just finished

    assert prosocial, 'not tested for not prosocial currently'

    agent = t % 2
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
