import torch
import numpy as np
import ecn


# def calc_rewards(t, prosocial, s, term, agent, alive_games):
#     # calcualate rewards for any that just finished

#     batch_size = term.size()[0]
#     utility = s.utilities[:, agent]
#     if t == 0:
#         # on first timestep theres no actual proposal yet, so score zero if terminate
#         return

#     reward_eligible_mask = term.view(batch_size).clone().byte()
#     if reward_eligible_mask.max() == 0:
#         # if none of them accepted proposal, by terminating
#         return

#     exceeded_pool, _ = ((s.last_proposal - s.pool) > 0).max(1)
#     if exceeded_pool.max() > 0:
#         reward_eligible_mask[exceeded_pool.nonzero().long().view(-1)] = 0
#         if reward_eligible_mask.max() == 0:
#             # all eligible ones exceeded pool
#             return

#     proposer = 1 - agent
#     accepter = agent
#     proposal = torch.zeros(batch_size, 2, 3).long()
#     proposal[:, proposer] = s.last_proposal
#     proposal[:, accepter] = s.pool - s.last_proposal
#     max_utility, _ = s.utilities.max(1)

#     reward_eligible_idxes = reward_eligible_mask.nonzero().long().view(-1)
#     for b in reward_eligible_idxes:
#         rewards = [0, 0]
#         for i in range(2):
#             rewards[i] = s.utilities[b, i].cpu().dot(proposal[b, i].cpu())

#         if prosocial:
#             total_actual_reward = np.sum(rewards)
#             total_possible_reward = max_utility[b].cpu().dot(s.pool[b].cpu())
#             scaled_reward = 0
#             if total_possible_reward != 0:
#                 scaled_reward = total_actual_reward / total_possible_reward
#             rewards = [scaled_reward, scaled_reward]
#         else:
#             for i in range(2):
#                 max_possible = s.utilities[b, i].cpu().dot(s.pool.cpu())
#                 if max_possible != 0:
#                     rewards[i] /= max_possible

#         alive_games[b]['rewards'] = rewards

# class State(object):
#     def __init__(self, batch_size):
#         self.N = sampling.sample_N(batch_size).int()
#         self.pool = sampling.sample_items(batch_size)
#         self.utilities = torch.zeros(batch_size, 2, 3).long()
#         self.utilities[:, 0] = sampling.sample_utility(batch_size)
#         self.utilities[:, 1] = sampling.sample_utility(batch_size)
#         self.last_proposal = torch.zeros(batch_size, 3).long()
#         self.m_prev = torch.zeros(batch_size, 6).long()


def test_rewards_t0():
    t = 0
    prosocial = True
    batch_size = 128
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(batch_size=batch_size)
    agent = 0 if t  % 2 == 0 else 1
    term = torch.from_numpy(np.random.choice(2, batch_size)).long()
    alive_games = []
    for b in range(batch_size):
        alive_games.append({'rewards': [0,0]})
    ecn.calc_rewards(s=s, t=t, prosocial=prosocial, alive_games=alive_games, agent=agent, term=term)
    # print('alive_games', alive_games)
    for game in alive_games:
        assert game['rewards'] == [0, 0]


def test_rewards_t1():
    t = 1
    prosocial = True
    batch_size = 97
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(batch_size=batch_size)
    agent = 0 if t  % 2 == 0 else 1
    term = torch.from_numpy(np.random.choice(2, batch_size)).long()
    alive_games = []
    for b in range(batch_size):
        alive_games.append({'rewards': [0,0]})
    ecn.calc_rewards(s=s, t=t, prosocial=prosocial, alive_games=alive_games, agent=agent, term=term)
    print('alive_games', alive_games)
    for b in range(batch_size):
        game = alive_games[b]
        assert game['rewards'] == [0, 0] or term[b] == 1
    # for game in alive_games:
        # assert game['rewards'] == [0, 0]


def test_single_game_noterm():
    t = 1
    prosocial = True
    batch_size = 1
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(batch_size=batch_size)
    agent = 0 if t  % 2 == 0 else 1
    term = torch.ByteTensor([0])
    # print('term', term)
    alive_games = [{'rewards': [0, 0]}]
    ecn.calc_rewards(s=s, t=t, prosocial=prosocial, alive_games=alive_games, agent=agent, term=term)
    # print('alive_games', alive_games)
    assert alive_games[0]['rewards'] == [0, 0]


def test_single_game_term_ideal():
    t = 1
    prosocial = True
    batch_size = 1
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(batch_size=batch_size)

    s.pool = torch.LongTensor([[3, 7, 2]])
    s.utilities = torch.LongTensor([[[5,4,3], [3,4,5]]])
    # last proposal means agent 0's, and we are now on agent 1, who is accepintg it
    s.last_proposal = torch.LongTensor([[3, 7, 0]])

    agent = 0 if t  % 2 == 0 else 1
    term = torch.ByteTensor([1])
    # print('term', term)
    alive_games = [{'rewards': [0, 0]}]
    ecn.calc_rewards(s=s, t=t, prosocial=prosocial, alive_games=alive_games, agent=agent, term=term)
    assert alive_games[0]['rewards'] == [1.0, 1.0]


def test_single_game_term_ideal2():
    t = 1
    prosocial = True
    batch_size = 1
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(batch_size=batch_size)

    s.pool = torch.LongTensor([[3, 7, 2]])
    s.utilities = torch.LongTensor([[[5,4,3], [3,4,5]]])
    # last proposal means agent 0's, and we are now on agent 1, who is accepintg it
    s.last_proposal = torch.LongTensor([[3, 0, 0]])

    agent = 0 if t  % 2 == 0 else 1
    term = torch.ByteTensor([1])
    # print('term', term)
    alive_games = [{'rewards': [0, 0]}]
    ecn.calc_rewards(s=s, t=t, prosocial=prosocial, alive_games=alive_games, agent=agent, term=term)
    assert alive_games[0]['rewards'] == [1.0, 1.0]


def test_single_game_term_ideal3():
    t = 1
    prosocial = True
    batch_size = 1
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(batch_size=batch_size)

    s.pool = torch.LongTensor([[3, 7, 2]])
    s.utilities = torch.LongTensor([[[5,4,3], [3,4,5]]])
    # last proposal means agent 0's, and we are now on agent 1, who is accepintg it
    s.last_proposal = torch.LongTensor([[3, 2, 0]])

    agent = 0 if t  % 2 == 0 else 1
    term = torch.ByteTensor([1])
    # print('term', term)
    alive_games = [{'rewards': [0, 0]}]
    ecn.calc_rewards(s=s, t=t, prosocial=prosocial, alive_games=alive_games, agent=agent, term=term)
    assert alive_games[0]['rewards'] == [1.0, 1.0]


def test_single_game_term_nonideal1():
    t = 1
    prosocial = True
    batch_size = 1
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(batch_size=batch_size)

    s.pool = torch.LongTensor([[3, 7, 2]])
    s.utilities = torch.LongTensor([[[5,4,3], [3,4,5]]])
    # last proposal means agent 0's, and we are now on agent 1, who is accepintg it
    s.last_proposal = torch.LongTensor([[0, 2, 0]])

    total_available = 3 * 5 + 7 * 4 + 2 * 5
    print('total_available', total_available)
    actual = 3 * 3 + 7 * 4 + 5 * 2
    print('actual', actual)
    ratio = actual / total_available
    print('ratio', ratio)

    agent = 0 if t  % 2 == 0 else 1
    term = torch.ByteTensor([1])
    # print('term', term)
    alive_games = [{'rewards': [0, 0]}]
    ecn.calc_rewards(s=s, t=t, prosocial=prosocial, alive_games=alive_games, agent=agent, term=term)
    assert alive_games[0]['rewards'] == [ratio, ratio]


def test_single_game_term_exceeds_pool():
    t = 1
    prosocial = True
    batch_size = 1
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(batch_size=batch_size)

    s.pool = torch.LongTensor([[3, 7, 2]])
    s.utilities = torch.LongTensor([[[5,4,3], [3,4,5]]])
    # last proposal means agent 0's, and we are now on agent 1, who is accepintg it
    s.last_proposal = torch.LongTensor([[0, 2, 3]])

    agent = 0 if t  % 2 == 0 else 1
    term = torch.ByteTensor([1])
    # print('term', term)
    alive_games = [{'rewards': [0.0, 0.0]}]
    ecn.calc_rewards(s=s, t=t, prosocial=prosocial, alive_games=alive_games, agent=agent, term=term)
    assert alive_games[0]['rewards'] == [0.0, 0.0]


def test_single_game_term_exceeds_withinpool():
    t = 1
    prosocial = True
    batch_size = 1
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(batch_size=batch_size)

    s.pool = torch.LongTensor([[3, 7, 2]])
    s.utilities = torch.LongTensor([[[5,4,3], [3,4,5]]])
    # last proposal means agent 0's, and we are now on agent 1, who is accepintg it
    s.last_proposal = torch.LongTensor([[0, 0, 0]])

    total_available = 3 * 5 + 7 * 4 + 2 * 5
    print('total_available', total_available)
    actual = 3 * 3 + 7 * 4 + 2 * 5
    print('actual', actual)
    ratio = actual / total_available
    print('ratio', ratio)

    agent = 0 if t  % 2 == 0 else 1
    term = torch.ByteTensor([1])
    # print('term', term)
    alive_games = [{'rewards': [0.0, 0.0]}]
    ecn.calc_rewards(s=s, t=t, prosocial=prosocial, alive_games=alive_games, agent=agent, term=term)
    assert alive_games[0]['rewards'] == [ratio, ratio]


def test_single_game_term_exceeds_withinpool2():
    t = 1
    prosocial = True
    batch_size = 1
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(batch_size=batch_size)

    s.pool = torch.LongTensor([[3, 7, 2]])
    s.utilities = torch.LongTensor([[[5,4,3], [3,4,5]]])
    # last proposal means agent 0's, and we are now on agent 1, who is accepintg it
    s.last_proposal = torch.LongTensor([[3, 7, 2]])

    total_available = 3 * 5 + 7 * 4 + 2 * 5
    print('total_available', total_available)
    actual = 3 * 5 + 7 * 4 + 2 * 3
    print('actual', actual)
    ratio = actual / total_available
    print('ratio', ratio)

    agent = 0 if t  % 2 == 0 else 1
    term = torch.ByteTensor([1])
    # print('term', term)
    alive_games = [{'rewards': [0.0, 0.0]}]
    ecn.calc_rewards(s=s, t=t, prosocial=prosocial, alive_games=alive_games, agent=agent, term=term)
    assert alive_games[0]['rewards'] == [ratio, ratio]
