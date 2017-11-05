import torch
import numpy as np
import ecn


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


def test_single_game_noterm():
    t = 1
    prosocial = True
    batch_size = 1
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(batch_size=batch_size)
    agent = 0 if t  % 2 == 0 else 1
    term = torch.ByteTensor([0])
    alive_games = [{'rewards': [0, 0]}]
    ecn.calc_rewards(s=s, t=t, prosocial=prosocial, alive_games=alive_games, agent=agent, term=term)
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
    alive_games = [{'rewards': [0.0, 0.0]}]
    ecn.calc_rewards(s=s, t=t, prosocial=prosocial, alive_games=alive_games, agent=agent, term=term)
    assert alive_games[0]['rewards'] == [ratio, ratio]


def test_single_game_term_t2():
    t = 2
    prosocial = True
    batch_size = 1
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(batch_size=batch_size)

    s.pool = torch.LongTensor([[3, 7, 2]])
    s.utilities = torch.LongTensor([[[5,4,3], [3,4,5]]])
    s.last_proposal = torch.LongTensor([[3, 0, 0]])

    total_available = 3 * 5 + 7 * 4 + 2 * 5
    print('total_available', total_available)
    # so, the  proposer is the second agent, ie agent 1
    # so, the proposer, agent 1, will take: 3 0 0
    # accepter, agent 0, will take 0 7 2
    actual = 0 * 5 + 7 * 4 + 2 * 3 + \
            3 * 3
    print('actual', actual)
    ratio = actual / total_available
    print('ratio', ratio)

    agent = 0 if t  % 2 == 0 else 1
    term = torch.ByteTensor([1])
    alive_games = [{'rewards': [0, 0]}]
    ecn.calc_rewards(s=s, t=t, prosocial=prosocial, alive_games=alive_games, agent=agent, term=term)
    assert alive_games[0]['rewards'] == [ratio, ratio]


def test_single_game_term_t2_batch3():
    t = 2
    prosocial = True
    batch_size = 3
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(batch_size=batch_size)

    s.pool = torch.from_numpy(np.random.choice(10, (batch_size, 3))).long()
    s.pool[1] = torch.LongTensor([3, 7, 2])

    s.utilities = torch.from_numpy(np.random.choice(10, (batch_size, 2, 3))).long()
    s.utilities[1] = torch.LongTensor([[5,4,3], [3,4,5]])

    s.last_proposal = torch.from_numpy(np.random.choice(10, (batch_size, 3))).long()
    s.last_proposal[1] = torch.LongTensor([3, 0, 0])

    term = torch.ByteTensor([0, 1, 0])
    # since only one terminated, reward should be for simply the hard-coded ones above
    # all others should be zero

    total_available = 3 * 5 + 7 * 4 + 2 * 5
    print('total_available', total_available)
    # so, the  proposer is the second agent, ie agent 1
    # so, the proposer, agent 1, will take: 3 0 0
    # accepter, agent 0, will take 0 7 2
    actual = 0 * 5 + 7 * 4 + 2 * 3 + \
            3 * 3
    print('actual', actual)
    ratio = actual / total_available
    print('ratio', ratio)

    agent = 0 if t  % 2 == 0 else 1
    alive_games = []
    for b in range(batch_size):
        alive_games.append({'rewards': [0, 0]})
    ecn.calc_rewards(s=s, t=t, prosocial=prosocial, alive_games=alive_games, agent=agent, term=term)
    assert alive_games[0]['rewards'] == [0.0, 0.0]
    assert alive_games[1]['rewards'] == [ratio, ratio]
    assert alive_games[2]['rewards'] == [0.0, 0.0]
