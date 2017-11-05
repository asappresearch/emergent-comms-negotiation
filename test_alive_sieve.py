import torch
import alive_sieve


def test_alive_sieve_init():
    batch_size = 12
    sieve = alive_sieve.AliveSieve(batch_size, enable_cuda=False)

    assert len(sieve.alive_mask) == batch_size
    assert len(sieve.alive_idxes) == batch_size
    for b in range(batch_size):
        assert sieve.alive_mask[b] == 1
        assert sieve.alive_idxes[b] == b


def test_alive_sieve_mark_dead():
    batch_size = 12
    sieve = alive_sieve.AliveSieve(batch_size, enable_cuda=False)

    dead_mask = torch.ByteTensor([0,1,0,1, 0,0,0,0, 1,1,1,1])
    sieve.mark_dead(dead_mask)

    assert len(sieve.alive_mask) == 12
    assert len(sieve.alive_idxes) == 6
    assert (sieve.alive_mask != torch.ByteTensor([1,0,1,0, 1,1,1,1, 0,0,0,0])).max() == 0
    assert (sieve.alive_idxes != torch.LongTensor([0,2, 4,5,6,7])).max() == 0


def test_alive_sieve_sieve_self():
    batch_size = 12
    sieve = alive_sieve.AliveSieve(batch_size, enable_cuda=False)

    dead_mask = torch.ByteTensor([0,1,0,1, 0,0,0,0, 1,1,1,1])
    sieve.mark_dead(dead_mask)

    assert len(sieve.alive_mask) == 12
    assert len(sieve.alive_idxes) == 6
    assert (sieve.alive_mask != torch.ByteTensor([1,0,1,0, 1,1,1,1, 0,0,0,0])).max() == 0
    assert (sieve.alive_idxes != torch.LongTensor([0,2, 4,5,6,7])).max() == 0

    assert (sieve.out_idxes != torch.LongTensor([0,1,2,3,4,5,6,7,8,9,10,11])).max() == 0
    assert sieve.batch_size == 12

    sieve.self_sieve_()
    assert (sieve.out_idxes != torch.LongTensor([0,2,4,5,6,7])).max() == 0
    assert sieve.batch_size == 6

    # sieve again...
    sieve.mark_dead(torch.ByteTensor([0,1,1,0,0,1]))
    assert (sieve.out_idxes != torch.LongTensor([0,2,4,5,6,7])).max() == 0
    sieve.self_sieve_()
    assert (sieve.out_idxes != torch.LongTensor([0,5,6])).max() == 0


def test_set_dead_global():
    batch_size = 12
    sieve = alive_sieve.AliveSieve(batch_size, enable_cuda=False)

    dead_mask = torch.ByteTensor([0,1,0,1, 0,0,0,0, 1,1,1,1])
    # so alive will be            1   1    1 1 1 1
    sieve.mark_dead(dead_mask)

    sieve.self_sieve_()

    sieve.mark_dead(torch.ByteTensor([0,1,1,0,0,1]))

    target = torch.rand(batch_size, 3)
    target_orig = target.clone()
    new_v = torch.rand(3, 3)
    sieve.set_dead_global(target, new_v)

    assert target[2][0] == new_v[0][0]
    assert target[4][0] == new_v[1][0]
    assert target[7][0] == new_v[2][0]
