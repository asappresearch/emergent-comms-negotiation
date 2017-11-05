"""
Handle alive/dead masks
"""
import torch


class AliveSieve(object):
    def __init__(self, batch_size):
        """
        assume all alive to start with, with given batch_size
        """
        self.batch_size = batch_size
        self.alive_mask = torch.ByteTensor(batch_size).fill_(1)
        self.alive_idxes = self.mask_to_idxes(self.alive_mask)
        """
        out_idxes are the indexes of the current members of alivesieve, in the original batch
        basically, we popualte it initially with 0...batch_size-1 , and then
        sieve this too each time we call sieve_self
        """
        self.out_idxes = self.alive_idxes.clone()

    @staticmethod
    def mask_to_idxes(mask):
        return mask.nonzero().long().view(-1)

    def mark_dead(self, dead_mask):
        """
        sets the mask to 0 at relevant positions
        doesnt remove them yet, ie doesnt call 'sieve'
        """
        # print('dead_mask', dead_mask)
        # print('type(dead_mask)', type(dead_mask))
        # print('dead_mask.size()', dead_mask.size())
        if dead_mask.max() == 0:
            return
        dead_idxes = self.mask_to_idxes(dead_mask)
        self.alive_mask[dead_idxes] = 0
        self.alive_idxes = self.mask_to_idxes(self.alive_mask)

    def get_dead_idxes(self):
        dead_mask = 1 - self.alive_mask
        return dead_mask.nonzero().long().view(-1)

    def any_alive(self):
        return self.alive_mask.max() == 1

    def all_dead(self):
        return self.alive_mask.max() == 0

    def set_dead_global(self, target, v):
        """
        What this does is assign v to target, masked to only modify items marked as dead
        It assumes that target is a global tensor, ie hasnt been sieved
        So we're going to use out_idxes to index into this
        """
        dead_idxes = self.get_dead_idxes()
        # print('dead_idxes', dead_idxes)
        if len(dead_idxes) == 0:
            return
        target[self.out_idxes[dead_idxes]] = v

    def self_sieve_(self):
        """
        removes all dead from alive_mask
        basically equivalent to resetting the batch_size, and
        recreating an all-1s mask
        """
        self.out_idxes = self.out_idxes[self.alive_idxes]
        # print('self.out_idxes', self.out_idxes)

        self.batch_size = self.alive_mask.int().sum()
        self.alive_mask = torch.ByteTensor(self.batch_size).fill_(1)
        self.alive_idxes = self.mask_to_idxes(self.alive_mask)

    def sieve_tensor(self, t):
        """
        returns t sieved by current alive mask (ie anything not alive is
        not returned)
        note: the returned tensor is NOT guaranteed to be a reference into
        the old one. In fact, I'm fairly sure it will never in fact be (but
        this isnt guaranteed either; I havent checked...)
        """
        return t[self.alive_idxes]

    def sieve_list(self, alist):
        """
        returns alist sieved by current alive mask (ie anything not alive is
        not returned)
        """
        return [alist[b] for b in self.alive_idxes]
