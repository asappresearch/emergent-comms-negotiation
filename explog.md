with context only, gets ~ 0.68, see [images/context-only.png], a6f3afc

adding in proposal, get ~0.73, after ~2500 rounds of 128 games, see [images/8b1caa8.png]

after adding utterance, and running for a while, d7d12843e77fd
```
  N=8
  pool=0,4,0
  util[0] 1,2,9
  util[1] 4,6,7
  A t=0.0 u=509771 p=0,0,0
  B t=1.0 u=832879 p=1,1,0
  steps=2 reward=1.00
episode 74270 avg rewards 0.74 0.74 b=0.74 games/sec 559.0 avg steps 2.0
saved model
  N=8
  pool=4,5,5
  util[0] 5,6,7
  util[1] 7,4,8
  A t=0.0 u=412343 p=0,0,0
  B t=1.0 u=933630 p=1,1,0
  steps=2 reward=0.90
episode 74287 avg rewards 0.73 0.73 b=0.72 games/sec 664.9 avg steps 2.0
  N=5
  pool=3,2,2
  util[0] 2,6,8
  util[1] 10,4,9
  A t=0.0 u=306570 p=0,0,0
  B t=1.0 u=933017 p=1,1,0
  steps=2 reward=0.93
episode 74304 avg rewards 0.73 0.73 b=0.74 games/sec 635.1 avg steps 2.0
saved model
  N=10
  pool=4,2,1
  util[0] 1,4,1
  util[1] 7,1,1
  A t=0.0 u=376059 p=0,0,0
  B t=1.0 u=565309 p=1,1,0
  steps=2 reward=0.84
episode 74317 avg rewards 0.73 0.73 b=0.74 games/sec 514.7 avg steps 2.0
saved model
  N=8
  pool=1,0,2
  util[0] 8,1,2
  util[1] 0,7,4
  A t=0.0 u=480752 p=0,0,0
  B t=1.0 u=133323 p=1,1,0
  steps=2 reward=0.50
episode 74332 avg rewards 0.73 0.73 b=0.72 games/sec 552.2 avg steps 2.0
  N=5
  pool=3,4,3
  util[0] 0,6,10
  util[1] 6,2,4
  A t=0.0 u=227836 p=0,0,0
  B t=1.0 u=735622 p=1,1,0
  steps=2 reward=0.53
episode 74345 avg rewards 0.73 0.73 b=0.73 games/sec 497.3 avg steps 2.0
saved model
```
=> seems to be in a local miminum ...

after ~150,000 batches of 128 games, still in local, 55de45b (see also [images/55de45b.png]) (saved as model model_saves/allnet_noentropy.dat):
```
episode 147066 avg rewards 0.74 0.74 b=0.73 games/sec 464.4 avg steps 2.0
saved model
  N=7
  pool=5,3,1
  util[0] 5,4,6
  util[1] 5,10,9
  A t=0.0 u=883613 p=0,0,0
  B t=1.0 u=884829 p=1,1,0
  steps=2 reward=1.00
episode 147080 avg rewards 0.73 0.73 b=0.72 games/sec 540.8 avg steps 2.0
  N=10
  pool=4,1,3
  util[0] 9,4,3
  util[1] 10,6,9
  A t=0.0 u=161003 p=0,0,0
  B t=1.0 u=556962 p=1,1,0
  steps=2 reward=1.00
episode 147093 avg rewards 0.72 0.72 b=0.73 games/sec 469.5 avg steps 2.0
saved model
  N=9
  pool=0,4,2
  util[0] 5,0,4
  util[1] 5,10,4
  A t=0.0 u=381030 p=0,0,0
  B t=1.0 u=752419 p=1,1,0
  steps=2 reward=1.00
episode 147105 avg rewards 0.72 0.72 b=0.73 games/sec 459.1 avg steps 2.0
saved model
  N=10
  pool=5,2,1
  util[0] 2,0,6
  util[1] 4,1,4
  A t=0.0 u=276362 p=0,0,0
  B t=1.0 u=327743 p=1,1,0
  steps=2 reward=0.93
episode 147117 avg rewards 0.72 0.72 b=0.72 games/sec 460.2 avg steps 2.0
```

fixed some bugs in lstm, because [seq] and [batch] dimensions were reversed/squished/mangled.

Still gets into a local maximum though, where `A` proposes 0,0,0; and `B` accepts....
=> probably needs the entropy regularization, to push it to try other options for 'term'

with term entropy regularization added and proposal entropy regularization added, 1f746b3, crashed:
```
episode 6091 avg rewards 0.74 0.74 b=0.73 games/sec 400 avg steps 2.00
  N=5
  pool=5,4,2
  util[0] 7,7,6
  util[1] 7,2,1
  A t=0.0 u=071278 p=0,0,0
  B t=1.0 u=981516 p=5,3,2
  steps=2 reward=0.60
episode 6102 avg rewards 0.73 0.73 b=0.72 games/sec 413 avg steps 2.00
  N=10
  pool=1,0,2
  util[0] 6,0,10
  util[1] 9,5,3
Traceback (most recent call last):
  File "ecn.py", line 574, in <module>
    run(**args.__dict__)
  File "ecn.py", line 472, in run
    render=render)
  File "ecn.py", line 305, in run_episode
    prev_proposal=Variable(last_proposal)
  File "/hugh/conda/lib/python3.6/site-packages/torch/nn/modules/module.py", line 224, in __call__
    result = self.forward(*input, **kwargs)
  File "ecn.py", line 246, in forward
    proposal_node, _entropy = proposal_policy(h_t)
  File "/hugh/conda/lib/python3.6/site-packages/torch/nn/modules/module.py", line 224, in __call__
    result = self.forward(*input, **kwargs)
  File "ecn.py", line 191, in forward
    out_node = torch.multinomial(x)
  File "/hugh/conda/lib/python3.6/site-packages/torch/autograd/variable.py", line 783, in multinomial
    return Multinomial(num_samples, replacement)(self)
  File "/hugh/conda/lib/python3.6/site-packages/torch/autograd/stochastic_function.py", line 23, in _do_forward
    result = super(StochasticFunction, self)._do_forward(*inputs)
  File "/hugh/conda/lib/python3.6/site-packages/torch/autograd/_functions/stochastic.py", line 16, in forward
    samples = probs.multinomial(self.num_samples, self.with_replacement)
RuntimeError: invalid argument 2: invalid multinomial distribution (sum of probabilities <= 0) at /opt/conda/conda-bld/pytorch_1503970438496/work/torch/lib/TH/generic/THTensorRandom.c:230
```
=> seems like the regularization pushed the weights to zero. I wonder how to solve that?

Also, games per second is much lower with current entropy regularization. I imagine because back-propping twice each time. I wonder if there is a better way of implementing the entropy regularization?

Fixed the entr reg speed issue somewhat, 653d3ad, from ~400 to ~570 games/sec, from the latest model save:
```
episode 6139 avg rewards 0.72 0.72 b=0.72 games/sec 565 avg steps 2.00
  N=9
  pool=1,0,2
  util[0] 9,0,9
  util[1] 5,5,7
  A t=0.0 u=164342 p=0,0,0
  B t=1.0 u=816912 p=0,4,2
  steps=2 reward=0.70
episode 6154 avg rewards 0.74 0.74 b=0.73 games/sec 577 avg steps 2.00
  N=4
  pool=2,5,3
  util[0] 0,1,4
  util[1] 6,2,6
```

invalid multinomial distr crashbug remains

fixed multinomial crashbug in 4d628ee65

turn to looking at the local minimum where first agent proposes all zeros, and second agent immediately accepts.

let's remove comms, to speed up training. trying 4599249 wiht `--disable-comms`, which runs at ~1000 games/second, using cpu.

after a bunch of epochs, using new cudarized version, still in local minimimum:
```
episode 156397 avg rewards 0.733 0.733 b=0.734 games/sec 2339 avg steps 0.0000
  N=7
  pool=3,5,5
  util[0] 2,9,4
  util[1] 4,5,0
  A t=0.0 u=000000 p=0,0,0
  B t=1.0 u=000000 p=3,4,1
  steps=2 reward=0.48
episode 156453 avg rewards 0.731 0.731 b=0.714 games/sec 2346 avg steps 0.0000
  N=7
  pool=0,1,5
  util[0] 6,8,7
  util[1] 0,0,9
  A t=0.0 u=000000 p=0,0,0
  B t=1.0 u=000000 p=3,4,1
  steps=2 reward=0.85
episode 156509 avg rewards 0.727 0.727 b=0.725 games/sec 2342 avg steps 0.0000
saved model
  N=9
  pool=2,0,1
  util[0] 8,9,5
  util[1] 10,6,0
  A t=0.0 u=000000 p=0,0,0
  B t=1.0 u=000000 p=3,4,1
  steps=2 reward=0.80
episode 156564 avg rewards 0.732 0.732 b=0.721 games/sec 2295 avg steps 0.0000
```

same for cpu version:
```
  N=5
  pool=4,1,2
  util[0] 3,3,10
  util[1] 4,3,2
  A t=0.0 u=000000 p=0,0,0
  B t=1.0 u=000000 p=2,4,3
  steps=2 reward=0.59
episode 126863 avg rewards 0.729 0.729 b=0.725 games/sec 1250 avg steps 2.0000
  N=6
  pool=2,5,2
  util[0] 4,1,3
  util[1] 7,4,9
  A t=0.0 u=000000 p=0,0,0
  B t=1.0 u=000000 p=2,3,3
  steps=2 reward=1.00
episode 126894 avg rewards 0.727 0.727 b=0.733 games/sec 1249 avg steps 2.0000
  N=10
  pool=5,2,0
  util[0] 2,6,0
  util[1] 1,0,5
  A t=0.0 u=000000 p=0,0,0
  B t=1.0 u=000000 p=2,3,3
  steps=2 reward=0.23
episode 126924 avg rewards 0.734 0.734 b=0.740 games/sec 1234 avg steps 2.0000
```

Not sure how to fix this. Increase entropy regularization term?

after a few more episodes, still stuck, cpu version:
```
  N=7
  pool=4,3,0
  util[0] 8,1,4
  util[1] 5,9,3
  A t=0.0 u=000000 p=0,0,0
  B t=1.0 u=000000 p=4,4,4
  steps=2 reward=0.80
episode 197938 avg rewards 0.721 0.721 b=0.717 games/sec 1271 avg steps 2.0000
```
gpu version:
```
  N=9
  pool=4,5,0
  util[0] 5,3,9
  util[1] 0,10,6
  A t=0.0 u=000000 p=0,0,0
  B t=1.0 u=000000 p=3,4,1
  steps=2 reward=0.71
episode 296182 avg rewards 0.726 0.726 b=0.733 games/sec 2309 avg steps 0.0000
```

=> lets try increasing regularization a bit, eg set both to 1.0 for now... or ... use .sum() instead of .mean() ?

Using .sum() instead of .mean() seems promising. using gpu version:
```
  N=7
  pool=0,4,0
  util[0] 9,7,10
  util[1] 8,4,10
  A t=0.0 u=000000 p=2,0,5
  B t=0.0 u=000000 p=5,2,5
  A t=0.0 u=000000 p=0,0,0
  B t=1.0 u=000000 p=1,5,1
  steps=4 reward=0.57
episode 4543 avg rewards 0.740 0.740 b=0.749 games/sec 791 avg steps 4.2285
```
