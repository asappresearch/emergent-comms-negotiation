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

after ~150,000 batches of 128 games, still in local, 55de45b (see also [images/55de45b.png]):
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
