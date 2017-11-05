# "Emergent Communication through Negotiation"

Reproduce https://openreview.net/forum?id=Hk6WhagRW&amp;noteId=Hk6WhagRW , "Emergent Communication through Negotation", ICLR 2018 anonymous submission.

## To install

- install pytorch 0.2, https://pytorch.org
- download this repo, `git clone https://github.com/asappinc/emergent_comms_negotiation`

## To run

CPU:
```
python ecn.py [--disable-comms]
```

GPU:
```
python ecn.py [--disable-comms] --enable-cuda
```

~~Note that comms currently not yet implemented/working.~~

## Stdout layout

eg if we have:
```
   000000 4:4/0 7:5/5 9:4/4
                                      000000 4:5/0 6:1/5 7:2/4
   000000 4:0/0 7:0/5 9:1/4
                                      ACC
  r: 0.91
```

Then:
- each of the first 4 lines here is the action of a single agent
- the `ACC` line is the agent accepting previous proposal
- each proposal line is laid out as:
```
  [utterance]   [utility 0]:[proposal 0]/[pool 0] ... etc ...
```
- if the agents run out of time, last line will be `[out of time]`

One negotation is printed out every 3 seconds or so, using the training set; the other negotations are executed silently.  There is no test set for now.

## Results so far

### proposal, no comms, prosocial

<img src="images/20171104_144936_proposal_social_nocomms.png?raw=true" width="800" />

### proposal, comms, prosocial

<img src="images/20171104_192343gpu2_proposal_social_comms_b.png?raw=true" width="800" />
