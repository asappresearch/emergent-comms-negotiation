# "Emergent Communications through Negotiation"

Reproduce https://openreview.net/forum?id=Hk6WhagRW&amp;noteId=Hk6WhagRW , "Emergent Communication through Negotation", ICLR 2018 anonymous submission.

## To run

CPU:
```
python ecn.py --disable-comms
```

GPU:
```
python ecn.py --disable-comms --enable-cuda
```

Note that comms currently not yet implemented/working.

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
- each of the first 4 lines is the action of a single agent
- the `ACC` line is the agent accepting previous proposal
- each proposal line is laid out as:
```
  [utterance]   [utility 0]:[proposal 0]/[pool 0] ... etc ...
```
- if the agents run out of time, last line will be `[out of time]`
