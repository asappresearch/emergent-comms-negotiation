test-context:

effect of entropy reg:
```
(root) ~/git/emergent_comms_negotiation (master|…2△1) $ python nets_test.py test-context --term-entropy-reg 1
episode 0 num_right 2 baseline 0.3
episode 100 num_right 2 baseline 0.6818555706423856
episode 200 num_right 2 baseline 0.9938958233414137
episode 300 num_right 2 baseline 0.7746209980485705
episode 400 num_right 2 baseline 0.9444644898336487
episode 500 num_right 1 baseline 0.545179411787141
episode 600 num_right 2 baseline 0.8696337750631387
episode 700 num_right 2 baseline 0.8364079471631221
episode 800 num_right 2 baseline 0.9973736089014604
episode 900 num_right 2 baseline 0.9870381382361351
episode 1000 num_right 2 baseline 0.9552785890972033
episode 1100 num_right 2 baseline 0.946063538655197
episode 1200 num_right 2 baseline 0.9990292805932697
episode 1300 num_right 2 baseline 0.9594112885125148
episode 1400 num_right 1 baseline 0.7311748112691385
episode 1500 num_right 2 baseline 0.9609664419399175
episode 1600 num_right 2 baseline 0.9358127652652215
episode 1700 num_right 2 baseline 0.979113037255114
episode 1800 num_right 2 baseline 0.9386126065203

(root) ~/git/emergent_comms_negotiation (master|…2△1) $ python nets_test.py test-context --term-entropy-reg 0.1
episode 0 num_right 2 baseline 0.3
episode 100 num_right 2 baseline 0.9407348088613829
episode 200 num_right 2 baseline 0.9999707292129745
episode 300 num_right 2 baseline 0.9999983430344679
episode 400 num_right 2 baseline 0.9999951701152274
episode 500 num_right 2 baseline 0.998375681416114
episode 600 num_right 2 baseline 0.9999999839489591
episode 700 num_right 2 baseline 0.9985458530895528
episode 800 num_right 2 baseline 0.9999999999999996

(root) ~/git/emergent_comms_negotiation (master|…2△1) $ python nets_test.py test-context --term-entropy-reg 0.05
episode 0 num_right 1 baseline 0.15
episode 100 num_right 1 baseline 0.7706678249350699
episode 200 num_right 1 baseline 0.8478791655530046
episode 300 num_right 2 baseline 0.9999541170201343
episode 400 num_right 2 baseline 0.9999999999992613
episode 500 num_right 2 baseline 0.9999999999999996
episode 600 num_right 2 baseline 0.9999900963496178
episode 700 num_right 2 baseline 0.9989826653907263
```