
### 1/4/22

### 1/3/22
**Did**
  - Got a better understanding of how CAMNet works—on a syntactic level—and concluded (big thanks to Alireza/Shichong)
    - the code is complex and it's not worth messing with much (but re-implementing the architecture _was_ a useful exercise nonetheless)
    - it's too slow to train to be useful while we optimize much else. However, this isn't a problem; we should just treat it like a de-corruption operator, freeze its weights, and train the contrastive learner and how we corrupt images instead. _This makes sense for all the reasons we talked about while I was flying, too!_

**Do**
  - Get CAMNet doing super-resolution and colorization on three-class combo dataset on the cluster, and make sure the config is set so it finishes
  - Literature search things that can act as corruption operators—dropout, augmentations
 
 **Musings**
 - It seems pretty clear that the contrastive learner will be basically SimCLR. What about corruptions? Even without gradients flowing through CAMNet, RL exists. There are a number of papers that propose learned corruptions, after a fashion:
    - [Phan and Le, _AutoDropout: Learning Dropout Patterns to Regularize Deep Neural Networks_, 2021](https://arxiv.org/pdf/2101.01761.pdf) learns patterned regions of dropout across the activations of multiple layers of a neural net. It's trained through REINFORCE with the optimized quantity being the main model's validation loss. The papers gets good results, but _I'm not sure it'd be optimal for contrastive learning, since it's probably more valuable to be able to drop out specific (semantic? non-semantic?) subsets of an object or the background instead of regular shapes of it._
    - [French et al., _Milking CowMask for Semi-Supervised Image Classification_, 2020](https://arxiv.org/pdf/2003.12022.pdf) is able to generate irregular patterns, but focuses on semi-supervised learning and doesn't generate the augmentation conditioned on the data.
    - ...
    - We could just learn the mask? Functionally, the model becomes
        $$
            f(g(c(x), z_1)), g(c(x), z_2))
        $$
     but gradients wouldn't flow to $c$, the corruption operator that generates masks for the low-resolution images fed to CAMNet. I'm still a fan of reward for something involved in image generation being a moving average of the negative derivative of loss of the contrastive learner. As long as gradients can be computed in some way to train $c$
 - Another option is to learn on CAMNet's latent codes. However, I'm not a huge fan because it means we have to worry about making CAMNet generate real images too.

### Before...
_There were some older things in the log, and they're either on HackMD or were deleted. RIP_
