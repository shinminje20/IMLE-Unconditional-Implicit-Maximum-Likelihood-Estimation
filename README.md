# **I**mage **S**ynthes**i**s for **C**ontrastive **Le**arning

## Method
**Baseline I**: SimCLR, but no manifold-leaving augmentations. This is interesting because it is in effect what happens when the strength of the corruption-based augmentations is zero. Expected performance is poor because the augmentations are weak.

**Baseline II**: SimCLR. This is interesting because
    a) the augmentations are strong
    b) the augmentations leave the manifold (can we quantify this?)
    c) it's the number-one thing to compare with

**Real-images baseline:** Pretrain CaMNet, use it to generate images and train SimCLR on top of it. This is interesting because it's probably the naive thing to try and immediately gets( to the heart of the real-images-only matter. I feel that this is the cake on which everything else is (possibly important) icing.

...

## Experiments
**Datasets**:
 - CIFAR-10 _I don't think we can run CaMNet on this since the images are literally too small, but I think it would be helpful for iteration while using a smaller generator!_
 - MiniImagenet (for iteration)
 - ImageNet (for paper)

1. All baselines vs all proposed models. Hypothesis is that SimCLR is good, our method is slightly better.

2. Our method vs SOTA contrastive learning papers. The post-SimCLR ones I've read that immediately occur to me are
    - [Tian et. al. _What Makes for Good Views for Contrastive Learning?_, 2020](https://arxiv.org/abs/2005.10243)
    - [Chuang et al. _Debiased Contrastive Learning_. 2020](https://arxiv.org/abs/2007.00224)
    - [Ramapuram et al. _Stochastic Contrastive Learning_, 2021.](https://arxiv.org/pdf/2110.00552.pdf)*

### Ablation Studies
1. Rerun experiment one a bunch of times, and each time remove one augmentation or corruption. This is a somewhat interesting empirically-focused three quarters of a page.

# Questions
_CaMNet/SimCLR Standardization?_
_What's the best way to extend CaMNet to the full dataset? Do we expect it to just work?_

## Setup
If you just want to use the SimCLR component of this, look under TBD instead.
Install and activate the following Python environment:
```
gdown==4.2.0
lpips==0.1.4
python==3.9.7
pytorch==1.10.0
tensorboard==2.7.0
torchvision==0.11.1
tqdm==4.62.2
```
Install CUDA toolkit 11.3 (if you just want the SimCLR implementation, don't do this):
```
```

Download and set up the datasets to use. To get the datasets we use in the paper, run
```
```

## Running Code
You can run the SimCLR baseline via `python TrainSimCLR`. All command line arguments are optional. Below is the code to (mostly) replicate the reported SimCLR CIFAR-10 performance, with some extraneous arguments specified for clarity too.
```
TrainSimCLR.py --data cifar10 --eval test --n_workers 6 --eval_iter 10 --save_iter 100 --backbone resnet50 --bs 1000 --color_s .5, --epochs 1000, --lars 1 --lr 1e-3, --mm .9 .99 --n_ramp 10 --opt adam --proj_dim 128 --temp .5  --trust 1e-3 --seed 0
```
You can run CaMNet as follows....

## Extending This Code
TODO once this is finished!

## TODOs
 - Get CaMNet working on miniImagenet
 - Improve/fix CaMNet setup description in the README

## DONE
- **12/18/201**: Added `MakeSmallDataset.py` for making a smaller split of a dataset. This aids in training CaMNet
- Setup miniImagenet data and test it with basic SimCLR. We get 65% accuracy in a reasonably large setting, which is probably fine since there's less data per class than in Imagenet.
- Get `TrainSimCLR.py` to work. Turned out evaluation needs to be done with training augmentations (grrrrr), and we can get 92% accuracy on the CIFAR-10, which is reasonably in line with the paper. The only difference is we use the Adam optimizer.
