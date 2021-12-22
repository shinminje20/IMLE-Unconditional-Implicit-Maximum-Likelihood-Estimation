# **I**mage **S**ynthes**i**s for **C**ontrastive **Le**arning

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
