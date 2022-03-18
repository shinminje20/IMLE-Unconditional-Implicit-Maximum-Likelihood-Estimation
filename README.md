# **I**mage **S**ynthes**i**s for **C**ontrastive **Le**arning

## Setup
If you just want to use the SimCLR component of this, look under TBD instead.
Install and activate the following Python environment:
```
gdown==4.2.0
python==3.9.7
pytorch==1.10.0
torchvision==0.11.1
tqdm==4.62.2
wandb==12.1
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'

```

Download and set up the datasets to use. To get the datasets we use in the paper, run
```
python data/SetupDataset.py --data DATASET_NAME --res 16 ... 256
```

## Running Code
We use a modified version of CAMNet []() as an image generator, which needs to be pretrained:
```
python TrainGenerator.py --data DATASET_NAME --res 16 ... 256
```
This takes several days to train. Once the generator is pretrained, it can be used as an input for contrastive learning:
```
python TrainISICLE.py --data DATASET_NAME --res 16 ... 256 --backbone BACKBONE --use_latent_supervision
```
Finally, you can test the trained contrastive learner:
```
python Evaluation.py --model ...
```
