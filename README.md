# ISICLE
**I**mage **S**ynthes**i**s for **C**ontrastive **Le**arning

## Setup

### Python environment
```
gdown==4.2.0
lpips==0.1.4
python==3.9.7
pytorch==1.10.0
tensorboard==2.7.0
torchvision==0.11.1
tqdm==4.62.2
```

###  Datasets
Download and set up the data we'll be using:
```
python Data/SetupMiniImagenet.py
```

### CaMNet
We use CamNet to generate images for training. _If you just want to use this project to run basic SimCLR---our baseline---you don't need to and probably shouldn't do the following!_

1. Prepare datasets for CamNet's usage:
    ```
    python Data/InvertDirStructure.py --folder Data/miniImagenet
    ```
2. Install additional Python packages:
    ```
    ```
3. Install CUDA.

## Running Code
You can run the SimCLR baseline via
```
python TrainSimCLR.py
```
You can run CaMNet as follows....
```
tbd
```
You can run the proposed method via
```
python TrainISICLE.py
```

### Model saving
Model checkpoints are saved to the `Models` directory, inside a folder named with a long string formed by concatenating the model's hyperparameters, as well as the training configuration. Inside this folder, checkpoints appear as `x.pt` where `x` is the index of the epoch run prior to saving. Checkpoints contain more information than just the model, so to load one you'll need to use the `load_()` function in `Utils.py`. Additionally, there are a number of TensorBoard files, and if you `cd` into the folder, so you can run
```
tensorboard --logdir Models/MODEL_DIRECTORY
```
to view logged data about the model's training.

## Extending This Code

### Training on new data
1. This project expects data to appear in an ImageFolder format. The validation split is optional:
    ```
    Data/dataset_folder
      ├───train
      │   ├── class 1
      │   │   ├── image 1
      │   │   ├── ...
      │   │   └── image m
      |   ├── ...
      |   └── class n
      ├── val
      └── test
    ```
2. Run `python Data/InvertDirStructure.py --folder DATASET_FOLDER`. This will make a copy of your dataset in the `Data/camnet_data` folder with the directory structure inverted for CamNet compatibility.
3. Decide on a short `NAME` for the dataset, eg. `cifar10`. Then, in `Data.py`, do the following:
    - If there's no validation split for the dataset, add add `NAME` to `no_val_split_datasets`.
    - If the images that will be loaded from the dataset are small (eg. 32x32), add `NAME` to `small_image_datasets`.
    - Add your dataset to the `get_data_splits()` function. You can see how this is done for miniImagenet and essentially copy that.
    - Add your dataset to the `get_ssl_data_augs()` function. Again, you can essentially copy an existing case.
4. In any file you want to run that takes a `--data` command line argument, add `NAME` to the `choices` kwarg of the corresponding `add_argument("--data", choices=[], ...)` for that file.

## TODOs
 - Get CaMNet working on miniImagenet
 - Improve CaMNet setup description in the README

## DONE
- Setup miniImagenet data and test it with basic SimCLR. We get 65% accuracy in a reasonably large setting, which is probably fine since there's less data per class than in Imagenet.
- Get `TrainSimCLR.py` to work. Turned out evaluation needs to be done with training augmentations (grrrrr), and we can get 92% accuracy on the CIFAR-10, which is reasonably in line with the paper. The only difference is we use the Adam optimizer.
