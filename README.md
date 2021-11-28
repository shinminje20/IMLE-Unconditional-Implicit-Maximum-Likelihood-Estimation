# ISICLE
**I**mage **S**ynthes**i**s for **C**ontrastive **Le**arning

### Setup

**Python Environment**
```
gdown==4.2.0
lpips==0.1.4
python==3.9.7
pytorch==1.10.0
tensorboard==2.7.0
torchvision==0.11.1
tqdm==4.62.2
```

**Datasets**
Download the data we'll be using:
```
python Data/SetupMiniImagenet.py
```

### Running Code

### Extending This Code
**Training on new data**:
1. In `Data.py` you need to
    1. Modify `get_data_splits()` so that one of the options corresponds to the dataset you want to use, eg. `if data_str == data_name` and have that case return a training, validation, and testing set for the dataset. Use the `val_frac` argument if you need to split data off from the training set to create the validation set. It's possible that in doing this step you'll have to create a `Dataset` subclass to read your dataset.
    2. Modify `dataset2n_classes` and `dataset2input_dim` to include your dataset. If there's no validation split for the dataset, include it in `no_val_split_datasets` too.
2. You need to add `data_name` to the `choices` keyword arguments of the `--data` argparse option in the following files:
    1.  `TrainContrastive.py`

### TODOs
