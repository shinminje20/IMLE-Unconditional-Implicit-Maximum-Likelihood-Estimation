import argparse
from collections import defaultdict
import numpy as np
import random
from tqdm import tqdm

import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from Data import *
from ModelsContrastive import *
from Utils import *

label2idx = None

def get_eval_data(data_tr, data_te, augs_finetune, augs_te, F=None, use_feats=True):
    """Returns datasets with correct augmentations and potential feature
    extraction.

    Args:
    data_tr -- training data
    data_te -- testing data
    F       -- a HeadlessResNet for constructing features only once or None for
                doing it on the fly with different augmentations
    """
    data_tr = XYDataset(data_tr, augs_finetune)
    data_te = XYDataset(data_te, augs_te) if not data_te == "cv" else "cv"

    if use_feats:
        return (FeatureDataset(data_tr, F),
            (FeatureDataset(data_te, F) if not data_te == "cv" else "cv"))
    else:
        return data_tr, data_te

def get_eval_splits(data_tr, data_te, augs_finetune, augs_te, label_frac=1):
    """Returns and training and testing data for a validation trial.

    Args:
    data_tr         -- a dataset to get data from for supervised training
    data_te         -- a dataset for testing, or 'cv' to use cross validation
    augs_finetune   -- data augmentation for finetuning
    label_frac      -- the fraction of labels to use for training in each trial
    """
    enough_test_data = lambda: (1 - label_frac) * len(data_tr) > len(label2idx)

    global label2idx
    if label2idx is None:
        label2idx = defaultdict(lambda: [])
        for idx,(_,y) in tqdm(enumerate(data_tr), desc="Building label2idx", total=len(data_tr)):
            label2idx[y.item()].append(idx)

    idxs_tr = set()
    for l in label2idx:
        n_idxs = int(len(label2idx[l]) * label_frac)
        idxs_tr |= set(random.sample(label2idx[l], k=n_idxs))

    data_tr_split = Subset(data_tr, list(idxs_tr))
    if data_te == "cv" and enough_test_data():
        idxs_te = [idx for idx in range(len(data_tr)) if not idx in idxs_tr]
        data_te_split = Subset(data_tr, idxs_te)
    elif data_te == "cv" and not enough_test_data():
        raise ValueError("Not enough data for testing. Set --label_frac lower.")
    else:
        data_te_split = data_te

    return data_tr_split, data_te_split

def accuracy(model, loader):
    """Returns the accuracy of [model] on [loader]."""
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            preds = torch.argmax(model(x.to(device)), dim=1)
            total += len(preds)
            correct += torch.sum((preds == y.to(device))).item()

    return correct / total

def one_epoch_supervised(model, optimizer, loader, F=None):
    """Returns a (model, optimizer, loss) tuple after training [model] on
    [loader] for one epoch according to [args].
    Ther returned loss is averaged over batches.
    model       -- a CondConvImplicitModel
    optimizer   -- the optimizer for model
    loader      -- a DataLoader over the data to train on
    F           -- a feature extractor (make sure there's no head)
    """
    # Set the feature extractor and model to the right modes on the right device
    F = nn.Identity() if F is None else F
    F, model = F.to(device), model.to(device)
    model.train()
    F.eval()

    loss_fn, loss_total = nn.CrossEntropyLoss().to(device), 0

    for x,y in loader:
        with torch.no_grad():
            x = F(x.to(device, non_blocking=True))
        model.zero_grad()
        loss = loss_fn(model(x), y.to(device, non_blocking=True))
        loss.backward()
        optimizer.step()
        loss_total += loss.item()

    return model, optimizer


def get_eval_accuracy(data_tr, data_te, F, epochs=100, bs=256):
    """Returns the accuracy of a linear model trained on features from [F] of
    [data_tr] on [data_te].

    Args:
    data_tr     -- the data for training the linear model
    data_te     -- the data for evaluating the accuracy of the linear model
    F           -- a feature extractor (HeadlessResNet) or None if [data_tr] and
                    [data_te] already consist of features
    epochs      -- the number of epochs to train for
    bs          -- the batch size to use
    """
    loader_tr = DataLoader(data_tr, shuffle=True, pin_memory=True,
        batch_size=min(int(len(data_tr) / 4), bs), drop_last=False,
        num_workers=6)
    loader_te = DataLoader(data_te, shuffle=False, pin_memory=True,
        batch_size=1024, drop_last=False, num_workers=6)

    model = nn.Linear(F.out_dim, len(label2idx)).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    for _ in tqdm(range(epochs), desc="Validation epochs", leave=False):
        model, optimizer = one_epoch_supervised(model, optimizer, loader_tr, F)
    return accuracy(nn.Sequential(F, model), loader_te)


def classification_eval(F, data_tr, data_te, augs_finetune, augs_te,
    label_frac=1, samples=1, use_feats=True):
    """Returns classification statistics using feature extractor [F] on Dataset
    [data] with [cv_folds] cross-validation folds.

    Args:
    F               -- the feature extractor being trained
    data_tr         -- a dataset for supervised training
    data_te         -- a dataset for testing, or None to turn on cross validation
    augs_finetune   -- data augmentation for finetuning
    label_frac      -- the fraction of labels to use for each trial. If
                        [data_te] is 'cv' this is the size of the training data
                        for each cross validation fold, otherwise it is the
                        subsample of data used for training. If greater than
                        one, computes splits containing [label_frac] examples
    samples         -- the number of samples to use for cross validation
    use_feats       -- w...
    """


    accuracies = []
    label_frac = label_frac / len(data_tr) if label_frac > 1 else label_frac
    data_tr, data_te = get_eval_data(data_tr, data_te, augs_finetune, augs_te,
        F=F, use_feats=use_feats)

    F = DimensionedIdentity(F.out_dim).to(device) if use_feats else F

    for _ in tqdm(range(samples), leave=False, desc="Validation trials"):
        data_tr_split, data_te_split = get_eval_splits(data_tr, data_te,
            augs_finetune, augs_te, label_frac=label_frac)
        accuracies.append(get_eval_accuracy(data_tr_split, data_te_split, F=F))

    return np.mean(accuracies), np.std(accuracies) * 1.96 / np.sqrt(samples)
