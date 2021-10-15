from collections import defaultdict
import numpy as np
import random
import sys
from tqdm import tqdm

import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from Data import FeatureDataset, XYDataset
from Utils import *

def one_epoch_classification(model, optimizer, loader, F=None):
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

    loss_fn = nn.CrossEntropyLoss(reduction="mean").to(device)
    loss_total = 0

    for x,y in loader:

        with torch.no_grad():
            x = F(x.to(device, non_blocking=True))

        model.zero_grad()
        fx = model(x)
        loss = loss_fn(fx, y.to(device, non_blocking=True))
        loss.backward()
        optimizer.step()

        loss_total += loss.item()

    return model, optimizer, loss_total / len(loader)

def accuracy(model, loader):
    """Returns the accuracy of [model] on [loader]."""
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for x,y in data:
            preds = torch.argmax(model(x.to(device)), dim=1)
            total += len(preds)
            correct += torch.sum((preds == y.to(device))).item()

    return correct / total

label2idx = None

def classification_eval(F, data_tr, data_te, augs_finetune, augs_te,
    percent_labels=.01, samples=3):
    """Returns classification statistics using feature extractor [F] on Dataset
    [data] with [cv_folds] cross-validation folds.

    Args:
    F               -- the feature extractor being trained
    data_tr         -- a dataset for supervised training
    data_te         -- a dataset for testing, or None to turn on cross validation
    augs_finetune   -- data augmentation for finetuning
    """
    accuracies = []
    epochs = 100
    bs = 100

    ############################################################################
    # Build the label2idx dictionary if it doesn't already exist. This is needed
    # for class-balanced subsampling.
    ############################################################################
    global label2idx
    if label2idx is None:
        label2idx = defaultdict(lambda: [])
        for idx,(_,y) in tqdm(enumerate(data_tr), desc="Building label2idx", total=len(data_tr), leave=False, file=sys.stdout):
            assert isinstance(y, int)
            label2idx[y].append(idx)

    for _ in tqdm(range(samples), file=sys.stdout, leave=False, desc="Validation trials"):

        # Get a list of indices to [data_tr] to build into a training dataset
        idxs_tr = set()
        for label in label2idx:
            n_idxs = int(len(label2idx[label]) * percent_labels)
            idxs_tr |= set(random.sample(label2idx[label], k=n_idxs))

        # Finish building the data for training and testing. If [data_te] is
        # None, cross validate on the training data.
        X = XYDataset(Subset(data_tr, list(idxs_tr)), augs_finetune)
        if data_te is None:
            idxs_te = [idx for idx in range(len(data_tr)) if not idx in idxs_tr]
            Y = XYDataset(Subset(data_tr, idxs_te), augs_te)
        else:
            Y = data_te

        loader_tr = DataLoader(X, batch_size=bs, shuffle=True,
            pin_memory=True, drop_last=False)
        loader_te = DataLoader(Y, batch_size=min(len(data_te), 1024),
            shuffle=True, pin_memory=True, drop_last=False)

        # Get a model, optimizer, and scheduler
        model = nn.Linear(F.out_dim, len(label2idx)).to(device)
        optimizer = SGD(model.parameters(), lr=.1, nesterov=True, momentum=.9)
        scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=.001)

        # Train [model] and the record its test accuracy
        for e in tqdm(range(epochs), desc="Validation: epochs", leave=False, file=sys.stdout):
            model, optimizer, _ = one_epoch_classification(model, optimizer,
                loader_tr, F=F)
            scheduler.step()
        accuracies.append(accuracy(nn.Sequential(F, model), loader_te))

    return np.mean(accuracies), np.std(accuracies) * 1.96 / np.sqrt(samples)


def visual_eval(model, dataset, args):
    """Returns [args.n_val] (x, fx) pairs where [x] is a an image selected
    randomly from [dataset] and [fx] is the model's image synthesis conditioned
    on [x].

    If [args.show] is True, prints the images as well

    Args:
    model   -- a CondConvImplicitModel
    dataset -- a dataset of images
    args    -- an Argparse object parameterizing the visual evalation
    """
    return None
