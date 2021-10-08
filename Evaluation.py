import numpy as np
import sys
from tqdm import tqdm

import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import Subset, ConcatDataset, DataLoader

from Data import FeatureDataset
from Utils import *

def one_epoch_classification(model, optimizer, loader, feature_extractor=None):
    """Returns a (model, optimizer, loss) tuple after training [model] on
    [loader] for one epoch according to [args].

    Ther returned loss is averaged over batches.

    model               -- a CondConvImplicitModel
    optimizer           -- the optimizer for model
    loader              -- a DataLoader over the data to train on
    feature_extractor   -- a feature extractor (make sure there's no head)
    """
    # Set model and the possible feature extractor to the right modes
    model.train()
    if feature_extractor is not None:
        feature_extractor.eval()

    loss_fn = nn.CrossEntropyLoss(reduction="mean").to(device)
    loss_total = 0

    for i,(x,y) in enumerate(loader):

        # If [feature_extractor] isn't None, convert [x] into its features and
        # on the right device, otherwise, just move [x] onto the correct device.
        if feature_extractor is not None:
            with torch.no_grad():
                x = feature_extractor(x.to(device, non_blocking=True))
        else:
            x = x.to(device, non_blocking=True)

        model.zero_grad()
        fx = model(x)
        loss = loss_fn(fx, y.to(device, non_blocking=True))
        loss.backward()
        optimizer.step()

        loss_total += loss.item()

    return model, optimizer, loss_total / len(loader)

def accuracy(model, data):
    """Returns the accuracy of [model] on [data]."""
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for x,y in data:
            preds = torch.argmax(model(x.to(device)), dim=1)
            total += len(preds)
            correct += torch.sum((preds == y.to(device))).item()

    return correct / total

def cv_classification_eval(F, data, n_classes, cv_folds=5, mode="linear"):
    """Returns classification statistics using feature extractor [F] on Dataset
    [data] with [cv_folds] cross-validation folds.
    """
    ############################################################################
    # Create a dataset [cv_data] of model features and a list of indices of
    # starting and stopping indices for cross validation folds [folds_idxs]
    ############################################################################
    f_size = len(data) // cv_folds
    folds_idxs = [(s * f_size, s * f_size + f_size) for s in range(cv_folds)]
    cv_data = Subset(data, range(f_size * cv_folds))

    if not len(data) == len(cv_data):
        tqdm.write(f"Validation dataset length is {len(data)} and not divisible by the {cv_folds} CV folds. The dataset has been truncated to {len(cv_data)} with a fold size of {f_size}.")

    ############################################################################
    # For each fold of the data, train linear(F(x)) and record the result.
    ############################################################################
    F = F.to(device)
    accuracies = []
    if mode == "linear":
        # Set the batch size using a heuristic
        bs = min(64, max(4, len(cv_data) // 16))
        for start_idx, stop_idx in tqdm(folds_idxs, desc="Validation: folds", leave=False, file=sys.stdout):

            # Get DataLoaders over the training and testing data for the current
            # cross validation fold
            loader_tr = DataLoader(Subset(cv_data, range(start_idx, stop_idx)),
                batch_size=bs, shuffle=True, pin_memory=True, drop_last=False)
            loader_te = DataLoader(ConcatDataset([
                    Subset(cv_data, range(0, start_idx)),
                    Subset(cv_data, range(stop_idx, len(data)))]),
                batch_size=min(f_size, 128), shuffle=True, pin_memory=True,
                drop_last=False)

            # Train a model on [loader_tr] and test it on [loader_te]
            model = nn.Linear(F.out_dim, n_classes).to(device)
            optimizer = Adam(model.parameters(), lr=1e-3)
            for e in tqdm(range(100), desc="Validation: epochs", leave=False, file=sys.stdout):
                model, optimizer, _ = one_epoch_classification(model, optimizer,
                    loader_tr, feature_extractor=F)

            # Compute the accuracy of the function model(F(x))
            accuracies.append(accuracy(nn.Sequential(F, model), loader_te))
    else:
        raise NotImplementedError()

    return np.mean(accuracies), np.std(accuracies)

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
