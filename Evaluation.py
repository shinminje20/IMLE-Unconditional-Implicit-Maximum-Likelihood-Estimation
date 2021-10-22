import argparse
from collections import defaultdict
import numpy as np
import random
from tqdm import tqdm

import torch
from torch.optim import Adam, SGD
import torch.nn as nn
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from Data import *
from ModelsContrastive import *
from Utils import *

label2idx = None

def get_eval_data(data_tr, data_te, augs_finetune, augs_te, F=None, precompute_feats=True):
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

    if precompute_feats:
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
            if isinstance(y, int):
                label2idx[y].append(idx)
            elif isinstance(y, torch.Tensor):
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


def get_eval_accuracy(data_tr, data_te, F, epochs=100, bs=256, verbose=False):
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
    optimizer = SGD(model.parameters(), lr=.1, nesterov=True, momentum=.9, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, epochs)

    for e in tqdm(range(epochs), desc="Validation epochs", leave=False):
        model, optimizer = one_epoch_supervised(model, optimizer, loader_tr, F)
        scheduler.step()
        if verbose:
            val_acc = accuracy(nn.Sequential(F, model), loader_te)
            tqdm.write(f"End of epoch {e + 1} | val acc {val_acc:.5f}")

    return accuracy(nn.Sequential(F, model), loader_te)


def classification_eval(F, data_tr, data_te, augs_finetune, augs_te,
    label_frac=1, samples=1, precompute_feats=True, epochs=100, bs=256,
    verbose=False):
    """Returns classification statistics using feature extractor [F] on Dataset
    [data] with [cv_folds] cross-validation folds.

    Args:
    F                   -- the feature extractor being trained
    data_tr             -- a dataset for supervised training
    data_te             -- a dataset for testing, or None to turn on cross validation
    augs_finetune       -- data augmentation for finetuning
    label_frac          -- the fraction of labels to use for each trial. If
                            [data_te] is 'cv' this is the size of the training
                            data for each cross validation fold, otherwise it is
                            the subsample of data used for training. If greater
                            than one, computes splits containing [label_frac]
                            examples
    samples             -- the number of samples to use for cross validation
    precompute_feats    -- w...
    """
    accuracies = []
    label_frac = label_frac / len(data_tr) if label_frac > 1 else label_frac
    data_tr, data_te = get_eval_data(data_tr, data_te, augs_finetune, augs_te,
        F=F, precompute_feats=precompute_feats)

    F = DimensionedIdentity(F.out_dim).to(device) if precompute_feats else F

    for _ in tqdm(range(samples), leave=False, desc="Validation trials"):
        data_tr_split, data_te_split = get_eval_splits(data_tr, data_te,
            augs_finetune, augs_te, label_frac=label_frac)
        accuracies.append(get_eval_accuracy(data_tr_split, data_te_split, F=F,
            epochs=epochs, bs=bs, verbose=verbose))

    return np.mean(accuracies), np.std(accuracies) * 1.96 / np.sqrt(samples)

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="IMLE training")
    P.add_argument("--eval", default="val", choices=["val", "cv", "test"],
        help="The data to evaluate linear finetunings on")
    P.add_argument("--resume", default=None, type=str,
        help="file to resume from")
    P.add_argument("--precompute_feats", default=0, choices=[0, 1],
        help="whether to precompute features")
    P.add_argument("--suffix", default="", type=str,
        help="suffix")
    P.add_argument("--bs", default=256, type=int,
        help="batch size")
    P.add_argument("--epochs", default=200, type=int,
        help="number of epochs")
    P.add_argument("--seed", default=0, type=int,
        help="random seed")
    args = P.parse_args()

    model, _, _, old_args, _ = load_(args.resume)
    model = model.to(device)

    data_tr, data_val = get_data_splits(old_args.data, args.eval)
    augs_tr, augs_finetune, augs_te = get_data_augs(old_args.data)

    val_acc_avg, val_acc_std = classification_eval(model.backbone, data_tr,
        data_val, augs_finetune, augs_te,
        precompute_feats=args.precompute_feats, epochs=args.epochs, bs=args.bs,
        verbose=True)
    tqdm.write(f"val acc {val_acc_avg:.5f} ± {val_acc_std:.5f}")
