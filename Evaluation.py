import argparse
from collections import defaultdict
import json
import numpy as np
import random
from tqdm import tqdm

import torch
from torch.optim import Adam, SGD
import torch.nn as nn
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from Data import *
from utils.UtilsContrastive import *
from utils.Utils import *
from torch.cuda.amp import GradScaler, autocast

def get_label2idxs(data_tr, data_name=None, data_split=None):
    """Returns a label to indices mapping for [data_tr]."""
    data_split = "train" if data_split == "cv" else data_split

    if data_name is not None and data_split is not None:
        save_path = f"{project_dir}/Data/{data_name}/{data_split}_label2idx.json"
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                return json.load(f)

    label2idxs = defaultdict(lambda: [])
    for idx,(_,y) in tqdm(enumerate(data_tr), desc="Building label2idxs", total=len(data_tr)):
        label2idxs[y].append(idx)

    if data_name is not None and data_split is not None:
        save_dir_path = f"{project_dir}/Data/{data_name}"
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)

        if not os.path.exists(f"{save_dir_path}/{data_split}_label2idx.json"):
            with open(f"{save_dir_path}/{data_split}_label2idx.json", "w+") as f:
                json.dump(label2idxs, f)

    return label2idxs

def accuracy(model, loader):
    """Returns the accuracy of [model] on [loader]."""
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            preds = torch.argmax(model(x.to(device)), dim=1)
            correct += torch.sum((preds == y.to(device))).item()
            total += len(preds)

    return correct / total

def get_eval_data(data_tr, data_te, augs_fn, augs_te, F, precompute_feats=True):
    """Returns validation training and testing datasets. The testing dataset may
    be a string 'cv' to indicate cross validation if [data_te] is 'cv'.

    Args:
    data_tr -- training data
    data_te -- testing data
    augs_fn -- training augmentations
    augs_te -- testing augmentations
    F       -- a HeadlessResNet for constructing features only once or None for
                doing it on the fly with different augmentations
    precompute_feats    -- whether to precompute features.
    """
    if precompute_feats:
        return (FeatureDataset(XYDataset(data_tr, augs_fn), F),
                FeatureDataset(XYDataset(data_te, augs_te), F))
    else:
        return XYDataset(data_tr, augs_fn), XYDataset(data_te, augs_te)

def get_eval_trial_accuracy(data_tr, data_te, F, out_dim, num_classes, trial=0, epochs=100, bs=16):
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
        batch_size=bs, drop_last=False,
        num_workers=6, **seed_kwargs(trial))
    loader_te = DataLoader(data_te, pin_memory=True, batch_size=1024,
        drop_last=False, num_workers=6, **seed_kwargs(trial))

    model = nn.Linear(out_dim, num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, epochs)
    F = nn.Identity().to(device) if F is None else F.to(device)
    F.eval()
    loss_fn = nn.CrossEntropyLoss().to(device)

    for e in tqdm(range(epochs), desc="Validation epochs", leave=False):
        model.train()

        scaler = GradScaler()
        for x,y in loader_tr:
            with autocast():
                with torch.no_grad():
                    x = x.to(device, non_blocking=True)
                    x = x if F is None else F(x)

                model.zero_grad()
                loss = loss_fn(model(x), y.to(device, non_blocking=True))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

    return accuracy(nn.Sequential(F, model), loader_te)

def classification_eval(feature_extractor, data_tr, data_te, augs_fn, augs_te,
    precompute_feats=True, ex_per_class="all", trials=3, epochs=100, bs=16,
    data_name=None, data_split=None):
    """Returns evaluation accuracy of feature extractor [feature_extractor].

    Args:
    feature_extractor   -- feature extractor
    data_name           -- the the name of the data
    data_tr             -- training data for linear model
    data_te             -- testing data
    augs_fn             -- augmentations for training (of linear model) data
    augs_te             -- augmentations for testing data
    precompute_feats    -- precompute features. This is massively faster because
                            the CNN need be run only once, but reduces the
                            number of augmentations used in total.
    ex_per_class        -- examples per class
    trials              -- number of trials. If [data_te] is 'cv', number of
                            cross-validation folds
    """
    def get_split_idxs(idxs, split_index, include):
        """Computes a split of [idxs] to be used for cross-validation.

        Args:
        idxs        -- the indices to be split
        split_index -- the index of the split (which trial is being run)
        include     -- whether to return indices between the computed start and
                        endpoints in [idxs], or those exluding those indices
        """
        start = split_index * (len(idxs) // trials)
        end = (1 + split_index) * (len(idxs) // trials)
        return idxs[start:end] if include else idxs[:start] + idxs[end:]

    label2idxs = get_label2idxs(data_tr, data_name=data_name, data_split=data_split)
    out_dim = feature_extractor.out_dim
    num_classes = len(data_tr.class_to_idx)
    accuracies = []

    for t in tqdm(range(trials), desc="Validation trials"):

        if data_te == "cv":
            label2idxs_te = {y: get_split_idxs(label2idxs[y], t, include=True)
                             for y in label2idxs}
            label2idxs_tr = {y: get_split_idxs(label2idxs[y], t, include=False)
                             for y in label2idxs}

            if not ex_per_class == "all":
                idxs_tr = [random.sample(label2idxs_tr[y], ex_per_class)
                                    for y in label2idxs_tr]
            else:
                idxs_tr = [idx for y in label2idxs_tr for idx in label2idxs_tr[y]]
            idxs_te = [idx for y in label2idxs_te for idx in label2idxs_te[y]]
            trial_data_tr = Subset(data_tr, indices=flatten(idxs_tr))
            trial_data_te = Subset(data_tr, indices=flatten(idxs_te))
        else:
            if not ex_per_class == "all":
                idxs_tr = [random.sample(label2idxs[y], ex_per_class)
                           for y in label2idxs]
            else:
                idxs_tr = [idx for y in label2idxs for idx in label2idxs[y]]
            trial_data_tr = Subset(data_tr, indices=flatten(idxs_tr))
            trial_data_te = data_te

        trial_data_tr, trial_data_te = get_eval_data(trial_data_tr, trial_data_te,
            augs_fn, augs_te, F=feature_extractor,
            precompute_feats=precompute_feats)

        accuracies.append(get_eval_trial_accuracy(trial_data_tr, trial_data_te,
            (None if precompute_feats else feature_extractor),
            out_dim, num_classes,
            epochs=epochs, bs=bs, trial=t))

    return np.mean(accuracies), np.std(accuracies) * 1.96 / np.sqrt(trials)


if __name__ == "__main__":
    P = argparse.ArgumentParser(description="IMLE training")
    P.add_argument("--eval", default="val", choices=["val", "test"],
        help="The data to evaluate linear finetunings on")
    P.add_argument("--model", default=None, type=str,
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

    model, _, _, old_args, _ = load_(args.model)
    model = model.to(device)

    data_tr, data_eval = get_data_splits(args.data, args.eval)
    augs_tr, augs_fn, augs_te = get_ssl_augs(args.data, args.color_s)

    val_acc_avg, val_acc_std = classification_eval(model.backbone, data_tr,
        data_val, augs_fn, augs_te,
        data_name=old_args.data,
        ex_per_class="all",
        precompute_feats=args.precompute_feats,
        epochs=args.epochs, bs=args.bs)
    tqdm.write(f"val acc {val_acc_avg:.5f} ± {val_acc_std:.5f}")
