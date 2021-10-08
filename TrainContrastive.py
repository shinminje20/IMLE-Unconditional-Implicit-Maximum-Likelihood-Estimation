import argparse
import sys
from tqdm import tqdm

import torch
from torch.optim import Adam, SGD
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from Data import *
from Evaluation import cv_classification_eval
from ModelsContrastive import get_resnet_with_head, ContrastiveLoss
from Utils import *

def one_epoch_contrastive(model, optimizer, loader, temp):
    """Returns a (model, optimizer, loss) tuple after training [model] on
    [loader] for one epoch according to [args].

    Ther returned loss is averaged over batches.

    model           -- a CondConvImplicitModel
    optimizer       -- the optimizer for model
    loader          -- a DataLoader over the data to train on
    temp            -- contrastive loss temperature
    """
    model.train()
    loss_fn = ContrastiveLoss(temp)
    loss_total = 0

    for i,(x1,x2) in tqdm(enumerate(loader), desc="Batches", file=sys.stdout, total=len(loader), leave=False):
        x1 = x1.float().to(device, non_blocking=True)
        x2 = x2.float().to(device, non_blocking=True)

        model.zero_grad()
        fx1, fx2 = model(x1), model(x2)
        loss = loss_fn(fx1, fx2)
        loss.backward()
        optimizer.step()

        loss_total += loss.item()

    return model, optimizer, loss_total / len(loader)

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="IMLE training")
    P.add_argument("--data", choices=["cifar10"], default="cifar10", type=str,
        help="dataset to load images from")
    P.add_argument("--backbone", default="resnet18",
        choices=["resnet18", "resnet50"],
        help="Resnet backbone to use")

    # Non-hyperparameter arguments
    P.add_argument("--resume", default=None, type=str,
        help="file to resume from")
    P.add_argument("--suffix", default="", type=str,
        help="suffix")
    P.add_argument("--n_workers", default=4, type=int,
        help="Number of workers for data loading")
    P.add_argument("--eval_iter", default=10, type=int,
        help="number of epochs between linear evaluations")
    P.add_argument("--save_iter", default=100, type=int,
        help="save a model every --save_iter epochs")

    # Hyperparameter arguments
    P.add_argument("--epochs", default=1000, type=int,
        help="number of epochs")
    P.add_argument("--n_ramp", default=10, type=int,
        help="Number of linear ramp epochs at start of training")
    P.add_argument("--bs", default=2500, type=int,
        help="batch size")
    P.add_argument("--opt", choices=["adam", "sgd", "lars"], default="adam",
        type=str,
        help="optimizer")
    P.add_argument("--lr", default=1e-3, type=float,
        help="base learning rate")
    P.add_argument("--mm", nargs="+", default=(.9, .99), type=float,
        help="momentum (one arg for SGD, two—beta1 and beta2 for Adam)")
    P.add_argument("--temp", default=.5, type=float,
        help="contrastive loss temperature")
    P.add_argument("--trust", default=.001, type=float,
        help="LARS trust coefficient")
    P.add_argument("--proj_dim", default=128, type=int,
        help="dimension of projection space")
    P.add_argument("--val_frac", default=.1, type=float,
        help="amount of data to use for validation")
    P.add_argument("--seed", default=0, type=int,
        help="random seed")
    args = P.parse_args()

    args.options = sorted([
        f"bs{args.bs}",
        f"epochs{args.epochs}",
        f"lr{args.lr}",
        f"mm{'_'.join([str(b) for b in flatten([args.mm])])}",
        f"n_ramp{args.n_ramp}",
        f"opt_{args.opt}",
        f"seed{args.seed}",
        f"temp{args.temp}",
        f"trust{args.trust}",
        f"val_frac{args.val_frac}"
    ])

    ############################################################################
    # Check arguments
    ############################################################################
    if not args.save_iter % args.eval_iter == 0:
        tqdm.write("WARNING: training will save a checkpoint without direct evaluation. Ensure --save_iter % --eval_iter is zero to avoid this.")
    if args.val_frac > 0 and args.eval_iter <= 0:
        tqdm.write("WARNING: since --val_frac is nonzero, some data will be split into a validation dataset, however, since --eval_iter is negative, no validation will be performed.")
    if args.val_frac > 0 and not args.data in no_val_split_datasets:
        tqdm.write("WARNING: since --data has a validation split, --val_frac is ignored and the given validation split used instead.")
    if args.val_frac == 0 and args.data in no_val_split_datasets:
        tqdm.write("Since --val_frac is zero and the given dataset has no validation split, setting --eval_iter to -1")
        args.eval_iter = -1
    if args.val_frac == -1 and args.data in no_val_split_datasets:
        tqdm.write("Since --val_frac is -1, validating on test data (this is to ensure that the model behaves like others)")
    if not args.opt in ["adam"] and isinstance(args.mm, tuple):
        raise ValueError("--mm must be a single momentum parameter unless --opt is one of 'adam'")

    args.mm = args.mm[0] if len(args.mm) == 1 else args.mm

    ############################################################################
    # Load prior state if it exists, otherwise instantiate a new training run.
    ############################################################################
    if args.resume is not None:
        model, optimizer, last_epoch, old_args, tb_results = load_(args.resume)
        model = model.to(device)
        last_epoch -= 1
    else:
        # Get the model
        model = get_resnet_with_head(args.backbone, args.proj_dim,
            head_type="projection").to(device)

        # Get the optimizer for the model
        if args.opt == "adam":
            optimizer = Adam(model.parameters(), lr=args.lr, betas=args.mm,
                weight_decay=1e-6)
        elif args.opt == "lars":
            optimizer = LARS(model, args.lr, args.mm,
                trust_coefficient=args.trust)
        elif args.opt == "sgd":
            optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.mm,
                weight_decay=1e-6)
        else:
            raise ValueError(f"--opt was {args.opt} but must be one of 'adam', 'lars', or 'sgd'")

        # Get a SummaryWriter to record results
        tb_results = SummaryWriter(resnet_folder(args), flush_secs=1, max_queue=0)

        # Set the last epoch to -1. If we're resuming, this else block won't be
        # run and last epoch will be something else. Here we set it in the
        # default case so it can be provided to the learning rate scheduler.
        last_epoch = -1

    # Get the scheduler.
    scheduler = CosineAnnealingLinearRampLR(optimizer, args.epochs, args.n_ramp,
        last_epoch=last_epoch)

    ############################################################################
    # Construct the dataset and dataloader. For each dataset, the last k indices
    # are cut off and used for the visual validation dataset.
    #
    # Negative arguments to --val_frac have meaning, but should be considered
    # zero when computing the lengths of the training and validation splits.
    ############################################################################
    data_tr, data_val, data_te = get_data_splits(args.data, seed=args.seed,
        val_frac=max(0, args.val_frac))
    data_tr = ImagesFromTransformsDataset(data_tr, cifar_augs_tr, cifar_augs_tr)
    data_val = ImageLabelDataset((data_te if args.val_frac == -1 else data_val),
        cifar10_augs_val)
    loader = DataLoader(data_tr, shuffle=True, batch_size=args.bs,
        drop_last=True, num_workers=args.n_workers, pin_memory=True)

    ############################################################################
    # Begin training!
    ############################################################################
    for e in tqdm(range(max(last_epoch + 1, 1), args.epochs + 1), desc="Epochs", file=sys.stdout):

        model, optimizer, loss_tr = one_epoch_contrastive(model, optimizer,
            loader, args.temp)

        # Perform a classification cross validation if desired, and otherwise
        # print/log results or merely that the epoch happened.
        if e % args.eval_iter == 0 and not e == 0 and args.eval_iter > 0:
            val_acc_avg, val_acc_std = cv_classification_eval(model.backbone,
                data_val, dataset2n_classes[args.data], cv_folds=5)
            tb_results.add_scalar("Loss/train", loss_tr / len(loader), e)
            tb_results.add_scalar("Accuracy/val", val_acc_avg, e)
            tb_results.add_scalar("LR", scheduler.get_last_lr()[0], e)
            tqdm.write(f"End of epoch {e} | lr {scheduler.get_last_lr()[0]:.5f} | loss {loss_tr / len(loader):.5f} | val acc {val_acc_avg:.5f} ± {val_acc_std:.5f}")
        else:
            tb_results.add_scalar("Loss/train", loss_tr / len(loader), e)
            tb_results.add_scalar("LR", scheduler.get_last_lr()[0], e)
            tqdm.write(f"End of epoch {e} | lr {scheduler.get_last_lr()[0]:.5f} | loss {loss_tr / len(loader):.5f}")

        # Saved the model and any visual validation results if they exist
        if e % args.save_iter == 0 and not e == 0:
            save_(model, optimizer, e, args, tb_results, resnet_folder(args))
            tqdm.write("Saved training state")

        scheduler.step()
