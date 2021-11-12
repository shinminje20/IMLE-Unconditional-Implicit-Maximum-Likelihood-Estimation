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
from Evaluation import classification_eval
from ContrastiveUtils import *
from Utils import *

def one_epoch_contrastive(model, optimizer, loader, temp=.5):
    """Returns a (model, optimizer, loss) tuple after training [model] on
    [loader] for one epoch.

    The returned loss is averaged over batches.

    model           -- a model of the form projection_head(feature_extractor())
    optimizer       -- the optimizer for [model]
    loader          -- a DataLoader over the data to train on
    temp            -- contrastive loss temperature
    """
    model.train()
    loss_fn = NTXEntLoss(temp)
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
    P.add_argument("--eval", default="val", choices=["val", "cv", "test"],
        help="The data to evaluate linear finetunings on")

    # Non-hyperparameter arguments
    P.add_argument("--resume", default=None, type=str,
        help="file to resume from")
    P.add_argument("--suffix", default="", type=str,
        help="suffix")
    P.add_argument("--n_workers", default=6, type=int,
        help="Number of workers for data loading")
    P.add_argument("--eval_iter", default=10, type=int,
        help="number of epochs between linear evaluations")
    P.add_argument("--save_iter", default=100, type=int,
        help="save a model every --save_iter epochs")

    # Hyperparameter arguments
    P.add_argument("--backbone", default="resnet18",
        choices=["resnet18", "resnet50"],
        help="Resnet backbone to use")
    P.add_argument("--bs", default=1000, type=int,
        help="batch size")
    P.add_argument("--epochs", default=1000, type=int,
        help="number of epochs")
    P.add_argument("--lars", default=1, choices=[0, 1],
        help="whether or not to use LARS")
    P.add_argument("--lr", default=1e-3, type=float,
        help="base learning rate")
    P.add_argument("--mm", nargs="+", default=(.9, .999), type=float,
        help="momentum (one arg for SGD, two—beta1 and beta2 for Adam)")
    P.add_argument("--n_ramp", default=10, type=int,
        help="Number of linear ramp epochs at start of training")
    P.add_argument("--opt", choices=["adam", "sgd"], default="adam",
        help="optimizer")
    P.add_argument("--proj_dim", default=128, type=int,
        help="dimension of projection space")
    P.add_argument("--temp", default=.5, type=float,
        help="contrastive loss temperature")
    P.add_argument("--trust", default=.001, type=float,
        help="LARS trust coefficient")
    P.add_argument("--seed", default=0, type=int,
        help="random seed")
    args = P.parse_args()

    args.options = sorted([
        f"bs{args.bs}",
        f"epochs{args.epochs}",
        f"eval_{args.eval}",
        f"lars{args.lars}",
        f"lr{args.lr}",
        f"mm{'_'.join([str(b) for b in flatten([args.mm])])}",
        f"n_ramp{args.n_ramp}",
        f"opt_{args.opt}",
        f"seed{args.seed}",
        f"temp{args.temp}",
        f"trust{args.trust}",
    ])
    ############################################################################
    # Check arguments
    ############################################################################
    if not args.save_iter % args.eval_iter == 0:
        tqdm.write("WARNING: training will save a checkpoint without direct evaluation. Ensure --save_iter % --eval_iter is zero to avoid this.")
    if args.opt == "sgd" and len(args.mm) == 2:
        raise ValueError("--mm must be a single momentum parameter if --opt is 'sgd'.")
    if args.data in no_val_split_datasets and args.eval == "val":
        args.eval = "cv"
        tqdm.write(f"--eval is set to 'val' but no validation split exists for {args.data}. Falling back to cross-validation.")

    args.mm = args.mm[0] if len(args.mm) == 1 else args.mm

    ############################################################################
    # Load prior state if it exists, otherwise instantiate a new training run.
    ############################################################################
    if args.resume is not None:
        model, optimizer, last_epoch, old_args, tb_results = load_(args.resume)
        model = model.to(device)
        last_epoch -= 1
    else:
        # Get the model and optimizer. [get_param_groups()] ensures that the
        # correct param groups are fed to the optimizer so that if the otimizer
        # is wrapped in LARS, LARS will see the right param groups.
        model = get_resnet_with_head(args.backbone, args.proj_dim,
            head_type="projection", is_cifar=("cifar" in args.data)).to(device)

        if args.opt == "adam":
            optimizer = Adam(get_param_groups(model, args.lars), lr=args.lr,
                betas=args.mm, weight_decay=1e-6)
        elif args.opt == "sgd":
            optimizer = SGD(get_param_groups(model, args.lars), lr=args.lr,
                momentum=args.mm, weight_decay=1e-6, nesterov=True)
        else:
            raise ValueError(f"--opt was {args.opt} but must be one of 'adam' or 'sgd'")

        optimizer = LARS(optimizer, args.trust) if args.lars else optimizer

        tb_results = SummaryWriter(resnet_folder(args), max_queue=0)
        last_epoch = -1

    scheduler = CosineAnnealingLinearRampLR(optimizer, args.epochs, args.n_ramp,
        last_epoch=last_epoch)

    data_tr, data_val = get_data_splits(args.data, args.eval)
    augs_tr, augs_finetune, augs_te = get_data_augs(args.data)
    data_ssl = ImagesFromTransformsDataset(data_tr, augs_tr, augs_tr)
    loader = DataLoader(data_ssl, shuffle=True, batch_size=args.bs,
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
            val_acc_avg, val_acc_std = classification_eval(model.backbone,
                data_tr, data_val, augs_finetune, augs_te)
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
