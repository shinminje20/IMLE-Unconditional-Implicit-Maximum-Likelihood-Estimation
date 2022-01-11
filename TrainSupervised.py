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
from utils.ContrastiveUtils import *
from utils.Utils import *
from utils.NestedNamespace import *

class KMeansLoss(nn.Module):

    def __init__(self):
        super(KMeansLoss, self).__init__()

    def forward(self, fx, y, method="one_distance_matrix"):
        """
        Args:
        fx  -- BSxD tensor of embeddings
        y   -- BS tensor of labels
        """
        if method == "one_distance_matrix":
            D = torch.cdist(fx, fx)
            L = torch.unique(y)
            loss = 0
            for label in L:
                loss += torch.sum(D[y == l][:, y == label])
        elif method == "many_distance_matrix":
            L = torch.unique(y)
            loss = 0
            for label in L:
                E = fx[y == l]
                loss += torch.sum(torch.cdist(E))
        return loss

def one_epoch_supervised(model, optimizer, loader, loss_fn):
    """Returns a (model, optimizer, loss) tuple after training [model] on
    [loader] for one epoch.

    The returned loss is averaged over batches.

    model           -- a model of the form projection_head(feature_extractor())
    optimizer       -- the optimizer for [model]
    loader          -- a DataLoader over the data to train on
    loss_fn         -- loss function to use
    """
    model.train()
    loss_total = 0

    for x,y in tqdm(loader, desc="Batches", file=sys.stdout, total=len(loader), leave=False):
        model.zero_grad()
        loss = loss_fn(model(x.float().to(device, non_blocking=True)))
        loss.backward()
        optimizer.step()
        loss_total += loss.item()

    return model, optimizer, loss_total / len(loader)

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="SimCLR training")
    P.add_argument("--data", choices=["cifar10", "miniImagenet"],
        default="cifar10",
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
    P.add_argument("--bs", default=64, type=int,
        help="batch size")
    P.add_argument("--color_s", default=1, type=float,
        help="color distortion strength")
    P.add_argument("--strong", default=1, choices=[0, 1],
        help="whether augmentations should be strong or not")
    P.add_argument("--epochs", default=1000, type=int,
        help="number of epochs")
    P.add_argument("--lars", default=1, choices=[0, 1],
        help="whether or not to use LARS")
    P.add_argument("--loss", choices=["xentropy", "kmeans"], required=True,
        help="loss function for proxy (non-validation) task")
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
    P.add_argument("--trust", default=.001, type=float,
        help="LARS trust coefficient")
    P.add_argument("--seed", default=0, type=int,
        help="random seed")
    args = NestedNamespace(P.parse_args())

    args.options = sorted([
        f"bs{args.bs}",
        f"epochs{args.epochs}",
        f"color_s{args.color_s}",
        f"eval_{args.eval}",
        f"lars{args.lars}",
        f"loss_{args.loss}",
        f"lr{args.lr}",
        f"mm{'_'.join([str(b) for b in flatten([args.mm])])}",
        f"n_ramp{args.n_ramp}",
        f"opt_{args.opt}",
        f"seed{args.seed}",
        f"strong{args.strong}",
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
        raise ValueError("The requested dataset has no validation split. Run with --eval test or cv instead.")

    # This needs to be a tuple or a float depending on its length
    args.mm = args.mm[0] if len(args.mm) == 1 else args.mm

    tqdm.write(f"Will save model to {simclr_folder(args)}")

    ############################################################################
    # Load prior state if it exists, otherwise instantiate a new training run.
    ############################################################################
    if args.resume is not None:
        model, optimizer, last_epoch, args, tb_results = load_(args.resume)
        model = model.to(device)
        last_epoch -= 1
    else:
        if args.loss == "xentropy":
            model = get_resnet_with_head(args.backbone,
                data2split2n_class[args.dataset]["train"], head_type="linear",
                small_image=(args.data in small_image_datasets)).to(device)
            loss_fn = nn.CrossEntropyLoss().to(device)
        elif args.loss == "kmeans":
            model = get_resnet_with_head(args.backbone, args.proj_dim,
                head_type="projection",
                small_image=(args.data in small_image_datasets)).to(device)
            loss_fn = KMeansLoss().to(device)
        if args.opt == "adam":
            optimizer = Adam(get_param_groups(model, args.lars), lr=args.lr,
                betas=args.mm, weight_decay=1e-6)
        elif args.opt == "sgd":
            optimizer = SGD(get_param_groups(model, args.lars), lr=args.lr,
                momentum=args.mm, weight_decay=1e-6, nesterov=True)
        else:
            raise ValueError(f"--opt was {args.opt} but must be one of 'adam' or 'sgd'")
        optimizer = LARS(optimizer, args.trust) if args.lars else optimizer

        # Get the TensorBoard logger and set last_epoch to -1
        tb_results = SummaryWriter(simclr_folder(args), max_queue=0)
        last_epoch = -1

    ############################################################################
    # Instantiate the scheduler and get the data
    ############################################################################
    set_seed(args.seed)
    scheduler = CosineAnnealingLinearRampLR(optimizer, args.epochs, args.n_ramp,
        last_epoch=last_epoch)
    data_tr, data_eval = get_data_splits_ssl(args.data, args.eval)
    augs_tr, augs_fn, augs_te = get_ssl_augs(args.data, color_s=args.color_s,
        strong=args.strong)
    loader = DataLoader(data_tr, shuffle=True, batch_size=args.bs,
        drop_last=True, num_workers=args.n_workers, pin_memory=True,
        **seed_kwargs(args.seed))

    ############################################################################
    # Begin training!
    ############################################################################
    for e in tqdm(range(max(last_epoch + 1, 1), args.epochs + 1), desc="Epochs", file=sys.stdout):
        model, optimizer, loss_tr = one_epoch_contrastive(model, optimizer,
            loader, loss_fn)

        # Perform a classification cross validation if desired, and otherwise
        # print/log results or merely that the epoch happened.
        if e % args.eval_iter == 0 and not e == 0 and args.eval_iter > 0:
            val_acc_avg, val_acc_std = classification_eval(model.backbone,
                data_tr, data_eval, augs_fn, augs_te, data_name=args.data,
                data_split=args.eval)
            tb_results.add_scalar("Loss/train", loss_tr / len(loader), e)
            tb_results.add_scalar("Accuracy/val", val_acc_avg, e)
            tb_results.add_scalar("LR", scheduler.get_last_lr()[0], e)
            tqdm.write(f"End of epoch {e} | lr {scheduler.get_last_lr()[0]:.5f} | loss {loss_tr / len(loader):.5f} | val acc {val_acc_avg:.5f} ± {val_acc_std:.5f}")
        else:
            tb_results.add_scalar("Loss/train", loss_tr / len(loader), e)
            tb_results.add_scalar("LR", scheduler.get_last_lr()[0], e)
            tqdm.write(f"End of epoch {e} | lr {scheduler.get_last_lr()[0]:.5f} | loss {loss_tr / len(loader):.5f}")

        # Saved the model if desired
        if e % args.save_iter == 0 and not e == 0:
            save_simclr(model, optimizer, e, args, tb_results, simclr_folder(args))
            tqdm.write("Saved training state")

        scheduler.step()