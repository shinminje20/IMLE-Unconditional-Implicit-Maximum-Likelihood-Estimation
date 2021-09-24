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

from Data import ImagesFromTransformsDataset, dataset2n_classes
from ModelsContrastive import get_resnet, ContrastiveLoss
from Utils import *

def one_epoch(model, optimizer, loader, args):
    """Returns a (model, optimizer, loss) tuple after training [model] on
    [loader] for one epoch according to [args].

    Ther returned loss is averaged over batches.

    model       -- a CondConvImplicitModel
    optimizer   -- the optimizer for model
    loader      -- a DataLoader over the data to train on
    args        -- an Argparse object parameterizing the run
    """
    model.train()
    loss_fn = nn.MSELoss(reduction="mean").to(device)

    loss_total, loss_intermediate = 0, 0
    print_interval = len(loader) % args.prints_per_epoch

    for i,(x1,x2) in tqdm(enumerate(loader), desc="Batches", file=sys.stdout, leave=False):
        x1 = x1.float().to(device, non_blocking=True)
        x2 = x2.float().to(device, non_blocking=True)

        model.zero_grad()
        fx1, fx2 = model(x1), model(x2)
        loss = loss_fn(fx1, fx2)
        loss.backward()
        optimizer.step()

        loss_intermediate += loss.item()

        if i % print_interval == 0 and not i == 0:
            tqdm.write(f"   intermediate loss: {loss_intermediate}")
            loss_total += loss_intermediate
            loss_intermediate = 0

    loss_total += loss_intermediate
    return model, optimizer, loss_total

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="IMLE training")
    P.add_argument("--resume", default=None, type=str,
        help="file to resume from")
    P.add_argument("--prints_per_epoch", default=5, type=int,
        help="intermediate loss prints per epoch")
    P.add_argument("--n_workers", default=6, type=int,
        help="Number of workers for data loading")
    P.add_argument("--suffix", default="", type=str,
        help="suffix")

    P.add_argument("--data", choices=["cifar10"], default="cifar10", type=str,
        help="dataset to load images from")
    P.add_argument("--backbone", default="resnet18",
        choices=["resnet18", "resnet50"],
        help="Resnet backbone to use")
    P.add_argument("--seed", default=0, type=int,
        help="random seed")

    P.add_argument("--epochs", default=20, type=int,
        help="number of epochs")
    P.add_argument("--bs", default=64, type=int,
        help="batch size")
    P.add_argument("--opt", choices=["adam", "sgd"], default="adam",
        type=str,
        help="optimizer")
    P.add_argument("--lr", default=1e-3, type=float,
        help="base learning rate")
    P.add_argument("--n_ramp", default=10, type=int,
        help="Number of linear ramp epochs at start of training")
    P.add_argument("--mm", nargs="+", default=.01, type=float,
        help="momentum (one arg for SGD, two—beta1 and beta2 for Adam)")
    args = P.parse_args()

    args.options = sorted([
        f"bs{args.bs}",
        f"epochs{args.epochs}",
        f"lr{args.lr}",
        f"mm{'_'.join([str(b) for b in args.mm])}"
        f"n_ramp{args.n_ramp}",
        f"opt_{args.opt}",
        f"seed{args.seed}",
    ])

    ############################################################################
    # Load prior state if it exists, otherwise instantiate a new training run.
    ############################################################################
    if args.resume is not None:
        model, optimizer, last_epoch, old_args, writer = load(args.resume)
        model = model.to(device)
    else:
        # Construct the model and optimizer.
        input_dim = dataset2input_dim[args.data]
        model = get_resnet(args.backbone, head_type="projection").to(device)
        if args.opt == "adam":
            optimizer = Adam(model.parameters(), lr=args.lr, betas=args.mm,
                weight_decay=5e-4)
        elif args.opt == "sgd":
            optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.mm,
                weight_decay=5e-4)
        else:
            raise ValueError(f"--opt was {args.opt} but must be one of 'adam' or 'sgd'")

        last_epoch = -1
        writer = SummaryWriter()
    scheduler = CosineAnnealingLinearRampLR(optimizer, args.epochs, args.n_ramp,
        last_epoch=last_epoch)

    ############################################################################
    # Construct the dataset and dataloader. For each dataset, the last k indices
    # are cut off and used for the visual validation dataset.
    ############################################################################
    if args.data == "cifar10":
        data = CIFAR10(root="../Datasets", train=True, download=True,
            transform=transforms.ToTensor())
        data_tr = Subset(data, range(len(data) - 1000))
        data_val = Subset(data, range(len(data) - 1000, len(data)))
    else:
        raise ValueError(f"--data was {args.data} but must be one of 'cifar10'")

    dataset = ImagesFromTransformsDataset(data_tr, cifar_augs_tr, cifar_augs_tr)
    loader = DataLoader(dataset, shuffle=True, batch_size=args.bs,
        drop_last=True, num_workers=args.n_workers, pin_memory=True)

    ############################################################################
    # Begin training!
    ############################################################################
    for e in tqdm(range(last_epoch + 1, args.epochs), desc="Epochs", file=sys.stdout):

        # Run one epoch
        tqdm.write(f"=== STARTING EPOCH {e} | lr {scheduler.get_last_lr()}")
        model, optimizer, loss_tr = one_epoch(model, optimizer, loader, args)

        # Perform a classification cross validation if desired, and otherwise
        # print/log results or merely that the epoch happened.
        if e % args.eval_iter == 0 and not e == 0:
            val_acc_avg, val_acc_std = cv_classification_eval(model.backbone,
                data_val, dataset2n_classes[args.data], cv_folds=5)
            writer.add_scalar("Loss/train", loss_tr / len(loader), e)
            writer.add_scalar("Accuracy/val", val_acc_avg, e)
            writer.add_scalar("Learning rate", scheduler.get_last_lr(), e)
            tqdm.write(f"=== END OF EPOCH {e} | loss {loss_tr / len(loader)} | val acc {val_acc_avg:f5} ± val acc {val_acc_std:f5}")
        else:
            writer.add_scalar("Loss/train", loss_tr / len(loader), e)
            writer.add_scalar("Learning rate", scheduler.get_last_lr(), e)
            tqdm.write(f"=== END OF EPOCH {e} | loss {loss_tr / len(loader)}")

        # Saved the model and any visual validation results if they exist
        if e % args.save_iter == 0 and not e == 0:
            save_model(model, optimizer, e, args, writer, resnet_folder(args))
            tqdm.write("Saved training state")

        scheduler.step()
