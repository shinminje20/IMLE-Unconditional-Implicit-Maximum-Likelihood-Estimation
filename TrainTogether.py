import argparse
import sys
from tqdm import tqdm

import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from Data import *
from VisualEval import *
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

    for i,(x,y,z) in tqdm(enumerate(loader), desc="Batches", file=sys.stdout, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        z = z.to(device, non_blocking=True)

        model.zero_grad()
        fx = model(x, z)
        loss = loss_fn(fx, y)
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
    P.add_argument("--seed", default=0, type=int,
        help="random seed")

    P.add_argument("--z_dim", nargs=3, default=[64, 1, 1], type=int,
        help="three integers giving the dimensionality of the generated noise")
    P.add_argument("--epochs", default=20, type=int,
        help="number of epochs")
    P.add_argument("--bs", default=64, type=int,
        help="batch size")
    P.add_argument("--opt", choices=["adam", "sgd"], default="adam",
        type=str,
        help="optimizer")
    P.add_argument("--lr", default=1e-3, type=float,
        help="base learning rate")
    P.add_argument("--lr_gamma", default=.1, type=float,
        help="decay learning rate by this factor every --lr_period epochs")
    P.add_argument("--lr_period", default=5, type=int,
        help="decay learning rate by --lr_gamma every --lr_period epochs")
    P.add_argument("--betas", nargs=2, default=[.9, .99], type=float,
        help="adam betas")
    args = P.parse_args()

    args.options = sorted([
        f"betas_{'_'.join([str(b) for b in args.betas])}"
        f"bs{args.bs}",
        f"epochs{args.epochs}",
        f"lr{args.lr}",
        f"lr_gamma{args.lr_gamma}",
        f"lr_period{args.lr_period}",
        f"opt_{args.opt}",
        f"seed{args.seed}",
        f"z_dim_{'_'.join([str(z) for z in args.z_dim])}"
    ])

    ############################################################################
    # Load prior state if it exists, otherwise instantiate a new training run.
    ############################################################################
    if args.resume is not None:
        model, optimizer, last_epoch, old_args, _ = load_generator(args.resume)
        model = model.to(device)
    else:
        # Construct the model and optimizer.
        input_dim = dataset2input_dim[args.data]
        model = CondConvImplicitModel(input_dim, args.z_dim).to(device)
        if args.opt == "adam":
            optimizer = Adam(model.parameters(), lr=args.lr, betas=args.betas,
                weight_decay=5e-4)
        else:
            raise ValueError(f"--opt was {args.opt} but must be one of 'adam'")

        last_epoch = -1

    scheduler = StepLR(optimizer, step_size=args.lr_period, gamma=args.lr_gamma,
        last_epoch=last_epoch)

    ############################################################################
    # Construct the dataset and dataloader. For each dataset, the last k indices
    # are cut off and used for the visual validation dataset.
    ############################################################################
    if args.data == "cifar10":
        data = CIFAR10(root="../Datasets", train=True, download=True,
            transform=transforms.ToTensor())
        dataset_tr = Subset(data, range(len(data) - 100))
        dataset_val = Subset(data, range(len(data) - 100, len(data)))
    else:
        raise ValueError(f"--data was {args.data} but must be one of 'cifar10'")

    dataset = IMLEDataset(dataset_tr, args.z_dim)
    loader = DataLoader(dataset, shuffle=True, batch_size=args.bs,
        drop_last=True, num_workers=args.n_workers, pin_memory=True)

    ############################################################################
    # Begin training!
    ############################################################################
    for e in tqdm(range(last_epoch + 1, args.epochs), desc="Epochs", file=sys.stdout):

        # Run one epoch
        tqdm.write(f"=== STARTING EPOCH {e} | lr {scheduler.get_last_lr()}")
        model, optimizer, loss_tr = one_epoch(model, optimizer, loader, args)
        tqdm.write(f"=== END OF EPOCH {e} | loss {loss_tr / len(loader)}")

        # Perform a visual validation if desired
        if e % args.eval_iter == 0 and not e == 0:
            eval_results = visual_eval(model, dataset_val, args)
        else:
            eval_results = None

        # Saved the model and any visual validation results if they exist
        if e % args.save_iter == 0 and not e == 0:
            save_generator(model, optimizer, e, args, eval_results)
            tqdm.write("Saved training state")

        scheduler.step()
