import argparse
from datetime import datetime
from tqdm import tqdm
import wandb

import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from CAMNet import CAMNet
from Corruptions import get_non_learnable_batch_corruption
from Data import *
from utils.Utils import *
from utils.UtilsColorSpace import *
from utils.UtilsLPIPS import LPIPSFeats
from utils.UtilsNN import *

################################################################################
# Loss whatnot
################################################################################
class LPIPSLoss(nn.Module):
    """Returns loss between LPIPS features of generated and target images."""
    def __init__(self, reduction="mean"):
        super(LPIPSLoss, self).__init__()
        self.lpips = LPIPSFeats()
        self.reduction = reduction
        self.loss = nn.MSELoss(reduction=self.reduction)

    def forward(self, fx, y): return self.loss(self.lpips(fx), self.lpips(y))

def get_loss_fn(loss_fn):
    """Returns an unreduced loss function of type [loss_fn], or [loss_fn] itself
    if [loss_fn] is an nn.Module (this allows LPIPS networks to be reused).
    """
    if loss_fn == "lpips":
        return LPIPSLoss(reduction="none").to(device)
    elif loss_fn == "mse":
        return nn.MSELoss(reduction="none")
    elif isinstance(loss_fn, nn.Module):
        return loss_fn
    else:
        raise ValueError(f"Unknown loss type {loss_fn}")

def compute_loss(fx, y, loss_fn, reduction="none", list_reduction="mean"):
    """Returns the loss of output [fx] against target [y] using loss function
    [loss_fn] and reduction strategies [reduction] and [list_reduction].

    Args:
    fx              -- list of or single BSxCxHxW generated images
    y               -- list of or single BSxCxHxW ground-truth image
    loss_fn         -- unreduced loss function that acts on 4D tensors
    reduction       -- how to reduce across the images
    list_reduction  -- how to reduce across the list if inputs include lists
    """
    if not loss_fn.reduction == "none":
        raise ValueError(f"Wrapped loss function's reduction must be 'none' but was '{loss_fn.reduction}'")

    if isinstance(fx, list) and isinstance(y, list) and len(fx) == len(y):
        losses = [compute_loss(f, t, loss_fn, reduction) for f,t in zip(fx, y)]
        if list_reduction == "none":
            return torch.stack(losses, axis=0)
        elif list_reduction == "mean":
            return torch.mean(torch.stack(losses), axis=0)
        elif list_reduction == "sum":
            return torch.sum(torch.stack(losses), axis=0)
        else:
            raise ValueError(f"Unknown list_reduction '{list_reduction}'")
    else:
        if reduction == "none":
            return loss_fn(fx, y)
        elif reduction == "mean":
            return torch.mean(loss_fn(fx, y))
        elif reduction == "batch":
            return torch.mean(loss_fn(fx, y).view(fx.shape[0], -1), axis=1)
        else:
            raise ValueError(f"Unknown reduction '{reduction}'")

################################################################################
# IMLE whatnot
################################################################################

def get_images(corruptor, model, dataset, idxs=[0], samples_per_image=1):
    """Returns a list of lists, where each sublist contains first a ground-truth
    image and then [samples_per_image] images conditioned on that one.

    Args:
    corruptor   -- a corruptor to remove information from images
    model       -- a model to fix corrupted images
    dataset     -- GeneratorDataset to load images from
    idxs        -- the indices to [dataset] to get images for
    """
    with torch.no_grad():
        images_dataset = Subset(dataset, idxs)

        corrupted_data = CorruptedDataset(images_dataset, corruptor,
            color_space_convert=color_space_convert_tr, bs=len(idxs))

        results = [[ys[-1], cx] for cx,ys in corrupted_data]

        corrupted_data = ExpandedDataset(corrupted_data,
            expand_factor=samples_per_image)
        codes_data = ZippedDataset(*get_new_codes(model.get_z_dims(),
            corrupted_data, model, num_samples=0))
        batch_dataset = ZippedDataset(codes_data, corrupted_data)

        for idx,(codes,(cx,_)) in enumerate(batch_dataset):
            codes = [c.unsqueeze(0) for c in make_device(codes)]
            fx = model(cx.to(device).unsqueeze(0), codes, loi=-1).cpu()
            results[idx // samples_per_image].append(fx)

    results = make_cpu(make_3dim(color_space_convert_view(results)))
    return results


def get_new_codes(z_dims, corrupted_data, backbone, loss_fn="mse",
    code_bs=6, num_samples=120, verbose=1):
    """Returns a list of new latent codes found via hierarchical sampling. For
    a batch size of size BS, and N elements to [z_dims], returns a list of codes
    that where the ith code is of the size of the ith elmenent of [z_dims]
    expanded across the batch dimension.

    Args:
    z_dims      -- list of shapes describing a latent code
    data        -- a Subset of the training dataset
    backbone    -- model backbone. Must support a 'loi' argument
    loss_fn     -- the means of determining distance. For inputs of size Nx...,
                    it should return a tensor of N losses.
    code_bs     -- batch size to test codes in
    num_samples -- number of times we try to find a better code for each image
    """
    loss_fn = get_loss_fn(loss_fn)
    bs = len(corrupted_data)
    level_codes = [torch.randn((bs,)+z, device=device) for z in z_dims]
    loader = DataLoader(corrupted_data, batch_size=code_bs, num_workers=num_workers)

    for level_idx in tqdm(range(len(z_dims)), desc="Resolutions", leave=False, dynamic_ncols=True):
        least_losses = torch.ones(bs, device=device) * float("inf")

        for i in tqdm(range(num_samples), desc="Sampling", leave=False, dynamic_ncols=True):
            for idx,(cx,ys) in enumerate(loader):
                start_idx, end_idx = code_bs * idx, code_bs * (idx + 1)
                least_losses_batch = least_losses[start_idx:end_idx]

                old_codes = [l[start_idx:end_idx] for l in level_codes[max(0, level_idx - 1):]]
                new_codes = torch.randn((code_bs,) + z_dims[level_idx], device=device)
                test_codes = old_codes + [new_codes]

                with torch.no_grad():
                    fx = backbone(cx.to(device), test_codes, loi=level_idx)
                    losses = compute_loss(fx, ys[level_idx].to(device), loss_fn,
                        reduction="batch")

                change_idxs = losses < least_losses_batch
                level_codes[level_idx][start_idx:end_idx][change_idxs] = new_codes[change_idxs]
                least_losses[start_idx:end_idx][change_idxs] = losses[change_idxs]

            if verbose == 1 and i % 20 == 0:
                tqdm.write(f"    Current average per-image loss {torch.mean(least_losses):.5f}")

    return make_cpu(level_codes)


def one_epoch_imle(corruptor, model, optimizer, dataset, loss_fn="lpips",
    bs=1, mini_bs=1, code_bs=1, iters_per_code_per_ex=1000, num_samples=12,
    verbose=1, color_space_convert=lambda x: x):
    """Returns a (corruptor, model, optimizer) tuple after training [model] and
    optionally [corruptor] for one epoch on data from [loader] via cIMLE.

    ****************************************************************************
    Note that in the typical terminology, a 'minibatch' and a 'batch' are
    synonymous, and here a 'minibatch' is a subset of a 'batch'.
    ****************************************************************************

    Args:
    model               -- a generative model that can undo corruptions
    optimizer               -- an optimizer for both the corruptor
                                (if wanted) and model
    loader                  -- DataLoader returning training and target data.
                                The data for both inputs and targets should be
                                a list of tensors
    loss_fn                 -- the loss function to use
    iters_per_code_per_ex   -- number of gradient steps for each code
    mini_bs                 -- the batch size to run per iteration. Must evenly
                                divide the batch size of [loader]
    """
    loss_fn = get_loss_fn(loss_fn)
    total_loss = 0

    rand_idxs = random.sample(range(len(dataset)), len(dataset))
    for batch_idx in tqdm(range(0, len(dataset), bs), desc="Batches", leave=False, dynamic_ncols=True):

        # Get a dataset of corrupted images and their targets
        images_data = Subset(dataset, rand_idxs[batch_idx:batch_idx + bs])
        corrupted_data = CorruptedDataset(images_data, corruptor, bs=mini_bs,
            color_space_convert=color_space_convert)

        # Get codes for the corrupted images. This takes the place of the min()
        # function in IMLE, since we return the best (minimum loss) codes for
        # each image from the function
        codes_data = ZippedDataset(*get_new_codes(model.get_z_dims(),
            corrupted_data, model, loss_fn=loss_fn, code_bs=code_bs,
            num_samples=num_samples, verbose=verbose))

        # Zip the codes and the corrupted images and their targets together
        batch_dataset = ZippedDataset(codes_data, corrupted_data)
        loader = DataLoader(batch_dataset, batch_size=mini_bs, shuffle=True,
            num_workers=num_workers)

        # Train the model
        inner_loop_iters = int(iters_per_code_per_ex * len(batch_dataset) / mini_bs)
        for _ in tqdm(range(inner_loop_iters), desc="Inner loop", leave=False, dynamic_ncols=True):
            for codes,(cx,ys) in tqdm(loader, desc="Minibatches", leave=False, dynamic_ncols=True):

                model.zero_grad()
                fx = model(cx.to(device), make_device(codes))
                loss = compute_loss(fx, make_device(ys), loss_fn,
                    reduction="mean", list_reduction="mean")
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        if verbose == 1:
            tqdm.write(f"    current loss {loss.item():.5f}")

    return corruptor, model, optimizer, total_loss / len(loader)


def rgb2lab_with_dims(input):
    if isinstance(input, list):
        return [rgb2lab_with_dims(x) for x in input]
    elif isinstance(input, torch.Tensor) and len(input.shape) == 4:
        return rgb2lab(input.to(device))
    elif isinstance(input, torch.Tensor) and len(input.shape) == 3:
        return rgb2lab(input.to(device).unsqueeze(0)).squeeze(0)

def lab2rgb_with_dims(input):
    if isinstance(input, list):
        return [lab2rgb_with_dims(x) for x in input]
    elif isinstance(input, torch.Tensor) and len(input.shape) == 4:
        return lab2rgb(input.to(device))
    elif isinstance(input, torch.Tensor) and len(input.shape) == 3:
        return lab2rgb(input.to(device).unsqueeze(0)).squeeze(0)

def parse_camnet_args(unparsed_args):
    """Returns an argparse Namespace and the remaining unparsed arguments after
    parsing [unparsed_args].
    """
    P = argparse.ArgumentParser(description="CAMNet architecture argparsing")
    P.add_argument("--code_nc", default=5, type=int,
        help="number of code channels")
    P.add_argument("--in_nc", default=3, type=int,
        help="number of input channels")
    P.add_argument("--out_nc", default=3, type=int,
        help=" number of output channels")
    P.add_argument("--map_nc", default=128, type=int,
        help="number of input channels to mapping net")
    P.add_argument("--latent_nc", default=512, type=int,
        help="number of channels inside the mapping net")
    P.add_argument("--resid_nc", default=[128, 64, 64, 64], type=int,
        nargs="+",
        help="list of numbers of residual channels in RRDB blocks for each CAMNet level")
    P.add_argument("--dense_nc", default=[256, 192, 128, 64], type=int,
        nargs="+",
        help="list of numbers of dense channels in RRDB blocks for each CAMNet level")
    P.add_argument("--n_blocks", default=6, type=int,
        help="number of RRDB blocks inside each level")
    P.add_argument("--act_type", default="leakyrelu",
        choices=["leakyrelu"],
        help="activation type")
    P.add_argument("--feat_scales", default=None, type=int,
        help="amount by which to scale features, or None")
    return P.parse_known_args(unparsed_args)

def get_corruptor_args(unparsed_args):
    """Returns an argparse Namespace and the remaining unparsed arguments after
    parsing [unparsed_args].
    """
    P = argparse.ArgumentParser(description="Corruptor argparsing")
    P.add_argument("--grayscale", default=1, type=int, choices=[0, 1],
        help="grayscale corruption")
    P.add_argument("--pix_mask_size", default=8, type=int,
        help="fraction of pixels to mask at 16x16 resolution")
    P.add_argument("--pix_mask_frac", default=.1, type=float,
        help="fraction of pixels to mask at 16x16 resolution")
    P.add_argument("--rand_illumination", default=.1, type=float,
        help="amount by which the illumination of an image can change")
    return P.parse_known_args(unparsed_args)

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="CAMNet training")
    P.add_argument("--wandb", default=1, choices=[0, 1], type=int,
        help="Use W&B logging")
    P.add_argument("--data", default="cifar10", choices=datasets,
        help="data to train on")
    P.add_argument("--eval", default="val", choices=["cv", "val", "test"],
        help="data for validation")
    P.add_argument("--res", nargs="+", required=True, type=int,
        default=[16, 32],
        help="resolutions to see data at")
    P.add_argument("--arch", default="camnet", choices=["camnet"],
        help="Model architecture to use. Architecture hyperparameters are parsed later based on this")
    P.add_argument("--suffix", default="",
        help="optional training suffix")
    P.add_argument("--options", default=[], nargs="+",
        help="options")
    P.add_argument("--verbose", choices=[0, 1], default=0,
        help="verbosity level")
    P.add_argument("--init_type", choices=["kaiming", "normal"], default="kaiming",
        help="NN weight initialization method")
    P.add_argument("--init_scale", type=float, default=1,
        help="Scale for weight initialization")
    P.add_argument("--loss", default="lpips", choices=["mse", "lpips"],
        help="loss function to use")
    P.add_argument("--epochs", default=20, type=int,
        help="number of epochs (months) to train for")
    P.add_argument("--n_ramp", default=1, type=int,
        help="number of epochs to ramp learning rate")
    P.add_argument("--bs", type=int, default=300,
        help="batch size")
    P.add_argument("--mini_bs", type=int, default=10,
        help="minibatch size")
    P.add_argument("--code_bs", type=int, default=6,
        help="batch size to use for sampling codes")
    P.add_argument("--num_samples", type=int, default=120,
        help="number of samples for IMLE")
    P.add_argument("--ipcpe", type=int, default=2,
        help="iters_per_code_per_ex")
    P.add_argument("--lr", type=float, default=1e-4,
        help="learning rate")
    P.add_argument("--wd", type=float, default=1e-6,
        help="weight decay")
    P.add_argument("--mm", nargs="+", default=(.9, .999), type=float,
        help="momentum (one arg for SGD, two—beta1 and beta2 for Adam)")
    P.add_argument("--color_space", choices=["rgb", "lab"], default="lab",
        help="Color space to use during training")
    args, unparsed_args = P.parse_known_args()


    ############################################################################
    # Collect the arguments for generating the model and corruptor from
    # separate functions
    ############################################################################
    if args.arch == "camnet":
        model_args, unparsed_args = parse_camnet_args(unparsed_args)
    else:
        raise ValueError(f"Unknown architecture '{args.arch}'")

    corruptor_args, unparsed_args = get_corruptor_args(unparsed_args)

    if len(unparsed_args) > 0:
        raise ValueError(f"Got unknown arguments. Unparseable arguments:\n    {' '.join(unparsed_args)}")
    else:
        args.__dict__ |= vars(model_args) | vars(corruptor_args)
    ############################################################################
    # Create the dataset and options, and check that various batch types have
    # okay sizes
    ############################################################################
    data_tr, data_eval = get_data_splits(args.data, args.eval, args.res)
    base_transform = get_gen_augs()
    data_tr = GeneratorDataset(data_tr, base_transform)
    data_eval = GeneratorDataset(data_eval, base_transform)

    if not evenly_divides(args.bs, len(data_tr)) or len(data_tr) < args.bs:
        raise ValueError(f"--bs {args.bs} must be at most and evenly divide the length of the dataset {len(data_tr)}")
    if not evenly_divides(args.mini_bs, args.bs) or args.bs < args.mini_bs:
        raise ValueError(f"--mini_bs {args.mini_bs} must be at most and evenly divide --bs {args.bs}")
    if not evenly_divides(args.code_bs, args.bs) or args.bs < args.code_bs:
         raise ValueError(f"--code_bs {args.code_bs} must be at most and evenly divide --bs {args.bs}")

    tqdm.write(f"Training will take {int(len(data_tr) / args.mini_bs * args.ipcpe * args.epochs)} gradient steps")
    ############################################################################
    # Create the corruption, mode,l and its optimizer. Any model specific
    ############################################################################
    corruptor = get_non_learnable_batch_corruption(**vars(corruptor_args) | {"color_space": args.color_space})

    if args.arch == "camnet":
        new_args = {"n_levels": len(args.res) - 1, "base_size": args.res[0]}
        model = CAMNet(**(vars(model_args) | new_args)).to(device)
        init_weights(model, init_type=args.init_type, scale=args.init_scale)
        core_params = [p for n,p in model.named_parameters() if not "map" in n]
        map_params = [p for n,p in model.named_parameters() if "map" in n]
        optimizer = Adam([{"params": core_params},
                          {"params": map_params, "lr": 1e-2 * args.lr}],
                          lr=args.lr, weight_decay=args.wd, betas=args.mm)
    else:
        raise ValueError(f"Unknown architecture '{args.arch}'")

    ############################################################################
    # Set up remaining training utilities and data logging
    ############################################################################
    if args.color_space == "lab":
        color_space_convert_tr = rgb2lab_with_dims
        color_space_convert_view = lab2rgb_with_dims
    else:
        color_space_convert_tr = lambda x: x
        color_space_convert_view = lambda x: x

    last_epoch = -1
    scheduler = CosineAnnealingLR(optimizer, args.epochs,
        last_epoch=last_epoch, verbose=args.verbose)

    save_dir = generator_folder(args)
    wandb.init(anonymous="allow",
               mode="online" if args.wandb else "disabled",
               name=save_dir.replace(f"{project_dir}/models/", "").replace("/", "-"),
               project="ISICLE generator training",
               notes=args.suffix,
               config=args)

    ############################################################################
    # Begin training!
    ############################################################################

    for e in tqdm(range(max(last_epoch + 1, 1), args.epochs + 1), desc="Epochs", dynamic_ncols=True):

        corruptor, model, optimizer, loss_tr = one_epoch_imle(
            corruptor,
            model,
            optimizer,
            data_tr,
            loss_fn=args.loss,
            bs=args.bs,
            mini_bs=args.mini_bs,
            code_bs=args.code_bs,
            num_samples=args.num_samples,
            iters_per_code_per_ex=args.ipcpe,
            verbose=args.verbose,
            color_space_convert=color_space_convert_tr)

        ########################################################################
        # After each epoch, log results and data
        ########################################################################
        tqdm.write(f"loss_tr {loss_tr:.5f} | lr {scheduler.get_last_lr()[0]:.5e}")

        # Save validation image results locally and to W&B
        val_images = get_images(corruptor, model, data_eval,
            idxs=random.sample(range(len(data_eval)), 10),
            samples_per_image=5)
        results_file = f"{save_dir}/val_images/epoch{e}.png"
        save_images_grid(val_images, results_file)
        wandb.log({
            "epochs": e,
            "loss_tr": loss_tr,
            "lr": scheduler.get_last_lr()[0],
            "results": wandb.Image(f"{save_dir}/val_images/epoch{e}.png")})

        # Save the model and optimizer locally and to W&B
        torch.save({
            "model": model.cpu(),
            "optimizer": optimizer,
            "last_epoch": e,
        }, f"{save_dir}/{e}.pt")
        wandb.save(f"{save_dir}/{e}", base_path=f"{save_dir}", policy="end")
        model.to(device)

        scheduler.step()
