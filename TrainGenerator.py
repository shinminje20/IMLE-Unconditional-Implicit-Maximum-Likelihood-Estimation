import argparse
from tqdm import tqdm

import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset

from CAMNet import CAMNet

from Corruptions import get_non_learnable_batch_corruption
from Data import *
from utils.Utils import *
from utils.UtilsLPIPS import LPIPSFeats

################################################################################
# Loss finageling
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

    if isinstance(fx, list) and isinstance(y, list):
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

def get_new_codes(z_dims, data_subset, corruptor, backbone, loss_fn="mse",
    code_bs=6, num_samples=120, verbose=1):
    """Returns new latent codes via hierarchical sampling.

    Args:
    z_dims      -- list of shapes describing a latent code
    data_subset -- a Subset of the training dataset
    backbone    -- model backbone. Must support a 'loi' argument
    loss_fn     -- the means of determining distance. For inputs of size Nx...,
                    it should return a tensor of N losses.
    code_bs     -- batch size to test codes in
    num_samples -- number of times we try to find a better code for each image
    """
    loss_fn = get_loss_fn(loss_fn)
    bs = len(data_subset)
    level_codes = [torch.randn((bs,)+z, device=device) for z in z_dims]
    loader = DataLoader(data_subset, batch_size=code_bs, shuffle=False,
                        num_workers=num_workers)

    for level_idx in tqdm(range(len(z_dims)), desc="Resolutions", leave=False, dynamic_ncols=True):
        least_losses = torch.ones(bs, device=device) * float("inf")

        for i in tqdm(range(num_samples), desc="Sampling", leave=False, dynamic_ncols=True):
            for idx,(x,ys) in enumerate(loader):
                start_idx = code_bs * idx
                end_idx = code_bs * (idx + 1)
                least_losses_batch = least_losses[start_idx:end_idx]

                old_codes = [l[start_idx:end_idx] for l in level_codes[max(0, level_idx - 1):]]
                new_codes = torch.randn((code_bs,) + z_dims[level_idx], device=device)
                test_codes = old_codes + [new_codes]

                with torch.no_grad():
                    cx = corruptor(x.to(device))
                    fx = backbone(cx, test_codes, loi=level_idx)
                    losses = compute_loss(fx, ys[level_idx].to(device), loss_fn,
                        reduction="batch")

                change_idxs = losses < least_losses_batch
                level_codes[level_idx][start_idx:end_idx][change_idxs] = new_codes[change_idxs]
                least_losses[start_idx:end_idx][change_idxs] = losses[change_idxs]

            if verbose == 1 and i % 20 == 0:
                tqdm.write(f"    least_losses avg {torch.mean(least_losses):.5f}")

    return [l.cpu() for l in level_codes]

def get_images(corruptor, generator, dataset, idxs=[0], samples_per_image=1):
    """Returns a list of lists, where each sublist contains first a ground-truth
    image and then [samples_per_image] images conditioned on that one.

    Args:
    corruptor   -- a corruptor to remove information from images
    generator   -- a generator to fix corrupted images
    dataset     -- GeneratorDataset to load images from
    idxs        -- the indices to [dataset] to get images for
    """
    images_dataset = Subset(dataset, idxs)
    loader = DataLoader(images_dataset, batch_size=1, num_workers=num_workers)

    expanded_shape = (samples_per_image,) + images_dataset[0][0].shape
    corrupted = [corruptor(x.to(device)).cpu() for x,_ in loader]
    corrupted = [[c_ for c_ in c.expand(expanded_shape)] for c in corrupted]
    corrupted = XDataset(flatten(corrupted))

    codes_dataset = ZippedDataset(*get_new_codes(generator.get_z_dims(),
        corrupted, corruptor, generator, num_samples=0, verbose=False))
    dataset = ZippedDataset(corrupted, codes_dataset)
    loader = DataLoader(dataset, batch_size=1, num_workers=num_workers)

    results = [[]] * len(idxs)
    for idx,(cx,codes) in enumerate(loader):
        y = images_dataset[idx // samples_per_image][1][-1]
        results[idx // samples_per_image] = [y, cx.squeeze(0).cpu()]

    with torch.no_grad():
        for idx,(cx,codes) in enumerate(loader):
            y = images_dataset[idx // samples_per_image][1][-1]
            fx = generator(cx.to(device), [c.to(device) for c in codes])[-1]
            results[idx // samples_per_image].append(fx.squeeze(0).cpu())

    return results


def one_epoch_imle(corruptor, generator, optimizer, dataset, loss_fn="lpips",
    bs=1, mini_bs=1, code_bs=1, iters_per_code_per_ex=1000, num_samples=12,
    verbose=1):
    """Trains [generator] and optionally [corruptor] for one epoch on data from
    [loader] via cIMLE.

    ****************************************************************************
    Note that in the typical terminology, a 'minibatch' and a 'batch' are
    synonymous, and here a 'minibatch' is a subset of a 'batch'.
    ****************************************************************************

    Args:
    generator               -- a generative model that can undo corruptions
    optimizer               -- an optimizer for both the corruptor
                                (if wanted) and generator
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
        images_dataset = Subset(dataset, rand_idxs[batch_idx:batch_idx + bs])
        codes_dataset = ZippedDataset(*get_new_codes(generator.get_z_dims(),
            images_dataset, corruptor, generator, loss_fn=loss_fn,
            code_bs=code_bs, num_samples=num_samples, verbose=verbose))
        batch_dataset = ZippedDataset(codes_dataset, images_dataset)
        loader = DataLoader(batch_dataset, batch_size=mini_bs,
                            num_workers=num_workers, shuffle=True)

        inner_loop_iters = int(iters_per_code_per_ex * len(batch_dataset) / mini_bs)

        for _ in tqdm(range(inner_loop_iters), desc="inner loop", leave=False, dynamic_ncols=True):

            for codes,(x,ys) in tqdm(loader, desc="Minibatches", leave=False, dynamic_ncols=True):

                generator.zero_grad()
                cx = corruptor(x.to(device))
                fx = generator(cx, [c.to(device) for c in codes])
                loss = compute_loss(fx, [y.to(device) for y in ys], loss_fn,
                    reduction="mean", list_reduction="mean")
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        if verbose == 1:
            tqdm.write(f"    current loss {loss.item():.5f}")

    return corruptor, generator, optimizer, total_loss / len(loader)

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="CAMNet training")
    P.add_argument("--data", default="cifar10", choices=["strawberry", "cifar10", "camnet3", "camnet3_deci", "camnet3_centi", "camnet3_milli"],
        help="data to train on")
    P.add_argument("--eval", default="val", choices=["cv", "val", "test"],
        help="data for validation")
    P.add_argument("--res", nargs="+", required=True, type=int,
        default=[16, 32],
        help="resolutiosn to see data at")

    # Model hyperparameters are parsed later
    P.add_argument("--arch", default="camnet", choices=["camnet"],
        help="generator architecture to use.")

    ############################################################################
    # Corruption hyperparameters
    ############################################################################
    P.add_argument("--grayscale", default=1, type=int, choices=[0, 1],
        help="grayscale corruption")
    P.add_argument("--pix_mask_frac", default=.5, type=float,
        help="fraction of pixels to mask at 16x16 resolution")
    P.add_argument("--rand_illumination", default=.2, type=float,
        help="amount by which the illumination of an image can change")

    ############################################################################
    # Training hyperparameters
    ############################################################################
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

    P.add_argument("--suffix", default="",
        help="optional training suffix")
    P.add_argument("--options", default=[], nargs="+",
        help="options")
    P.add_argument("--verbose", choices=[0, 1], default=1,
        help="verbosity level")
    args, unparsed_args = P.parse_known_args()

    ############################################################################
    # Create the dataset and options, and check that various batch types have
    # okay sizes
    ############################################################################
    data_tr, data_eval = get_data_splits(args.data, args.eval, args.res)
    base_transform = get_gen_augs()
    data_tr = GeneratorDataset(data_tr, base_transform)
    data_eval = GeneratorDataset(data_eval, base_transform)

    if len(data_tr) < args.bs:
        tqdm.write(f"Setting batch size to {len(data_tr)} to match dataset length")
        args.bs = len(data_tr)
    if not evenly_divides(args.bs, len(data_tr)):
        raise ValueError(f"--bs {args.bs} must evenly divide the length of the dataset {len(data_tr)}")
    if not evenly_divides(args.mini_bs, args.bs) or args.bs < args.mini_bs:
        raise ValueError(f"--mini_bs {args.mini_bs} must be at most and evenly divide --bs {args.bs}")
    if not evenly_divides(args.code_bs, args.bs) or args.bs < args.code_bs:
         raise ValueError(f"--code_bs {args.code_bs} must be at most and evenly divide --bs {args.bs}")

    args.options = sorted([
        f"bs{args.bs}",
        f"code_bs{args.code_bs}",
        f"grayscale{args.grayscale}",
        f"ipcpe{args.ipcpe}",
        f"loss{args.loss}",
        f"lr{args.lr}",
        f"mini_bs{args.mini_bs}",
        f"mm{'_'.join([str(m) for m in list(args.mm)])}",
        f"pix_mask_frac{args.pix_mask_frac}"
        f"wd{args.wd}",
    ])

    ############################################################################
    # Create the corruption
    ############################################################################
    corruptor = get_non_learnable_batch_corruption(
        grayscale=args.grayscale,
        rand_illumination=args.rand_illumination,
        pixel_mask_frac=args.pix_mask_frac)

    ############################################################################
    # Create the generator and its optimizer
    ############################################################################
    if args.arch == "camnet":
        camnet_args, _ = CAMNet.parse_args_to_dict(unparsed_args, args.res)
        generator = CAMNet(**camnet_args).to(device)
        core_params = [v for name,v in generator.named_parameters()
                               if not "map" in name]
        map_params = [v for name,v in generator.named_parameters()
                               if "map" in name]
        optimizer = Adam([{"params": core_params},
                           {"params": map_params, "lr": 1e-2 * args.lr}],
                          lr=args.lr, weight_decay=args.wd, betas=args.mm)
    else:
        raise ValueError(f"Unknown architecture '{args.arch}'")

    ############################################################################
    # Set up remaining training utilities
    ############################################################################
    last_epoch = -1
    scheduler = CosineAnnealingLinearRampLR(optimizer, args.epochs,
        args.n_ramp,
        last_epoch=last_epoch)

    ############################################################################
    # Begin training!
    ############################################################################

    for e in tqdm(range(max(last_epoch + 1, 1), args.epochs + 1), desc="Epochs", dynamic_ncols=True):
        corruptor, generator, optimizer, loss_tr = one_epoch_imle(
            corruptor,
            generator,
            optimizer,
            data_tr,
            loss_fn=args.loss_fn,
            bs=args.bs,
            mini_bs=args.mini_bs,
            code_bs=args.code_bs,
            num_samples=args.num_samples,
            iters_per_code_per_ex=args.ipcpe,
            verbose=args.verbose)

        results = get_images(corruptor, generator, data_eval,
            idxs=random.sample(range(len(data_eval)), 10),
            samples_per_image=5)
        save_images_grid(results, f"{new_camnet_folder(args)}/generated_images/epoch{e}.png")
        tqdm.write(f"loss_tr {loss_tr:.5f} | lr {scheduler.get_last_lr()[0]:.5e}")
        scheduler.step()
