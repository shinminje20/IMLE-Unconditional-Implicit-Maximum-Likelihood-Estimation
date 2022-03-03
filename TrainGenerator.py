import argparse
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import wandb

import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

from CAMNet import CAMNet, get_z_dims
from Corruptions import Corruption
from Data import *
from utils.Utils import *
from utils.UtilsColorSpace import *
from utils.UtilsLoss import *
from utils.UtilsLPIPS import LPIPSFeats
from utils.UtilsNN import *

def check_args(args, data_tr_len):
    """Returns [args] if they are okay or raises an informative ValueError."""
    if not evenly_divides(args.bs, data_tr_len) or data_tr_len < args.bs:
        raise ValueError(f"--bs {args.bs} must be at most and evenly divide the length of the dataset {len(data_tr)}")
    if not evenly_divides(args.mini_bs, args.bs) or args.bs < args.mini_bs:
        raise ValueError(f"--mini_bs {args.mini_bs} must be at most and evenly divide --bs {args.bs}")
    if not evenly_divides(args.code_bs, args.bs) or args.bs < args.code_bs:
         raise ValueError(f"--code_bs {args.code_bs} must be at most and evenly divide --bs {args.bs}")

    for ns,sp in zip(args.ns, args.sp):
        if not evenly_divides(sp, ns) and not ns < sp:
            raise ValueError(f"--sp {args.sp} evenly divide --ns {args.ns} on all indices")
    tqdm.write(f"Training will take {int(len(data_tr) / args.mini_bs * args.ipcpe * args.epochs)} gradient steps and {args.epochs * len(data_tr)} different codes")

    return args

def get_images(corruptor, model, dataset, idxs=list(range(0, 60, 6)),
    samples_per_image=5, in_color_space="rgb", out_color_space="rgb"):
    """Returns a list of lists, where each sublist contains first a ground-truth
    image and then [samples_per_image] images conditioned on that one.

    Args:
    corruptor   -- a corruptor to remove information from images
    model       -- a model to fix corrupted images
    dataset     -- GeneratorDataset to load images from
    idxs        -- the indices to [dataset] to get images for
    """
    model = model.module
    idxs = list(range(len(dataset))) if idxs is None else idxs
    images_dataset = Subset(dataset, idxs)
    corrupted_data = CorruptedDataset(images_dataset, corruptor, bs=len(idxs))

    results = [[ys[-1], cx] for cx,ys in corrupted_data]
    results = lab2rgb_with_dims(results) if in_color_space == "lab" else results

    corrupted_data = ExpandedDataset(corrupted_data, samples_per_image)
    codes_data = ZippedDataset(*get_new_codes(get_z_dims(model),
        corrupted_data, model, "mse", ns=0))
    batch_dataset = ZippedDataset(codes_data, corrupted_data)

    with torch.no_grad():
        for idx,(codes,(cx,_)) in enumerate(batch_dataset):
            codes = [c.unsqueeze(0) for c in make_device(codes)]
            fx = model(cx.to(device).unsqueeze(0), codes, loi=-1,
                       in_color_space=in_color_space,
                       out_color_space=out_color_space)
            results[idx // samples_per_image].append(fx.cpu())

    return make_cpu(make_3dim(results))


def get_new_codes(z_dims, corrupted_data, backbone, loss_type, code_bs=6,
    ns=128, sp=2, in_color_space="rgb", out_color_space="rgb",
    proj_dim=None, verbose=1):
    """Returns a list of new latent codes found via hierarchical sampling. For
    a batch size of size BS, and N elements to [z_dims], returns a list of codes
    that where the ith code is of the size of the ith elmenent of [z_dims]
    expanded across the batch dimension.

    Args:
    z_dims          -- list of shapes describing a latent code
    corrupted_data  -- a Subset of the training dataset
    backbone    -- model backbone. Must support a 'loi' argument
    loss_fn     -- the means of determining distance. For inputs of size Nx...,
                    it should return a tensor of N losses.
    code_bs     -- batch size to test codes in
    num_samples -- number of times we try to find a better code for each image

    """
    tqdm.write(f"{ns}")
    loss_fn = get_unreduced_loss_fn(loss_type, proj_dim=proj_dim)
    bs = len(corrupted_data)
    sample_parallelism = make_list(sp, length=len(z_dims))
    num_samples = make_list(ns, length=len(z_dims))
    level_codes = [torch.randn((bs,)+z, device=device) for z in z_dims]
    loader = DataLoader(corrupted_data, batch_size=code_bs, num_workers=num_workers, pin_memory=True)

    for level_idx in tqdm(range(len(z_dims)), desc="Levels", leave=False, dynamic_ncols=True):
        least_losses = torch.ones(bs, device=device) * float("inf")

        ns = num_samples[level_idx]
        sp = min(ns, sample_parallelism[level_idx])
        shape = z_dims[level_idx]

        tqdm.write(f"ns {ns} | sp {sp} | sp1 {sample_parallelism[level_idx]}")

        for i in tqdm(range(ns // sp), desc="Sampling", leave=False, dynamic_ncols=True):
            for idx,(cx,ys) in enumerate(loader):
                start_idx, end_idx = code_bs * idx, code_bs * (idx + 1)
                least_losses_batch = least_losses[start_idx:end_idx]

                old_codes = [l[start_idx:end_idx] for l in level_codes[:level_idx]]
                new_codes = torch.randn((code_bs * sp,) + shape, device=device)
                test_codes = old_codes + [new_codes]

                with autocast():
                    with torch.no_grad():
                        fx = backbone(cx.to(device), test_codes, loi=level_idx,
                            in_color_space=in_color_space,
                            out_color_space=out_color_space)
                        ys = ys[level_idx].to(device)
                        losses = compute_loss(fx, ys, loss_fn, reduction="batch")

                    if sp > 1:
                        _, idxs = torch.min(losses.view(code_bs, sp), axis=1)
                        new_codes = new_codes.view((code_bs, sp) + new_codes.shape[1:])
                        new_codes = new_codes[torch.arange(code_bs), idxs]
                        losses = losses.view(code_bs, sp)[torch.arange(code_bs), idxs]

                change_idxs = losses < least_losses_batch
                level_codes[level_idx][start_idx:end_idx][change_idxs] = new_codes[change_idxs]
                least_losses[start_idx:end_idx][change_idxs] = losses[change_idxs]

            if verbose == 2:
                tqdm.write(f"    Processed {i * sp} samples | mean loss {torch.mean(least_losses):.5f}")

    return make_cpu(level_codes)

def one_epoch_imle(corruptor, model, optimizer, scheduler, dataset,
    loss_type="lpips", bs=1, mini_bs=1, code_bs=1, iters_per_code_per_ex=1,
    ns=1, sp=1, in_color_space="rgb", out_color_space="rgb", verbose=0,
    num_prints=10, proj_dim=None):
    """Returns a (corruptor, model, optimizer) tuple after training [model] and
    optionally [corruptor] for one epoch on data from [loader] via cIMLE.

    ****************************************************************************
    Note that in the typical terminology, a 'minibatch' and a 'batch' are
    synonymous, but here a 'minibatch' is a subset of a 'batch'.
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
    loss_fn = get_unreduced_loss_fn(loss_type, proj_dim=proj_dim)
    total_loss = 0
    print_iter = len(dataset) // num_prints
    rand_idxs = random.sample(range(len(dataset)), len(dataset))

    for batch_idx in tqdm(range(0, len(dataset), bs), desc="Batches", leave=False, dynamic_ncols=True):

        # (1) Get a dataset of corrupted images and their targets, (2) Get codes
        # for the corrupted images. This takes the place of the min() function
        # in IMLE, since we return the best (minimum loss) codes for each image
        # from the function, (3) Zip the codes and the corrupted images and
        # their targets together
        images_data = Subset(dataset, rand_idxs[batch_idx:batch_idx + bs])
        corrupted_data = CorruptedDataset(images_data, corruptor, bs=bs)
        codes_data = ZippedDataset(*get_new_codes(get_z_dims(model),
            corrupted_data, model, loss_type=loss_type, code_bs=code_bs,
            ns=ns, sp=sp, in_color_space=in_color_space,
            out_color_space=out_color_space, verbose=verbose))
        batch_dataset = ZippedDataset(codes_data, corrupted_data)
        loader = DataLoader(batch_dataset, batch_size=mini_bs, shuffle=True,
            num_workers=num_workers, pin_memory=True)

        for _ in tqdm(range(iters_per_code_per_ex), desc="Inner loop", leave=False, dynamic_ncols=True):
            for codes,(cx,ys) in tqdm(loader, desc="Minibatches", leave=False, dynamic_ncols=True):

                model.zero_grad(set_to_none=True)
                fx = model(cx.to(device), make_device(codes),
                    in_color_space=in_color_space,
                    out_color_space=out_color_space)
                loss = compute_loss(fx, make_device(ys), loss_fn, reduction="mean")
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

                total_loss += loss.item()

        model.zero_grad(set_to_none=True)
        scheduler.step()

        if verbose > 0 and batch_idx % print_iter == 0:
            tqdm.write(f"    current loss {loss.item():.5f} | lr {scheduler.get_last_lr()[0]:.5f}")

    model.zero_grad(set_to_none=True)
    loss = total_loss * mini_bs / iters_per_code_per_ex / len(dataset)
    return corruptor, model, optimizer, scheduler, loss

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="CAMNet training")
    P.add_argument("--wandb", default=1, choices=[0, 1], type=int,
        help="Whether to use W&B logging or not")
    P.add_argument("--data", default="cifar10", choices=datasets,
        help="data to train on")
    P.add_argument("--eval", default="val", choices=["cv", "val", "test"],
        help="data for validation")
    P.add_argument("--res", nargs="+", type=int, default=[64, 64, 64, 64, 128],
        help="resolutions to see data at")
    P.add_argument("--suffix", default="",
        help="optional training suffix")
    P.add_argument("--verbose", choices=[0, 1, 2], default=0, type=int,
        help="verbosity level")
    P.add_argument("--data_folder_path", default=f"{project_dir}/data", type=str,
        help="path to data if not in normal place")
    P.add_argument("--seed", type=int, default=0,
        help="random seed")
    P.add_argument("--resume", type=str, default=None,
        help="WandB run to resume from")

    P.add_argument("--arch", default="camnet", choices=["camnet"],
        help="Model architecture to use. Architecture hyperparameters are parsed later based on this")
    P.add_argument("--loss", default="resolution", choices=["mse", "lpips", "resolution"],
        help="loss function to use")
    P.add_argument("--proj_dim", default=None, type=int,
        help="projection dimensionality")
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
    P.add_argument("--ns", type=int, nargs="+", default=[512, 64, 32, 16],
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
    P.add_argument("--sp", type=int, default=[1], nargs="+",
        help="parallelism across samples during code training")
    P.add_argument("--gpus", type=int, default=[0], nargs="+",
        help="GPU ids")

    P.add_argument("--levels", default=4, type=int,
        help="number of CAMNet levels")
    P.add_argument("--code_nc", default=5, type=int,
        help="number of code channels")
    P.add_argument("--in_nc", default=3, type=int,
        help="number of input channels. SHOULD ALWAYS BE THREE")
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
    P.add_argument("--init_type", choices=["kaiming", "normal"], default="kaiming",
        help="NN weight initialization method")
    P.add_argument("--init_scale", type=float, default=.1,
        help="Scale for weight initialization")

    P.add_argument("--grayscale", default=1, type=int, choices=[0, 1],
        help="grayscale corruption")
    P.add_argument("--pix_mask_size", default=8, type=int,
        help="fraction of pixels to mask at 16x16 resolution")
    P.add_argument("--pix_mask_frac", default=.1, type=float,
        help="fraction of pixels to mask at 16x16 resolution")
    P.add_argument("--rand_illumination", default=0, type=float,
        help="amount by which the illumination of an image can change")
    P.add_argument("--fill", default="zero", choices=["color", "zero"],
        help="how to fill masked out areas")
    args = P.parse_args()

    ############################################################################
    # If resuming, resume; otherwise, validate arguments and construct training
    # objects.
    ############################################################################
    if not args.resume is None:
        run_id, resume_data = wandb_load(args.resume)
        set_seed(resume_data["seed"])
        data_folder_path = args.data_folder_path

        wandb.init(id=run_id, resume="must", project="isicle-generator")
        wandb.save("*.pt")
        model = resume_data["model"].to(device)
        optimizer = resume_data["optimizer"]
        corruptor = resume_data["corruptor"].to(device)
        last_epoch = resume_data["last_epoch"]
        args = resume_data["args"]
        args.resume = True
        args.data_folder_path = data_folder_path

        save_dir = generator_folder(args)
    else:
        set_seed(args.seed)
        args.ns = make_list(args.ns, length=args.levels)
        args.sp = make_list(args.sp, length=args.levels)

        save_dir = generator_folder(args)
        run_id = wandb.util.generate_id()
        wandb.init(anonymous="allow", id=run_id, project="isicle-generator",
            mode="online" if args.wandb else "disabled", config=args,
            name=save_dir.replace(f"{project_dir}/generators/", ""))

        model = CAMNet(**(vars(args)))
        model = nn.DataParallel(model, device_ids=args.gpus).to(device)
        core_params = [p for n,p in model.named_parameters() if not "map" in n]
        map_params = [p for n,p in model.named_parameters() if "map" in n]
        optimizer = Adam([{"params": core_params},
                          {"params": map_params, "lr": 1e-2 * args.lr}],
                          lr=args.lr, weight_decay=args.wd, betas=args.mm)
        corruptor = Corruption(**vars(args)).to(device)
        last_epoch = -1

    # Setup the datasets
    data_tr, data_eval = get_data_splits(args.data, args.eval, args.res,
        data_path=args.data_folder_path)
    data_tr = GeneratorDataset(data_tr, get_gen_augs())
    data_eval = GeneratorDataset(data_eval, get_gen_augs())

    # Setup the color spaces
    in_color_space = "lab" if "_lab" in args.data else "rgb"
    out_color_space = "rgb" if args.loss in ["lpips", "resolution"] or args.color_space == "rgb" else "lab"
    tqdm.write(f"Color space settings: input {in_color_space} | internal {args.color_space} | output {out_color_space}")

    # Setup the scheduler
    scheduler = CosineAnnealingLR(optimizer,
        args.epochs * (len(data_tr) // args.bs), last_epoch=last_epoch)

    ############################################################################
    # Begin training!
    ############################################################################
    if args.resume is None:
        results_file = f"{save_dir}/val_images/with_no_training.png"
        save_image_grid(get_images(corruptor, model, data_eval), results_file)
        wandb.log({"before_training": wandb.Image(results_file)})
    tqdm.write(f"----- Beginning Training -----")

    tqdm.write(f"1 {args.ns}")

    for e in tqdm(range(max(last_epoch + 1, 1), args.epochs + 1), desc="Epochs", dynamic_ncols=True):
        corruptor, model, optimizer, scheduler, loss_tr = one_epoch_imle(
            corruptor, model, optimizer, scheduler, data_tr,
            loss_type=args.loss, bs=args.bs, mini_bs=args.mini_bs,
            code_bs=args.code_bs, ns=args.ns,
            iters_per_code_per_ex=args.ipcpe, verbose=args.verbose,
            in_color_space=in_color_space, out_color_space=out_color_space,
            sp=args.sp, proj_dim=args.proj_dim)

        ########################################################################
        # After each epoch, log results and data
        ########################################################################
        lr = scheduler._last_lr[0]
        tqdm.write(f"loss_tr {loss_tr:.5f} | lr {lr:.5f}")
        images_file = f"{save_dir}/val_images/epoch{e}.png"
        save_image_grid(get_images(corruptor, model, data_eval), images_file)
        wandb.log({"loss_tr": loss_tr, "epochs": e, "lr": lr,
                   "results": wandb.Image(images_file)}, step=e)
        state_file = f"{save_dir}/{e}.pt"
        wandb_save({"run_id": run_id, "corruptor": corruptor.cpu(),
            "model": model.cpu(), "optimizer": optimizer, "last_epoch": e,
            "args": args}, state_file)
        model.to(device)
        corruptor.to(device)
