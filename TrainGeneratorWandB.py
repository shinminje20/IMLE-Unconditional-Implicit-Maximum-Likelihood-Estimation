import argparse
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import wandb

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
import torchvision.utils as tv_utils

from CAMNet import *
from Corruptions import Corruption
from Data import *
from Losses import *
from utils.Utils import *
from utils.UtilsColorSpace import *
from utils.UtilsNN import *

from functools import partial

def get_z_dims(args):
    """Returns a list of random noise dimensionalities for one sampling."""
    return [(args.map_nc + args.code_nc * r ** 2,) for r in args.res[:-1]]

mm = None
def get_z_gen(z_dims, bs, level=0, sample_method="normal", input=None, num_components=5,  **kwargs):
    """Returns a latent code for a model.

    Args:
    z_dims          -- list of tuples of shapes to generate
    bs              -- batch size to generate for
    level           -- the level to generate a shape for or all for 'all' to get
                        a list of codes, one for each level
    sample_method   -- the method to use to sample
    input           -- input for test-time sampling
    num_components  -- number of components for mixture-based sampling
    """
    if sample_method == "normal":
        if level == "all":
            return [torch.randn((bs,) + dim) for dim in z_dims]
        else:
            return torch.randn((bs,) + z_dims[level])
    elif sample_method == "mixture":
        global mm
        if mm is None:
            mm = [torch.rand(1, num_components, *dim) for dim in z_dims]
            mm = [nn.functional.normalize(m, dim=2) for m in mm]

        if input is None:
            idxs = torch.tensor(random.choices(range(num_components), k=bs))
        elif input == "show_components":
            idxs = torch.tensor([i % num_components for i in range(bs)])
        elif isinstance(input, torch.Tensor):
            idxs = input
        else:
            pass

        neg_ones = [[-1] * (1 + len(dim)) for dim in z_dims]
        if level == "all":
            means = [mm[level].expand(bs, *neg_ones[level])[torch.arange(bs), idxs] for level in range(len(mm))]
            return [m + torch.randn(m.shape) / num_components for m in means]
        else:
            means = mm[level].expand(bs, *neg_ones[level])[torch.arange(bs), idxs]
            return means + torch.randn(means.shape) / num_components
    else:
        raise NotImplementedError()

def get_new_codes(cx, y, model, z_gen, loss_fn, num_samples=16, sample_parallelism=16):
    """Returns a list of new latent codes found via hierarchical sampling.

    Args:
    cx          -- a BSxCxHxW tensor of corrupted images, on device
    model       -- model backbone. Must support a 'loi' argument and a tensor of
                    losses, one for each element in an input batch
    z_gen       -- function mapping from batch sizes and levels to z_dims
    sp          -- list of sample parallelisms, one for each level
    num_samples -- list of numbers of samples, one for each level
    """
    num_samples = make_list(num_samples, len(y))
    sample_parallelism = make_list(sample_parallelism, len(y))

    bs = len(cx)
    level_codes = z_gen(bs, level="all")
    with torch.no_grad():
        for level_idx in tqdm(range(len(num_samples)),
            desc="Sampling: levels",
            leave=False,
            dynamic_ncols=True):

            # Get inputs for sampling for the current level. We need to
            # store the least losses we have for each example, and to find
            # the level-specific number of samples [ns], sample parallelism
            # [sp], and shape to sample noise in [shape].
            old_codes = level_codes[:level_idx]
            least_losses = torch.ones(bs, device=device) * float("inf")
            ns = num_samples[level_idx]
            sp = min(ns, sample_parallelism[level_idx])

            # Handle arbitrary sample parallelism. If [sp] evenly divides
            # [ns], then we just run [ns // sp] tries. Otherwise, we run an
            # extra try where the sample parallelism is [ns % sp].
            if ns % sp == 0:
                iter_range = range(ns // sp)
                sps = make_list(sp, len(iter_range))
            else:
                iter_range = range(ns // sp + 1)
                sps = make_list(sp, length=ns // sp) + [ns % sp]

            for idx in tqdm(iter_range,
                desc="Sampling: iterations over level",
                leave=False,
                dynamic_ncols=True):

                # Get the sample parallelism for this trial. Then, get new
                # codes to sample for the CAMNet level currently being
                # sampled with while using the prior best old codes.
                sp = sps[idx]
                new_codes = z_gen(bs * sp, level=level_idx)
                test_codes = old_codes + [new_codes]

                # Compute loss for the new codes.
                outputs = model(cx, test_codes, loi=level_idx)
                losses = loss_fn(outputs, y[level_idx])

                # [losses] may have multiple values for each input example
                # due to using sample parallelism. Therefore, we find the
                # best-comuted loss for each example, giving a tensor of new
                # losses of the same size as [least_losses]. We do the same
                # with the newly sampled codes.
                _, idxs = torch.min(losses.view(bs, sp), axis=1)
                new_codes = new_codes.view((bs, sp) + new_codes.shape[1:])
                new_codes = new_codes[torch.arange(bs), idxs]
                losses = losses.view(bs, sp)[torch.arange(bs), idxs]

                # Update [level_codes] and [last_losses] to reflect new
                # codes that get least loss.
                change_idxs = losses < least_losses
                level_codes[level_idx][change_idxs] = new_codes[change_idxs]
                least_losses[change_idxs] = losses[change_idxs]

    return level_codes

def get_codes_in_chunks(cx, y, model, z_gen, loss_fn, num_samples=16,
    sample_parallelism=16, code_bs=128):
    """Returns a list of new latent codes found via hierarchical sampling with
    the batch dimension chunked to allow running larger batches.

    Args:
    cx          -- a BSxCxHxW tensor of corrupted images, on device
    model       -- model backbone. Must support a 'loi' argument and a tensor of
                    losses, one for each element in an input batch
    z_gen       -- function mapping from batch sizes and levels to z_dims
    sp          -- list of sample parallelisms, one for each level
    num_samples -- list of numbers of samples, one for each level
    code_bs     -- the size of each batch dimension chunk
    """
    def partition_into_batches(x, chunks):
        """Returns [x] split into [chunks] sections along each constituent
        tensor's zero dimension.
        """
        if isinstance(x, (list, tuple)):
            return [partition_into_batches(x_, chunks) for x_ in x]
        elif isinstance(x, torch.Tensor):
            return torch.tensor_split(x, chunks)
        else:
            raise ValueError()

    chunks = max(1, len(cx) // code_bs)
    cx = partition_into_batches(cx, chunks)
    y = partition_into_batches(y, chunks)

    level_codes = level_codes = z_gen(0, level="all")
    for cx_ys in tqdm(zip(cx, *y),
        total=chunks,
        desc="Sampling: chunks",
        leave=False,
        dynamic_ncols=True):

        chunk_codes = get_new_codes(cx_ys[0], cx_ys[1:], model, z_gen, loss_fn,
            num_samples=num_samples,
            sample_parallelism=sample_parallelism)
        level_codes = [torch.cat(c) for c in zip(level_codes, chunk_codes)]
    
    return level_codes

def validate(corruptor, model, z_gen, loader_eval, loss_fn, args):
    """Returns a list of lists, where each sublist contains first a ground-truth
    image and then [samples_per_image] images conditioned on that one.

    Args:
    corruptor   -- a corruptor to remove information from images
    model       -- a model to fix corrupted images
    z_gen       -- noise generator for [model]
    loader_eval -- dataloader over evaluation data
    loss_fn     -- loss function for one CAMNet level
    args        -- argparse arguments for the run

    Returns:
    results     -- 2D grid of images to show
    loss        -- loss for the returned images. Because it's computed over
                    only the last level, it will be much less than recorded
                    training loss.
    """
    results = []
    loss = 0
    with torch.no_grad():
        for x,y in tqdm(loader_eval, desc="Generating samples", leave=False, dynamic_ncols=True):
            bs = len(x)
            cx = corruptor(x)
            cx_expanded = cx.repeat_interleave(args.spi, dim=0)
            codes = z_gen(bs * args.spi, level="all", input="show_components")
            outputs = model(cx_expanded, codes, loi=-1)
            losses = loss_fn(outputs, y[-1])
            outputs = outputs.view(bs, args.spi, 3, args.res[-1], args.res[-1])

            idxs = torch.argsort(losses.view(bs, args.spi), dim=-1)
            outputs = outputs[torch.arange(bs).unsqueeze(1), idxs]
            images = [[s for s in samples] for samples in outputs]
            images = [[y_, c] + s for y_,c,s in zip(y[-1], cx, images)]

            loss += losses.mean().item()
            results += images

    return results, loss / len(loader_eval)

def get_args(args=None):
    P = argparse.ArgumentParser(description="CAMNet training")
    # Non-hyperparameter arguments. These aren't logged!
    P.add_argument("--wandb", choices=["disabled", "online", "offline"],
        default="online",
        help="disabled: no W&B logging, online: normal W&B logging")
    P.add_argument("--suffix", default="",
        help="optional training suffix")
    P.add_argument("--jobid", default=None, type=str,
        help="Variable for storing SLURM job ID")
    P.add_argument("--resume", type=str, default=None,
        help="a path or epoch number to resume from or nothing for no resuming")

    P.add_argument("--data_path", default=data_dir, type=str,
        help="path to where datasets are stored")
    P.add_argument("--spi", type=int, default=6,
        help="samples per image in logging, showing the model's diversity.")
    P.add_argument("--chunk_epochs", type=int, choices=[0, 1], default=0,
        help="whether to chunk by epoch. Useful for ComputeCanada, annoying otherwise.")
    P.add_argument("--gpus", type=int, default=[0, 1], nargs="+",
        help="GPU ids")
    P.add_argument("--code_bs", type=int, default=128,
        help="GPU ids")
    
    # Training hyperparameter arguments. These are logged!
    P.add_argument("--data", required=True, choices=datasets,
        help="data to train on")
    P.add_argument("--res", nargs="+", type=int, default=[64, 64, 64, 64, 128],
        help="resolutions to see data at")
    P.add_argument("--seed", type=int, default=0,
        help="random seed")
    P.add_argument("--proj_dim", default=1000, type=int,
        help="projection dimensionality")
    P.add_argument("--epochs", default=20, type=int,
        help="number of epochs (months) to train for")
    P.add_argument("--bs", type=int, default=64,
        help="batch size")
    P.add_argument("--mini_bs", type=int, default=8,
        help="batch size")
    P.add_argument("--ns", type=int, nargs="+", default=[128],
        help="number of samples for IMLE")
    P.add_argument("--ipc", type=int, default=10240,
        help="Effective gradient steps per set of codes. --ipc // --mini_bs is equivalent to num_days in the original CAMNet formulation")
    P.add_argument("--lr", type=float, default=1e-4,
        help="learning rate")
    P.add_argument("--color_space", choices=["rgb", "lab"], default="rgb",
        help="Color space to use during training")
    P.add_argument("--sp", type=int, default=[128], nargs="+",
        help="parallelism across samples during code training")

    P.add_argument("--sample_method", choices=["normal", "mixture"], default="normal",
        help="The method with which to sample latent codes")

    # Corruption hyperparameter arguments
    P.add_argument("--grayscale", default=0, type=float, choices=[0, .5, 1],
        help="grayscale corruption")
    P.add_argument("--mask_res", default=8, type=int,
        help="sidelength of image at which to do masking")
    P.add_argument("--mask_frac", default=0, type=float,
        help="fraction of pixels to mask")
    P.add_argument("--fill", default="zero", choices=["color", "zero"],
        help="how to fill masked out areas")

    # Model hyperparameter arguments
    P.add_argument("--code_nc", default=5, type=int,
        help="number of code channels")
    P.add_argument("--in_nc", default=3, type=int,
        help="number of input channels. SHOULD ALWAYS BE THREE.")
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
    P.add_argument("--act_type", default="leakyrelu", choices=["leakyrelu"],
        help="activation type")
    P.add_argument("--init_type", choices=["kaiming", "normal"],
        default="kaiming",
        help="NN weight initialization method")
    P.add_argument("--init_scale", type=float, default=.1,
        help="Scale for weight initialization")
    
    args = P.parse_args() if args is None else P.parse_args(args)
    args.levels = len(args.res) - 1
    args.ns = make_list(args.ns, length=args.levels)
    args.sp = make_list(args.sp, length=args.levels)

    # Make sure we won't break sampling.
    assert args.bs % len(args.gpus) == 0
    for ns,sp in zip(args.ns, args.sp):
        if not (ns * sp) % len(args.gpus) == 0:
            raise ValueError(f"number of samples * sample parallelism must be a multiple of the number of GPUS for each level")
    args.spi = args.spi - (args.spi % len(args.gpus))

    if not args.ipc % args.mini_bs == 0 or args.ipc // args.mini_bs == 0:
        raise ValueError(f"--ipc should be a multiple of --mini_bs")

    return args

if __name__ == "__main__":
    args = get_args()

    ############################################################################
    # Handle resuming.
    ############################################################################
    save_dir = generator_folder(args)
    if str(args.resume).isdigit():
        args.resume = int(args.resume) - 1
        if int(args.resume) == -1:
            resume_file = None
        elif os.path.exists(f"{save_dir}/{args.resume}.pt"):
            resume_file = f"{save_dir}/{args.resume}.pt"
        else:
            raise ValueError(f"File {save_dir}/{args.resume}.pt doesn't exist")
    elif isinstance(args.resume, str):
        resume_file = args.resume
    else:
        resume_file = None

    if resume_file is None:
        save_dir = generator_folder(args, ignore_conflict=False)
        cur_seed = set_seed(args.seed)

        # Setup the experiment. Importantly, we copy the experiment's ID to
        # [args] so that we can resume it later.
        args.run_id = wandb.util.generate_id()
        wandb.init(anonymous="allow", id=args.run_id, config=args,
            mode=args.wandb, project="isicle-generator")
        corruptor = Corruption(**vars(args))
        model = nn.DataParallel(CAMNet(**vars(args)), device_ids=args.gpus).to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        last_epoch = -1
    else:
        tqdm.write(f"Resuming from {resume_file}")
        resume_data = torch.load(resume_file)

        # Copy non-hyperparameter information from the current arguments to the
        # ones we're resuming
        curr_args = args
        args = resume_data["args"]
        args.data_path = curr_args.data_path
        args.gpus = curr_args.gpus
        args.chunk_epochs = curr_args.chunk_epochs
        args.wandb = curr_args.wandb
        save_dir = generator_folder(args)
        cur_seed = set_seed(resume_data["seed"])

        wandb.init(id=args.run_id, resume="must", mode=args.wandb,
            project="isicle-generator", config=args)

        model = resume_data["model"].to(device)
        optimizer = resume_data["optimizer"]
        corruptor = resume_data["corruptor"].to(device)
        last_epoch = resume_data["last_epoch"]
        scheduler = resume_data["scheduler"]

    # Set up the loss function
    loss_fn = nn.DataParallel(ResolutionLoss(), device_ids=args.gpus).to(device)

    # Set up the training and evaluation DataLoaders
    data_tr, data_eval = get_data_splits(args.data,
        eval_str="val",
        res=args.res,
        data_path=args.data_path)

    data_tr = GeneratorDataset(data_tr, get_gen_augs(args))

    # Get the evaluation data. We need to do this carefully so as to use
    # DataParallel and not have the data get dropped in the DataLoader.
    data_eval = GeneratorDataset(data_eval, get_gen_augs(args))
    eval_len = len(data_eval) // (args.spi + 2)
    eval_len = round_so_evenly_divides(eval_len, len(args.gpus))
    data_eval = Subset(data_eval, indices=range(0, len(data_eval), eval_len))

    loader_tr = DataLoader(data_tr, pin_memory=True, shuffle=True,
        batch_size=max(len(args.gpus), args.bs), num_workers=8, drop_last=True, **seed_kwargs(cur_seed))
    loader_eval = DataLoader(data_eval, shuffle=False,
        batch_size=max(len(args.gpus), args.mini_bs // args.spi), num_workers=8,
        drop_last=True)

    # Get a function that returns random codes given a level. We will use
    # this to do cool things with non-Gaussian sampling via the
    # [sample_method] input.
    z_gen = partial(get_z_gen,
        get_z_dims(args),
        sample_method=args.sample_method)

    ########################################################################
    # Construct the scheduler—strictly speaking, constructing it makes no sense
    # here, but we need to do it only if we're starting a new run.
    ########################################################################
    if resume_file is None:
        cycle_size = args.ipc // args.mini_bs
        scheduler = CosineAnnealingLR(optimizer,
            args.epochs * len(loader_tr) * cycle_size,
            eta_min=1e-8,
            last_epoch=max(-1, last_epoch * len(loader_tr) * cycle_size))

    tqdm.write(f"----- Final Arguments -----")
    tqdm.write(dict_to_nice_str(vars(args)))
    tqdm.write(f"----- Beginning Training -----")

    end_epoch = last_epoch + 2 if args.chunk_epochs else args.epochs
    cur_step = (last_epoch + 1) * len(loader_tr) * (args.ipc // args.mini_bs)
    for e in tqdm(range(last_epoch + 1, end_epoch),
        desc="Epochs",
        dynamic_ncols=True):
        
        for batch_idx,(x,ys) in tqdm(enumerate(loader_tr),
            desc="Batches",
            leave=False,
            dynamic_ncols=True,
            total=len(loader_tr)):
            batch_loss = 0
            
            ys = [y.to(device, non_blocking=True) for y in ys]
            cx = corruptor(x.to(device, non_blocking=True))         
            codes = get_codes_in_chunks(cx, ys, model, z_gen, loss_fn,
                num_samples=args.ns,
                sample_parallelism=args.sp,
                code_bs=args.code_bs)
            batch_dataset = CorruptedCodeYDataset(cx, codes, ys,
                expand_factor=args.ipc // args.bs)
            batch_loader = DataLoader(batch_dataset,
                batch_size=args.mini_bs,
                shuffle=True,
                num_workers=8)

            for idx,(cx, codes, ys) in tqdm(enumerate(batch_loader),
                desc="Minibatches",
                leave=False,
                dynamic_ncols=True,
                total=len(batch_loader)):

                cx = cx.to(device)
                codes = [c.to(device) for c in codes]
                ys = [y.to(device) for y in ys]

                fx = model(cx, codes, loi=None)                    
                loss = compute_loss_over_list(fx, ys, loss_fn)    
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                scheduler.step()
                batch_loss += loss.detach()
                cur_step += 1
                wandb.log({
                    "minibatch loss": loss.detach(),
                    "learning rate": get_lr(scheduler)[0]
                }, step=cur_step)
                
            batch_loss = batch_loss / args.ipc
            del x, codes, ys, loss, cx

            ####################################################################
            # Log data after each batch
            ####################################################################
            images_val, loss_val = validate(corruptor, model, z_gen,
                loader_eval, loss_fn, args)
            images_file = f"{save_dir}/val_images/step{e * len(loader_tr) + batch_idx}.png"
            save_image_grid(images_val, images_file)
            wandb.log({
                "validation loss": loss_val,
                "generated images": wandb.Image(images_file),
            }, step=cur_step)
            
            tqdm.write(f"Epoch {e:3}/{args.epochs} | batch {batch_idx:5}/{len(loader_tr)} | batch training loss {batch_loss.item():.5e} | lr {get_lr(scheduler)[0]:.5e} | loss_val {loss_val:.5e}")

            del images_val, loss_val

        save_checkpoint({"corruptor": corruptor.cpu(), "model": model.cpu(),
            "last_epoch": e, "args": args, "scheduler": scheduler,
            "optimizer": optimizer}, f"{save_dir}/{e}.pt")
        corruptor, model = corruptor.to(device), model.to(device)
