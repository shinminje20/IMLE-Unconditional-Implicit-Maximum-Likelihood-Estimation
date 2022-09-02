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
from torch.utils.data import Subset
from torch.cuda.amp import autocast, GradScaler
import torchvision.utils as tv_utils

from ConditionalIMLE import KorKMinusOne
from DatasetNewCode import Dataset_new_code
# from IMLEDataLoader import IMLEDataLoade
from Generator import Generator
from CAMNet import *
from Corruptions import Corruption
from Data import *
from Losses import *
from utils.Utils import *
from utils.UtilsColorSpace import *
from utils.UtilsNN import *

from functools import partial
import numpy as np

def get_z_dims(args):
    """Returns a list of random noise dimensionalities for one sampling."""
    return [(64,)] #TODO hard code need to be updated

# def get_z_dims(args):
#     """Returns a list of random noise dimensionalities for one sampling."""
# # =-=-=-=-=-=-=-=-=-=-=-=-=-=-==-
# # args.map_nc:  128
# # args.code_nc:  5
# # r ** 2:  1024
# # r:  32
# # 5048
# # =-=-=-=-=-=-=-=-=-=-=-=-=-=-==-
#     return [(args.map_nc + args.code_nc * r ** 2,) for r in args.res[:-1]]

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
            mm = torch.rand(1, num_components, *z_dims[0])
            mm = nn.functional.normalize(mm, dim=2)

        if input is None:
            idxs = torch.tensor(random.choices(range(num_components), k=bs))
        elif input == "show_components":
            idxs = torch.tensor([i % num_components for i in range(bs)])
        elif isinstance(input, torch.Tensor):
            idxs = input
        else:
            pass

        neg_ones = [-1] * (1 + len(z_dims[0]))

        means = mm.expand(bs, *neg_ones)[torch.arange(bs), idxs]
        return means + torch.randn(means.shape)
    else:
        raise NotImplementedError()

def validate(model, z_gen, loader_eval, loss_fn, alpha, depth, current_depth, latent_size, args):
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
    loss        -- average (LPIPS, MSE, Resolution) losses for the images.
                    Because it's computed over only the last level, the
                    Resolution loss will be less than recorded training loss
    # function to write 
    """
    results = []
    lpips_loss = 0
    with torch.no_grad():
        for iteartion, y in tqdm(enumerate(loader_eval), desc="Validation", leave=False, dynamic_ncols=True):
            # y.to(device)
            y = progressive_down_sampling(y, depth, current_depth, alpha)

            bs = y.shape[0]
            # noise = torch.randn(bs, latent_size).to(device)
            noise = z_gen(bs, input="show_components")
            # Note:
            # 1. whether the image is faithful to mixture component: mapping with 
            # 2. 
            # print("latent_size: ", latent_size)
            # print("noise.shape: ", noise.shape)
            # print("batch_size: ", batch_size)
            # print("current_depth: ", current_depth)
            outputs = model(noise, current_depth, alpha)
            
            # print("outputs.shape: ", outputs.shape)
            
            # model(noise, 3, alpha) noise = get_z_gen([(batch_size,)], batch_size, sample_method="mixture")
            lpips_loss_ = loss_fn(outputs, y, 'batch')
            outputs = outputs.view(bs, 1, 3, y.shape[2], y.shape[3])

            idxs = torch.argsort(lpips_loss_.view(bs, 1), dim=-1)
            # print("outpus.shape: ", outputs.shape)
            # print("lpips_loss_.shape: ", lpips_loss_.shape)
            # print("lpips_loss_.view(bs, ).shape: ", lpips_loss_.view(bs, ).shape)
            # idxs = torch.argsort(lpips_loss_.view(bs, ), dim=-1)
            outputs = outputs[torch.arange(bs).unsqueeze(1), idxs]
            # print("outpus.shape: ", outputs.shape)
            images = [[s for s in samples] for samples in outputs]
            images = [[y_] + s for y_, s in zip(y, images)]
            
            lpips_loss += lpips_loss_.mean()
            results += images
    return results, lpips_loss / len(loader_eval)

# function specification.
# 
def get_args(args=None):
    P = argparse.ArgumentParser(description="CAMNet training")
    # Non-hyperparameter arguments. These aren't logged!
    P.add_argument("--wandb", choices=["disabled", "online", "offline"],
        default="online",
        help="disabled: no W&B logging, online: normal W&B logging")
    P.add_argument("--suffix", default="",
        help="optional training suffix")
    P.add_argument("--job_id", default=None, type=str,
        help="Variable for storing SLURM job ID")
    P.add_argument("--uid", default=None, type=str,
        help="Unique identifier for the run. Should be specified only when resuming, as it needs to be generated via WandB otherwise")
    P.add_argument("--resume", type=str, default=None,
        help="a path or epoch number to resume from or nothing for no resuming")

    P.add_argument("--data_path", default=data_dir, type=str,
        help="path to where datasets are stored")
    P.add_argument("--num_val_images", type=int, default=10,
        help="Number of images to use for validation")
    P.add_argument("--spi", type=int, default=6,
        help="samples per image in logging, showing the model's diversity.")
    P.add_argument("--chunk_epochs", type=int, choices=[0, 1], default=0,
        help="whether to chunk by epoch. Useful for ComputeCanada, annoying otherwise.")
    P.add_argument("--gpus", type=int, default=[0, 1], nargs="+",
        help="GPU ids")
    P.add_argument("--code_bs", type=int, default=2,
        help="GPU ids")

    # Training hyperparameter arguments. These are logged!
    P.add_argument("--data_tr", type=is_valid_data, required=True,
        help="data to train on")
    P.add_argument("--data_val", type=is_valid_data, required=True,
        help="data to train on")
    P.add_argument("--res", nargs="+", type=int, default=[64, 64, 64, 64, 128],
        help="resolutions to see data at")
    P.add_argument("--alpha", type=float, default=.1,
        help="Amount of weight on MSE loss")
    P.add_argument("--seed", type=int, default=0,
        help="random seed")
    P.add_argument("--proj_dim", default=1000, type=int,
        help="projection dimensionality")
    P.add_argument("--epochs", nargs="+", default=[4,4,4,4,8,16,32,64,64], type=int,
        help="number of epochs (months) to train for")
    P.add_argument("--outer_loops", default=20, type=int,
        help="number of outer_loops to train for")
    P.add_argument("--bs", nargs="+", type=int, default=[128, 128, 128, 64, 32, 16, 8, 4, 2],
        help="batch size")
    P.add_argument("--mini_bs", type=int, default=8,
        help="batch size")
    P.add_argument("--ns", type=int, nargs="+", default=[128],
        help="number of samples for IMLE")
    P.add_argument("--ipc", type=int, default=10240,
        help="Effective gradient steps per set of codes. --ipc // --mini_bs is equivalent to num_days in the original CAMNet formulation")
    P.add_argument("--lr", nargs="+", type=float, default=[0.001,0.001,0.001,0.001,0.001, 0.0015, 0.002, 0.003],
        help="learning rate")
    P.add_argument("--color_space", choices=["rgb", "lab"], default="rgb",
        help="Color space to use during training")
    P.add_argument("--sp", type=int, default=[128], nargs="+",
        help="parallelism across samples during code training")
    P.add_argument("--subsample_size", default=None, type=int,
        help="number of subsample data ")
    P.add_argument("--num_iteration", default=1, type=int,
        help="number of subsample data ")
    P.add_argument("--num_components", default=5, type=int,
        help="number of components in mixture noise ")
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
    # assert args.bs % len(args.gpus) == 0
    for ns,sp in zip(args.ns, args.sp):
        if not (ns * sp) % len(args.gpus) == 0:
            raise ValueError(f"number of samples * sample parallelism must be a multiple of the number of GPUS for each level")
    args.spi = args.spi - (args.spi % len(args.gpus))

    assert args.code_bs >= len(args.gpus)

    if not args.ipc % args.mini_bs == 0 or args.ipc // args.mini_bs == 0:
        raise ValueError(f"--ipc should be a multiple of --mini_bs")

    args.uid = wandb.util.generate_id() if args.uid is None else args.uid
    return args

def progressive_down_sampling(real_batch, depth, curr_depth, alpha):
        """
        private helper for down_sampling the original images in order to facilitate the
        progressive growing of the layers.
        :param real_batch: batch of real samples
        :param depth: depth at which training is going on
        :param alpha: current value of the fade-in alpha
        :return: real_samples => modified real batch of samples
        """

        from torch.nn import AvgPool2d
        from torch.nn.functional import interpolate


        # down_sample the real_batch for the given depth
        # print("depth, curr_depth,: ", depth, curr_depth)
        down_sample_factor = int(np.power(2, depth - curr_depth - 1))
        prior_down_sample_factor = max(int(np.power(2, depth - curr_depth)), 0)

        ds_real_samples = AvgPool2d(down_sample_factor)(real_batch)

        if curr_depth > 3:
            # print(" curr_depth > 3")
            prior_ds_real_samples = interpolate(AvgPool2d(prior_down_sample_factor)(real_batch), scale_factor=2)
        else:
            # print(" curr_depth == 3")
            prior_ds_real_samples = ds_real_samples

        # real samples are a combination of ds_real_samples and prior_ds_real_samples
        # print("ds_real_samples: ", ds_real_samples.shape)
        # print("prior_ds_real_samples: ", prior_ds_real_samples.shape)
 
        real_samples = (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)

        # return the so computed real_samples
        return real_samples

def get_new_codes(ys, model, z_gen, loss_fn, num_samples=16, sample_parallelism=16):
    """Returns a list of new latent codes found via hierarchical sampling.

    Args:
    cx          -- a BSxCxHxW tensor of corrupted images, on device
    model       -- model backbone. Must support a 'loi' argument and a tensor of
                    losses, one for each element in an input batch
    z_gen       -- function mapping from batch sizes and levels to z_dims
    sp          -- list of sample parallelisms, one for each level
    num_samples -- list of numbers of samples, one for each level
    """
    num_samples = make_list(num_samples, len(ys))
    sample_parallelism = make_list(sample_parallelism, len(ys))

    bs = len(ys)
    level_codes = z_gen(bs)
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
                outputs = model(test_codes, loi=level_idx)
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

def get_codes_in_chunks(data, model, z_gen, loss_fn, num_samples=16,
    sample_parallelism=16, code_bs=128):
    """Returns a list of new latent codes found via hierarchical sampling with
    the batch dimension chunked to allow running larger batches.
    Args:
    data        -- GeneratorDataset, or Subset thereof
    model       -- model backbone. Must support a 'loi' argument and a tensor of
                    losses, one for each element in an input batch
    z_gen       -- function mapping from batch sizes and levels to z_dims
    sp          -- list of sample parallelisms, one for each level
    num_samples -- list of numbers of samples, one for each level
    code_bs     -- the size of each batch dimension chunk
    """
    level_codes = z_gen(len(data))
    loader = DataLoader(data,
        batch_size=code_bs,
        pin_memory=True,
        num_workers=24,
        drop_last=False)

    corrupted_images = []
    targets_images = None
    
    for idx, ys in tqdm(enumerate(loader),
        desc="Sampling chunks of batch",
        total=len(loader),
        leave=False,
        dynamic_ncols=True):
        ys = [y.to(device, non_blocking=True) for y in ys]
        # cx = corruptor(x.to(device, non_blocking=True))
        chunk_codes = get_new_codes(ys, model, z_gen, loss_fn,
            num_samples=num_samples,
            sample_parallelism=sample_parallelism)
        
        indices = range(idx * code_bs, min(len(data), (idx+1) * code_bs))
        indices = torch.tensor(indices)
        for level_idx in range(len(chunk_codes)):
            level_codes[level_idx][indices] = chunk_codes[level_idx]
        
        # Save the resulting images
        # corrupted_images.append(cx.cpu())
        if targets_images is None:
            targets_images = [[y.cpu()] for y in ys]
        else:
            for t,y in zip(targets_images, ys):
                t.append(y.cpu())

    # corrupted_images = torch.cat(corrupted_images, axis=0)
    targets_images = [torch.cat(t, axis=0) for t in targets_images]
    
    return level_codes, targets_images

if __name__ == "__main__":
    args = get_args()

    ############################################################################
    # Handle resuming.
    ############################################################################
    save_dir = generator_folder(args)
    print("save directory : ", save_dir)
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
        save_dir = generator_folder(args)
        cur_seed = set_seed(args.seed)

        # Setup the experiment. Importantly, we copy the experiment's ID to
        # [args] so that we can resume it later.
        args.run_id = wandb.util.generate_id()
        wandb.init(anonymous="allow", id=args.uid, config=args,
            mode=args.wandb, project="IMLE-StyleGan",
            name=save_dir.replace(f"{project_dir}/generators/", ""))
        corruptor = Corruption(**vars(args))
        model = nn.DataParallel(Generator(num_channels=3,
                                latent_size=args.res[-1],
                                dlatent_size=args.res[-1],
                                resolution=args.res[-1],
                             structure='linear',
                             **seed_kwargs(cur_seed))).to(device)
        # optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        last_loop = -1
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
        last_loop = resume_data["last_loop"]
        scheduler = resume_data["scheduler"]
    
    depth = int(np.log2(args.res[-1])) - 1
    latent_size = args.res[-1]
    start_depth = 3
    # Set up the loss function
    loss_fn = nn.DataParallel(UnconditionalIMLELoss(alpha=args.alpha), device_ids=args.gpus).to(device)

    ############################################################################
    # Set up the Datasets and DataLoaders. We need to to treat datasets where
    # the distribution is the different for training and validation splits
    # (eg. miniImagenet) differently from those where it's the same. We're more
    # interested in within-distribution learning.
    ############################################################################
    data_tr, data_val = get_imagefolder_IMLE_data(args.data_tr, args.data_val,
        res=args.res,
        data_path=args.data_path)
    
    data_tr = IMLEDataset(data_tr, get_gen_augs(args))
    # If args.data_val is 'cv', then we need to split it off from the training
    # data. If it's its own dataset, we can use each dataset directly. However,
    # we still need to select args.num_val_images from it.
    if args.data_val == "cv":
        step = int((len(data_tr) / args.num_val_images) + .5)
        idxs_val = {idx for idx in range(0, len(data_tr), step)}

        if len(idxs_val) == len(data_tr):
            raise ValueError(f"Too many validation images selected; no data is left for training. Reduce --num_val_images to below {len(data_tr) // 2}")

        idxs_tr = [idx for idx in range(len(data_tr)) if not idx in idxs_val]
        data_tr = Subset(data_tr, indices=idxs_tr)
        data_val = Subset(data_tr, indices=list(idxs_val))
    else:
        data_val = IMLEDataset(data_val, get_gen_augs(args))
        step = int((len(data_val) / args.num_val_images) + .5)
        # idxs_val = {idx for idx in range(0, len(data_val), step)}
        # data_val = Subset(data_val, indices=list(idxs_val))


    # Get a function that returns random codes given a level. We will use
    # this to do cool things with non-Gaussian sampling via the
    # [sample_method] input.
    z_gen = partial(get_z_gen, [(latent_size,)],
        sample_method=args.sample_method, num_components=args.num_components)
    
    ########################################################################
    # Construct the scheduler—strictly speaking, constructing it makes no sense
    # here, but we need to do it only if we're starting a new run.
    ########################################################################

    tqdm.write(f"----- Final Arguments -----")
    tqdm.write(dict_to_nice_str(vars(args)))
    tqdm.write(f"----- Beginning Training -----")

    # end_epoch = last_epoch + 2 if args.chunk_epochs else args.epochs
    # cur_step = (last_epoch + 1) * len(loader_tr) * (args.ipc // args.mini_bs)

    

    # loader_tr = IMLEDataLoader(data_tr, model, z_gen, loss_fn, args.ns, args.sp, args.code_bs,
    #                 subsample_size=args.subsample_size,
    #                 num_iteration=args.num_iteration,
    #                 pin_memory=True,
    #                 shuffle=True,
    #                 batch_size=max(len(args.gpus), args.bs),
    #                 num_workers=8,
    #                 drop_last=True)
    
    # if resume_file is None:
    #     k_or_k_minus_one = KorKMinusOne(range(len(data_tr)), shuffle=True)
    #     scheduler = CosineAnnealingLR(optimizer,
    #         args.outer_loops * args.num_iteration,
    #         eta_min=1e-8,
    #         last_epoch=max(-1, last_loop * args.num_iteration))

    
    
    # 1. iterate thru different resolution, 
    # 2. torch.cat with those different resolustion samples
    # 3. Train 
    end_loop = last_loop + 2 if args.chunk_epochs else args.outer_loops
    cur_step = (last_loop + 1) * args.num_iteration
    tqdm.write(f"LOG: Running loops indexed {last_loop + 1} to {end_loop}")
    

    loader_eval = DataLoader(data_val,
        shuffle=False,
        batch_size=18,
        num_workers=8,
        drop_last=True)

    # if resume_file is None:
    #         model = nn.DataParallel(Generator(num_channels=3,
    #                             resolution=512,
    #                          structure='linaer',
    #                          **seed_kwargs(cur_seed))).to(device)
    # else:
    #     model = resume_data["model"].to(device)

    model.train()
    cur_step = 0    
    for current_depth in tqdm(range(start_depth, depth), 
                                desc="Depths",
                                dynamic_ncols=True):
        
        
        batch_size = args.bs[current_depth]
        loader = DataLoader(data_tr,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=8,
                            drop_last=True,
                            pin_memory=True
                            )
        optimizer = AdamW(model.parameters(), lr=args.lr[current_depth], weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer,
            args.epochs[current_depth] * len(loader),
            eta_min=args.lr[current_depth],
            last_epoch=max(-1, last_loop * len(loader)))
        
        ticker = 1
        
        for loop in tqdm(range(args.epochs[current_depth]),
            desc="OuterLoops",
            dynamic_ncols=True):


            total_batches = len(loader)

            fade_point = int((1 /2 * args.epochs[current_depth] * total_batches))

            for batch_idx, ys in tqdm(enumerate(loader),
                desc="Batches",
                leave=False,
                dynamic_ncols=True,
                total=len(loader)):
                
                alpha = ticker / fade_point if ticker <= fade_point else 1
                
                batch_loss = 0
                images = ys
                labels = None
                
                images = images.to(device)
                
                # noise = torch.randn(images.shape[0], latent_size).to(device)
                noise = z_gen(images.shape[0])
                
                real_samples = progressive_down_sampling(images, depth, current_depth, alpha)

                # generate fake samples:
                generated_samples = model(noise, current_depth, alpha)
                del noise
                # Change this implementation for making it compatible for relativisticGAN                
                loss = loss_fn(generated_samples, real_samples, 'mean')

                # optimize the generator
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                # Gradient Clipping
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.)
                scheduler.step()

                cur_step += 1
                ticker += 1
                wandb.log({
                    "batch loss": loss,
                    "learning rate": get_lr(scheduler)[0]
                }, step=cur_step)
                
                
                del ys, loss, batch_loss
                
                ####################################################################
                # Log data after each epoch
                ####################################################################
            
            images_val, lpips_loss_val = validate(model, z_gen, loader_eval, loss_fn, alpha, depth, current_depth, latent_size, args)
            images_file = f"{save_dir}/val_images/res{np.power(2,current_depth+2)}/step{cur_step}.png"
            save_image_grid(images_val, images_file)
            wandb.log({
                "Loop_LPIPS loss": lpips_loss_val,
                "Loop_generated images": wandb.Image(images_file),
            }, step=cur_step)

            tqdm.write(f"Loop {loop:3}/{args.epochs[current_depth]} | Loop_lr {get_lr(scheduler)[0]:.5e} | Loop_lpips_loss_val {lpips_loss_val:.5e}")

            del images_val, lpips_loss_val

                # save_checkpoint({"model": model.cpu(),
                #     "last_loop": loop, "args": args, "scheduler": scheduler,
                #     "optimizer": optimizer, "k_or_k_minus_one": k_or_k_minus_one}, f"{save_dir}/{loop}.pt")
                # model = model.to(device)
