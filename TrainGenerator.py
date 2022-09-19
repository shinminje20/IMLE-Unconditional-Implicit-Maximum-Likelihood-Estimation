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

from IMLEDataLoader import IMLEDataLoader, KorKMinusOne
from Generator import Generator
from CAMNet import *
from Data import *
from Losses import *
from utils.Utils import *
from utils.UtilsColorSpace import *
from utils.UtilsNN import *

from functools import partial
import numpy as np

mm = None
def get_z_gen(z_dims, bs, level=0, sample_method="normal", input=None, num_components=5,  **kwargs):
    """Returns a latent code for a model.

    Args:
    z_dims          -- list of tuples of shapes to generate
    bs              -- batch size to generate for
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

        means = mm[level].expand(bs, *neg_ones[level])[torch.arange(bs), idxs]
        return means + torch.randn(means.shape)
    else:
        raise NotImplementedError()

def validate(model, z_gen, loss_fn, alpha, depth, current_depth, latent_size, args):
    """Returns a list of lists, where each sublist contains first a ground-truth
    image and then [samples_per_image] images conditioned on that one.

    Args:
    model       -- a model
    z_gen       -- noise generator for [model]
    loss_fn     -- loss function for one CAMNet level
    args        -- argparse arguments for the run

    Returns:
    results     -- 2D grid of images to show
    loss        -- average (LPIPS) losses for the images.
                    Because it's computed over only the last level, the
                    Resolution loss will be less than recorded training loss
    # function to write 
    """
    results = []
    with torch.no_grad():

        noise = z_gen(args.num_components * args.spi, input="show_components")
        outputs = model(noise, current_depth, alpha)
        resolution = int(np.power(2, current_depth + 2))
        outputs = outputs.view(args.num_components, args.spi, 3, resolution, resolution)
        # outputs = outputs.view(1, 1, 3, resolution, resolution)
        images = [[s for s in samples] for samples in outputs]
        images = [s for s in images]
                
        results += images
    return results

def get_args(args=None):
    P = argparse.ArgumentParser(description="CAMNet training")
    # Non-hyperparameter arguments. These aren't logged!
    P.add_argument("--wandb", choices=["disabled", "online", "offline"],
        default="online",
        help="disabled: no W&B logging, online: normal W&B logging")
    P.add_argument("--job_id", default=None, type=str,
        help="Variable for storing SLURM job ID")
    P.add_argument("--uid", default=None, type=str,
        help="Unique identifier for the run. Should be specified only when resuming, as it needs to be generated via WandB otherwise")
    P.add_argument("--suffix", default="",
        help="optional training suffix")
    P.add_argument("--data_path", default=data_dir, type=str,
        help="path to where datasets are stored")
    P.add_argument("--spi", type=int, default=6,
        help="samples per image in logging, showing the model's diversity.")
    P.add_argument("--gpus", type=int, default=[0, 1], nargs="+",
        help="GPU ids")
    P.add_argument("--code_bs", type=int, default=2,
        help="GPU ids")

    # Training hyperparameter arguments. These are logged!
    P.add_argument("--data_tr", type=is_valid_data, required=True,
        help="data to train on")
    P.add_argument("--data_val", type=is_valid_data, required=True,
        help="data to train on")
    P.add_argument("--res", nargs="+", type=int, default=[256],
        help="resolutions to see data at")
    P.add_argument("--alpha", type=float, default=.1,
        help="Amount of weight on MSE loss")
    P.add_argument("--seed", type=int, default=0,
        help="random seed")
    P.add_argument("--outer_loops", default=20, type=int,
        help="number of outer_loops to train for")
    P.add_argument("--bs", type=int, default=512,
        help="batch size")
    P.add_argument("--ns", type=int, nargs="+", default=[128],
        help="number of samples for IMLE")
    P.add_argument("--lr", nargs="+", type=float, default=[1e-4, 0.5e-3, 1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3],
        help="learning rate")
    # [3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3]
    # [3.5e-4, 4e-4, 4.5e-4, 5e-4, 1e-3, 1.5e-3, 2e-3]
    # [3e-3, 2.5e-3, 2e-3, 1.5e-3, 1e-3, 0.5e-3, 1e-4]
        
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

    args = P.parse_args() if args is None else P.parse_args(args)
    args.levels = len(args.res) - 1
    args.ns = make_list(args.ns, length=1)
    args.sp = make_list(args.sp, length=1)

    # Make sure we won't break sampling.
    assert args.bs % len(args.gpus) == 0
    for ns,sp in zip(args.ns, args.sp):
        if not (ns * sp) % len(args.gpus) == 0:
            raise ValueError(f"number of samples * sample parallelism must be a multiple of the number of GPUS for each level")
    args.spi = args.spi - (args.spi % len(args.gpus))

    assert args.code_bs >= len(args.gpus)

    args.uid = wandb.util.generate_id() if args.uid is None else args.uid
    return args

if __name__ == "__main__":
    args = get_args()

    # Depth and latent size is determined by input resolution.
    depth = int(np.log2(args.res[-1])) - 1
    latent_size = args.res[-1]
    start_depth = 0
    
    # Calculate outer_loops for different resolutions.
    outer_loops = [ args.outer_loops // int(np.power(2, (depth - i)))  for i in range(depth, start_depth, -1)]
    outer_loops.reverse()
    args.outer_loops = outer_loops

    save_dir = mixture_generator_folder(args)
    print("save directory : ", save_dir)

    cur_seed = set_seed(args.seed)

    args.run_id = wandb.util.generate_id()

    wandb.init(anonymous="allow", id=args.uid, config=args,
        mode=args.wandb, project="IMLE-StyleGan",
        name=save_dir.replace(f"{project_dir}/generators/", ""))

    model = nn.DataParallel(Generator(num_channels=3,
                            latent_size=args.res[-1],
                            dlatent_size=args.res[-1],
                            resolution=args.res[-1],
                            structure='linear',
                            **seed_kwargs(cur_seed))).to(device)

    # Set up the loss function
    loss_fn = nn.DataParallel(UnconditionalIMLELoss(), device_ids=args.gpus).to(device)

    data_tr, data_val = get_imagefolder_IMLE_data(args.data_tr, args.data_val,
        res=args.res,
        data_path=args.data_path)
    
    data_tr = IMLEDataset(data_tr, get_gen_augs(args))

    # Get a function that returns random codes. We will use
    # this to do cool things with non-Gaussian sampling via the
    # [sample_method] input.
    z_gen = partial(get_z_gen, [(latent_size,)],
        sample_method=args.sample_method, num_components=args.num_components)
    
    tqdm.write(f"----- Final Arguments -----")
    tqdm.write(dict_to_nice_str(vars(args)))
    tqdm.write(f"----- Beginning Training -----")

    k_or_k_minus_one = KorKMinusOne(range(len(data_tr)), shuffle=True)    

    loader_eval = DataLoader(data_val,
        shuffle=False,
        batch_size=18,
        num_workers=8,
        drop_last=True)

    # model to train mode
    model.train()

    cur_step = 0    


    # start depth iteration
    for current_depth in tqdm(range(start_depth, depth), 
                                desc="Depths",
                                dynamic_ncols=True):

        loader = IMLEDataLoader(data_tr, k_or_k_minus_one, model, z_gen, loss_fn, args.ns, args.sp, args.code_bs,
                    depth,
                    current_depth,
                    subsample_size=args.subsample_size,
                    num_iteration=args.num_iteration,
                    pin_memory=True,
                    shuffle=True,
                    batch_size=args.bs,
                    num_workers=4,
                    drop_last=True)

        # Set optimizer and scheduler here to reset at every depth iterations
        optimizer = AdamW(model.parameters(), lr=args.lr[current_depth - start_depth], weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, 
                                        outer_loops[current_depth - start_depth] * args.num_iteration,
                                        eta_min=1e-8,
                                        last_epoch=-1)
        ticker = 1

        for loop in tqdm(range(outer_loops[current_depth - start_depth]),
            desc="OuterLoops",
            dynamic_ncols=True):

            total_batches = len(loader)
            fade_point = int((1 /2 * outer_loops[current_depth - start_depth] * total_batches))
            alpha = 1
            
            noise_save = None
            for batch_idx, (noise, y) in tqdm(enumerate(loader),
                desc="Batches",
                leave=False,
                dynamic_ncols=True,
                total=len(loader)):
                
                y = y.squeeze(0)
                noise = noise.squeeze(0)

                alpha = ticker / fade_point if ticker <= fade_point else 1
                
                batch_loss = 0
                
                # generate fake samples:
                generated_samples = model(noise, current_depth, 1)
                del noise

                # Change this implementation for making it compatible for relativisticGAN                
                loss = loss_fn(generated_samples, y, 'mean')

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
                
                
                del y, loss, batch_loss

            ####################################################################
            # Log data after each loop
            ####################################################################
            images_val = validate(model, z_gen, loss_fn, 1, depth, current_depth, latent_size, args)
            images_file = f"{save_dir}/val_images/res{np.power(2,current_depth+2)}/step{cur_step}.png"
            save_image_grid(images_val, images_file)
            wandb.log({
                "Loop_generated images": wandb.Image(images_file),
            }, step=cur_step)

            tqdm.write(f"Loop {loop:3}/{outer_loops[current_depth - start_depth]} | Loop_lr {get_lr(scheduler)[0]:.5e}")
            del images_val

