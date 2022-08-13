import copy
import datetime
import os
import random
import time
import timeit
import warnings
from collections import OrderedDict
from torch.optim import AdamW

from Losses import *
import numpy as np
import torch
import torch.nn as nn
from data import get_data_loader
from torch.nn.functional import interpolate
from torch.nn.modules.sparse import Embedding
from IMLEDataLoader import IMLEDataLoader


class GMapping(nn.Module):

    def __init__(self, latent_size=512, dlatent_size=512, dlatent_broadcast=None,
                 mapping_layers=8, mapping_fmaps=512, mapping_lrmul=0.01, mapping_nonlinearity='lrelu',
                 use_wscale=True, normalize_latents=True, **kwargs):
        """
        Mapping network used in the StyleGAN paper.
        :param latent_size: Latent vector(Z) dimensionality.
        # :param label_size: Label dimensionality, 0 if no labels.
        :param dlatent_size: Disentangled latent (W) dimensionality.
        :param dlatent_broadcast: Output disentangled latent (W) as [minibatch, dlatent_size]
                                  or [minibatch, dlatent_broadcast, dlatent_size].
        :param mapping_layers: Number of mapping layers.
        :param mapping_fmaps: Number of activations in the mapping layers.
        :param mapping_lrmul: Learning rate multiplier for the mapping layers.
        :param mapping_nonlinearity: Activation function: 'relu', 'lrelu'.
        :param use_wscale: Enable equalized learning rate?
        :param normalize_latents: Normalize latent vectors (Z) before feeding them to the mapping layers?
        :param kwargs: Ignore unrecognized keyword args.
        """

        super().__init__()

        self.latent_size = latent_size
        self.mapping_fmaps = mapping_fmaps
        self.dlatent_size = dlatent_size
        self.dlatent_broadcast = dlatent_broadcast

        # Activation function.
        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[mapping_nonlinearity]

        # Embed labels and concatenate them with latents.
        # TODO

        layers = []
        # Normalize latents.
        if normalize_latents:
            layers.append(('pixel_norm', PixelNormLayer()))

        # Mapping layers. (apply_bias?)
        layers.append(('dense0', EqualizedLinear(self.latent_size, self.mapping_fmaps,
                                                 gain=gain, lrmul=mapping_lrmul, use_wscale=use_wscale)))
        layers.append(('dense0_act', act))
        for layer_idx in range(1, mapping_layers):
            fmaps_in = self.mapping_fmaps
            fmaps_out = self.dlatent_size if layer_idx == mapping_layers - 1 else self.mapping_fmaps
            layers.append(
                ('dense{:d}'.format(layer_idx),
                 EqualizedLinear(fmaps_in, fmaps_out, gain=gain, lrmul=mapping_lrmul, use_wscale=use_wscale)))
            layers.append(('dense{:d}_act'.format(layer_idx), act))

        # Output.
        self.map = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        # First input: Latent vectors (Z) [mini_batch, latent_size].
        x = self.map(x)

        # Broadcast -> batch_size * dlatent_broadcast * dlatent_size
        if self.dlatent_broadcast is not None:
            x = x.unsqueeze(1).expand(-1, self.dlatent_broadcast, -1)
        return x


class GSynthesis(nn.Module):

    def __init__(self, dlatent_size=512, num_channels=3, resolution=1024,
                 fmap_base=8192, fmap_decay=1.0, fmap_max=512,
                 use_styles=True, const_input_layer=True, use_noise=True, nonlinearity='lrelu',
                 use_wscale=True, use_pixel_norm=False, use_instance_norm=True, blur_filter=None,
                 structure='linear', **kwargs):
        """
        Synthesis network used in the StyleGAN paper.
        :param dlatent_size: Disentangled latent (W) dimensionality.
        :param num_channels: Number of output color channels.
        :param resolution: Output resolution.
        :param fmap_base: Overall multiplier for the number of feature maps.
        :param fmap_decay: log2 feature map reduction when doubling the resolution.
        :param fmap_max: Maximum number of feature maps in any layer.
        :param use_styles: Enable style inputs?
        :param const_input_layer: First layer is a learned constant?
        :param use_noise: Enable noise inputs?
        # :param randomize_noise: True = randomize noise inputs every time (non-deterministic),
                                  False = read noise inputs from variables.
        :param nonlinearity: Activation function: 'relu', 'lrelu'
        :param use_wscale: Enable equalized learning rate?
        :param use_pixel_norm: Enable pixel_wise feature vector normalization?
        :param use_instance_norm: Enable instance normalization?
        :param blur_filter: Low-pass filter to apply when resampling activations. None = no filtering.
        :param structure: 'fixed' = no progressive growing, 'linear' = human-readable
        :param kwargs: Ignore unrecognized keyword args.
        """

        super().__init__()

        # if blur_filter is None:
        #     blur_filter = [1, 2, 1]

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.structure = structure

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.depth = resolution_log2 - 1

        self.num_layers = resolution_log2 * 2 - 2
        self.num_styles = self.num_layers if use_styles else 1

        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]

        # Early layers.
        self.init_block = InputBlock(nf(1), dlatent_size, const_input_layer, gain, use_wscale,
                                     use_noise, use_pixel_norm, use_instance_norm, use_styles, act)
        # create the ToRGB layers for various outputs
        rgb_converters = [EqualizedConv2d(nf(1), num_channels, 1, gain=1, use_wscale=use_wscale)]

        # Building blocks for remaining layers.
        blocks = []
        for res in range(3, resolution_log2 + 1):
            last_channels = nf(res - 2)
            channels = nf(res - 1)
            # name = '{s}x{s}'.format(s=2 ** res)
            blocks.append(GSynthesisBlock(last_channels, channels, blur_filter, dlatent_size, gain, use_wscale,
                                          use_noise, use_pixel_norm, use_instance_norm, use_styles, act))
            rgb_converters.append(EqualizedConv2d(channels, num_channels, 1, gain=1, use_wscale=use_wscale))

        self.blocks = nn.ModuleList(blocks)
        self.to_rgb = nn.ModuleList(rgb_converters)

        # register the temporary upsampler
        self.temporaryUpsampler = lambda x: interpolate(x, scale_factor=2)

    def forward(self, dlatents_in, depth=0, alpha=0., labels_in=None):
        """
            forward pass of the Generator
            :param dlatents_in: Input: Disentangled latents (W) [mini_batch, num_layers, dlatent_size].
            :param labels_in:
            :param depth: current depth from where output is required
            :param alpha: value of alpha for fade-in effect
            :return: y => output
        """

        assert depth < self.depth, "Requested output depth cannot be produced"

        if self.structure == 'fixed':
            x = self.init_block(dlatents_in[:, 0:2])
            for i, block in enumerate(self.blocks):
                x = block(x, dlatents_in[:, 2 * (i + 1):2 * (i + 2)])
            images_out = self.to_rgb[-1](x)
        elif self.structure == 'linear':
            x = self.init_block(dlatents_in[:, 0:2])

            if depth > 0:
                for i, block in enumerate(self.blocks[:depth - 1]):
                    x = block(x, dlatents_in[:, 2 * (i + 1):2 * (i + 2)])

                residual = self.to_rgb[depth - 1](self.temporaryUpsampler(x))
                straight = self.to_rgb[depth](self.blocks[depth - 1](x, dlatents_in[:, 2 * depth:2 * (depth + 1)]))

                images_out = (alpha * straight) + ((1 - alpha) * residual)
            else:
                images_out = self.to_rgb[0](x)
        else:
            raise KeyError("Unknown structure: ", self.structure)

        return images_out


class Generator(nn.Module):

    def __init__(self, resolution, latent_size=512, dlatent_size=512, truncation_psi=0.7,
                 truncation_cutoff=8, dlatent_avg_beta=0.995, style_mixing_prob=0.9, **kwargs):
        """
        # Style-based generator used in the StyleGAN paper.
        # Composed of two sub-networks (G_mapping and G_synthesis).
        :param resolution:
        :param latent_size:
        :param dlatent_size:
        :param truncation_psi: Style strength multiplier for the truncation trick. None = disable.
        :param truncation_cutoff: Number of layers for which to apply the truncation trick. None = disable.
        :param dlatent_avg_beta: Decay for tracking the moving average of W during training. None = disable.
        :param style_mixing_prob: Probability of mixing styles during training. None = disable.
        :param kwargs: Arguments for sub-networks (G_mapping and G_synthesis).
        """
        
        super(Generator, self).__init__()

        self.style_mixing_prob = style_mixing_prob

        # Setup components.
        self.num_layers = (int(np.log2(resolution)) - 1) * 2
        self.g_mapping = GMapping(latent_size, dlatent_size, dlatent_broadcast=self.num_layers, **kwargs)
        self.g_synthesis = GSynthesis(resolution=resolution, **kwargs)

        if truncation_psi > 0:
            self.truncation = Truncation(avg_latent=torch.zeros(dlatent_size),
                                         max_layer=truncation_cutoff,
                                         threshold=truncation_psi,
                                         beta=dlatent_avg_beta)
        else:
            self.truncation = None

    def forward(self, latents_in, depth, alpha):
        """
        :param latents_in: First input: Latent vectors (Z) [mini_batch, latent_size].
        :param depth: current depth from where output is required
        :param alpha: value of alpha for fade-in effect
        :param labels_in: Second input: Conditioning labels [mini_batch, label_size].
        :return:
        """
        dlatents_in = self.g_mapping(latents_in)

        if self.training:
            # Update moving average of W(dlatent).
            # TODO
            if self.truncation is not None:
                self.truncation.update(dlatents_in[0, 0].detach())

            # Perform style mixing regularization.
            if self.style_mixing_prob is not None and self.style_mixing_prob > 0:
                latents2 = torch.randn(latents_in.shape).to(latents_in.device)
                dlatents2 = self.g_mapping(latents2)
                layer_idx = torch.from_numpy(np.arange(self.num_layers)[np.newaxis, :, np.newaxis]).to(
                    latents_in.device)
                cur_layers = 2 * (depth + 1)
                mixing_cutoff = random.randint(1,
                                               cur_layers) if random.random() < self.style_mixing_prob else cur_layers
                dlatents_in = torch.where(layer_idx < mixing_cutoff, dlatents_in, dlatents2)

            # Apply truncation trick.
            if self.truncation is not None:
                dlatents_in = self.truncation(dlatents_in)

        fake_images = self.g_synthesis(dlatents_in, depth, alpha)

        return fake_images 

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
    P.add_argument("--code_bs", type=int, default=2,
        help="GPU ids")

    # Training hyperparameter arguments. These are logged!
    P.add_argument("--data", required=True, choices=datasets,
        help="data to train on")
    P.add_argument("--res", nargs="+", type=int, default=[64, 64, 64, 64, 128],
        help="resolutions to see data at")
    P.add_argument("--alpha", type=float, default=.1,
        help="Amount of weight on MSE loss")
    P.add_argument("--seed", type=int, default=0,
        help="random seed")
    P.add_argument("--proj_dim", default=1000, type=int,
        help="projection dimensionality")
    P.add_argument("--epochs", default=20, type=int,
        help="number of epochs (months) to train for")
    P.add_argument("--outer_loops", default=20, type=int,
        help="number of outer_loops to train for")
    P.add_argument("--bs", type=int, default=512,
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
    P.add_argument("--subsample_size", default=None, type=int,
        help="number of subsample data ")
    P.add_argument("--num_iteration", default=1, type=int,
        help="number of subsample data ")
        
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

    assert args.code_bs >= len(args.gpus)

    if not args.ipc % args.mini_bs == 0 or args.ipc // args.mini_bs == 0:
        raise ValueError(f"--ipc should be a multiple of --mini_bs")

    return args

if __name__ == "__main__":
    args = get_args()

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
        save_dir = generator_folder(args, ignore_conflict=False)
        cur_seed = set_seed(args.seed)

        # Setup the experiment. Importantly, we copy the experiment's ID to
        # [args] so that we can resume it later.
        args.run_id = wandb.util.generate_id()
        wandb.init(anonymous="allow", id=args.run_id, config=args,
            mode=args.wandb, project="isicle-generator")

        # changed from model to generator
        generator = Generator(num_channels=num_channels,
                             resolution=resolution,
                             structure=self.structure,
                             conditional=self.conditional,
                             n_classes=self.n_classes,
                             **seed_kwargs(cur_seed)).to(self.device)

        optimizer = AdamW(generator.parameters(), lr=args.lr, weight_decay=1e-4)
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

        generator = resume_data["generator"].to(device)
        optimizer = resume_data["optimizer"]
        last_epoch = resume_data["last_epoch"]
        scheduler = resume_data["scheduler"]
    
    loss_fn = nn.DataParallel(UnconditionalIMLELoss(alpha=args.alpha), device_ids=args.gpus).to(device)

    data_tr, data_eval = get_data_splits(args.data,
        eval_str="val" if same_distribution_splits else "cv",
        res=args.res,
        data_path=args.data_path)

    data_tr = UCGeneratorDataset(data_tr, get_gen_augs(args))
    data_eval = UCGeneratorDataset(data_eval, get_gen_augs(args))

    eval_len = len(data_eval) // (args.spi + 2)
    eval_len = round_so_evenly_divides(eval_len, len(args.gpus))
    eval_idxs = set(range(0, len(data_eval), eval_len))
    data_eval = Subset(data_eval, indices=list(eval_idxs))

    if not same_distribution_splits:
        idxs = [idx for idx in range(len(data_tr)) if not idx in eval_idxs]
        data_tr = Subset(data_tr, indices=idxs)

    # Get a function that returns random codes given a level. We will use
    # this to do cool things with non-Gaussian sampling via the
    # [sample_method] input.
    z_gen = partial(get_z_gen, get_z_dims(args), sample_method=args.sample_method)

    loader_eval = DataLoader(data_eval,
        shuffle=False,
        batch_size=max(len(args.gpus), args.bs),
        num_workers=8,
        drop_last=True)
    
    # Naming 
    loader_tr = IMLEDataLoader(data_tr, z_gen, loss_fn, args.ns, args.sp, args.code_bs,
                    subsample_size=args.subsample_size,
                    num_iteration=args.num_iteration,
                    pin_memory=True,
                    shuffle=True,
                    batch_size=max(len(args.gpus), args.bs),
                    num_workers=8,
                    drop_last=True)

    for e in tqdm(range(args.outer_loops),
        desc="OuterLoops",
        dynamic_ncols=True):
                
        # total_batches = len(iter(data))
        total_batches = len(loader_tr)

        for batch_idx, (codes, ys) in tqdm(enumerate(loader_tr),
            desc="Batches",
            leave=False,
            dynamic_ncols=True,
            total=len(loader_tr)):

            batch_loss = 0

            # generate fake samples:
            fake_samples = generator(noise, alpha)

            loss = compute_loss_over_list(fake_samples, [y.to(device) for y in ys], loss_fn)

            loss.backward()
            
            # Gradient Clipping
            nn.utils.clip_grad_norm_(generator.parameters(), max_norm=10.)
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            scheduler.step()

            batch_loss += loss.detach()
            cur_step += 1
            wandb.log({
                "batch loss": loss.detach(),
                "learning rate": get_lr(scheduler)[0]
            }, step=cur_step)
            
            del ys, loss, batch_loss 