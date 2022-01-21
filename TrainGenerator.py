import argparse
from tqdm import tqdm

import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from CAMNet import CAMNet

from Corruptions import get_non_learnable_batch_corruption
from Data import *
from utils.Utils import *
from utils.UtilsLPIPS import LPIPSFeats

class BatchMSELoss(nn.Module):
    """MSELoss but with the reduction leaving the batch dimension intact."""
    def __init__(self):
        super(BatchMSELoss, self).__init__()

    def forward(self, fx, y):
        unreduced_loss = F.mse_loss(fx, y, reduction="none")
        return torch.sum(unreduced_loss.view(len(fx), -1), axis=1)

class LPIPSLoss(nn.Module):
    """Returns loss between LPIPS features of generated and target images."""
    def __init__(self, reduction="mean"):
        super(LPIPSLoss, self).__init__()
        self.l_feats = LPIPSFeats()
        self.reduction = reduction
        if reduction == "mean":
            self.loss = nn.MSELoss(reduction="mean")
        elif reduction == "batch":
            self.loss = BatchMSELoss()
        else:
            raise ValueError(f"Unknown reduction '{reduction}'")

    def forward(self, fx, y):
        """Returns the loss between generated images [fx] and real images [y].
        [fx] and [y] can either be lists of images and loss is the sum of the
        loss computed on their pairwise elements, or simply batches of images.
        """
        if isinstance(fx, list) and isinstance(y, list):
            if self.reduction == "mean":
                losses = [self.loss(self.l_feats(fx_), self.l_feats(y_)) for fx_,y_ in zip(fx, y)]
                return torch.mean(torch.stack(losses))
        else:
            return self.loss(self.l_feats(fx), self.l_feats(y))

def get_new_codes(z_dims, data_subset, corruptor, backbone, loss_fn="lpips", code_bs=6, num_samples=120):
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
    if loss_fn == "lpips":
        loss_fn = LPIPSLoss(reduction="batch").to(device)
    elif loss_fn == "mse":
        loss_fn = nn.MSELoss(reduction="none").to(device)
    else:
        raise ValueError(f"Unknown loss type {loss_fn}")

    bs = len(data_subset)
    level_codes = [torch.randn((bs,)+z, device=device) for z in z_dims]
    loader = DataLoader(data_subset, batch_size=code_bs, shuffle=False,
                        num_workers=num_workers)
    for level_idx in tqdm(range(len(z_dims)), desc="levels", leave=False):
        least_losses = torch.ones(bs, device=device) * float("inf")

        for _ in tqdm(range(num_samples), desc="sampling", leave=False):
            for idx,(x,ys) in enumerate(loader):
                start_idx = code_bs * idx
                end_idx = code_bs * (idx + 1)
                least_losses_batch = least_losses[start_idx:end_idx]

                old_codes = [l[start_idx:end_idx] for l in level_codes[level_idx:]]
                new_codes = torch.randn((code_bs,) + z_dims[level_idx], device=device)
                test_codes = old_codes + [new_codes]

                with torch.no_grad():
                    cx = corruptor(x.to(device))
                    fx = backbone(cx, test_codes, loi=level_idx)
                    losses = loss_fn(fx, ys[level_idx].to(device))

                change_idxs = losses < least_losses_batch
                level_codes[level_idx][start_idx:end_idx][change_idxs] = new_codes[change_idxs]

    return [l.cpu() for l in level_codes]

def one_epoch_imle(corruptor, generator, optimizer, dataset, loss_fn, bs=1,
    mini_bs=1, code_bs=1, iters_per_code_per_ex=1000, num_samples=12):
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
    loss_fn = LPIPSLoss(reduction="mean").to(device)
    total_loss = 0

    rand_idxs = random.sample(range(len(dataset)), len(dataset))
    for batch_idx in tqdm(range(0, len(dataset), bs), desc="Batches", leave=False):
        images_dataset = Subset(dataset, rand_idxs[batch_idx:batch_idx + bs])
        codes_dataset = ZippedDataset(*get_new_codes(generator.get_z_dims(),
                                                     images_dataset, corruptor,
                                                     generator, code_bs=code_bs,
                                                     num_samples=num_samples))
        batch_dataset = ZippedDataset(codes_dataset, images_dataset)
        loader = DataLoader(batch_dataset, batch_size=mini_bs,
                            num_workers=num_workers, shuffle=True)

        inner_loop_iters = int(iters_per_code_per_ex * len(batch_dataset) / mini_bs)

        for _ in tqdm(range(inner_loop_iters), desc="inner loop", leave=False):

            for codes,(x,ys) in tqdm(loader, desc="Minibatches", leave=False):

                generator.zero_grad()
                cx = corruptor(x.to(device))
                fx = generator(cx, [c.to(device) for c in codes])
                loss = loss_fn(fx, [y.to(device) for y in ys])
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            tqdm.write(f"Cur loss {loss.item()}")

    return corruptor, generator, optimizer, total_loss / len(loader)

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="CAMNet training")
    P.add_argument("--data", default="cifar10", choices=["cifar10", "camnet3"],
        help="data to train on")
    P.add_argument("--eval", default="cv", choices=["cv", "eval", "test"],
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
    P.add_argument("--grayscale", default=1, choices=[0, 1],
        help="grayscale corruption")
    P.add_argument("--pixel_mask_frac", default=.5, type=float,
        help="fraction of pixels to mask at 16x16 resolution")
    P.add_argument("--rand_illumination", default=.2, type=float,
        help="amount by which the illumination of an image can change")

    ############################################################################
    # Training hyperparameters
    ############################################################################
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
    P.add_argument("--ipcpe", type=int, default=10,
        help="iters_per_code_per_ex")
    P.add_argument("--lr", type=float, default=1e-3,
        help="learning rate")
    P.add_argument("--wd", type=float, default=1e-6,
        help="weight decay")
    P.add_argument("--mm", nargs="+", default=(.9, .999), type=float,
        help="momentum (one arg for SGD, two—beta1 and beta2 for Adam)")

    P.add_argument("--suffix", default="",
        help="optional training suffix")
    P.add_argument("--options", default=[], nargs="+",
        help="options")
    args, unparsed_args = P.parse_known_args()

    if not int(args.bs / args.mini_bs) == float(args.bs / args.mini_bs):
        raise ValueError(f"--mini_bs {args.mini_bs} must evenly divide --bs {args.bs}")
    if not int(args.bs / args.mini_bs) == float(args.bs / args.mini_bs):
        raise ValueError(f"--mini_bs {args.mini_bs} must evenly divide --bs {args.bs}")

    ############################################################################
    # Create the dataset
    ############################################################################
    data_tr, data_eval = get_data_splits(args.data, args.eval, args.res)
    base_transform = get_gen_augs()
    data_tr = GeneratorDataset(data_tr, base_transform)

    if not evenly_divides(args.bs, len(data_tr)):
        raise ValueError(f"--bs {args.bs} must evenly divide the length of the dataset {len(data_tr)}")

    ############################################################################
    # Create the corruption
    ############################################################################
    corruptor = get_non_learnable_batch_corruption(
        grayscale=args.grayscale,
        rand_illumination=args.rand_illumination,
        pixel_mask_frac=args.pixel_mask_frac)

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

    last_epoch = -1
    schedulers = CosineAnnealingLinearRampLR(optimizer, args.epochs, args.n_ramp,
        last_epoch=last_epoch)
    loss_fn = LPIPSLoss(reduction="mean").to(device)
    ############################################################################
    # Begin training!
    ############################################################################

    for e in tqdm(range(max(last_epoch + 1, 1), args.epochs + 1), desc="Epochs", file=sys.stdout):
        corruptor, generator, optimizer, loss_tr = one_epoch_imle(corruptor,
            generator, optimizer, data_tr, loss_fn, bs=args.bs,
            code_bs=args.code_bs, mini_bs=args.mini_bs,
            num_samples=args.num_samples,
            iters_per_code_per_ex=args.ipcpe)
        #
        # # Perform a classification cross validation if desired, and otherwise
        # # print/log results or merely that the epoch happened.
        # if e % args.eval_iter == 0 and not e == 0 and args.eval_iter > 0:
        #     val_acc_avg, val_acc_std = classification_eval(
        #         model.backbone,
        #         data_tr, data_eval,
        #         augs_fn, augs_te,
        #         data_name=args.data,
        #         data_split=args.eval)
        #     tb_results.add_scalar("Loss/train", loss_tr / len(loader), e)
        #     tb_results.add_scalar("Accuracy/val", val_acc_avg, e)
        #     tb_results.add_scalar("LR", scheduler.get_last_lr()[0], e)
        #     tqdm.write(f"End of epoch {e} | lr {scheduler.get_last_lr()[0]:.5f} | loss {loss_tr / len(loader):.5f} | val acc {val_acc_avg:.5f} ± {val_acc_std:.5f}")
        # else:
        #     tb_results.add_scalar("Loss/train", loss_tr / len(loader), e)
        #     tb_results.add_scalar("LR", scheduler.get_last_lr()[0], e)
        #     tqdm.write(f"End of epoch {e} | lr {scheduler.get_last_lr()[0]:.5f} | loss {loss_tr / len(loader):.5f}")
        #
        # # Saved the model if desired
        # if e % args.save_iter == 0 and not e == 0:
        #     save_simclr(model, optimizer, e, args, tb_results, simclr_folder(args))
        #     tqdm.write("Saved training state")

        scheduler.step()
