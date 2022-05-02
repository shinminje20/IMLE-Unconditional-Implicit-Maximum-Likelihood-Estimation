import argparse
from tqdm import tqdm
import wandb

import torch
from torch.optim import Adam, SGD
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from Data import *
from Evaluation import classification_eval
from utils.UtilsContrastive import *
from utils.Utils import *
from utils.UtilsNN import *


def one_epoch_contrastive(model, optimizer, loader, temp=.5):
    """Returns a (model, optimizer, loss) tuple after training [model] on
    [loader] for one epoch.

    The returned loss is averaged over batches.

    model           -- a model of the form projection_head(feature_extractor())
    optimizer       -- the optimizer for [model]
    loader          -- a DataLoader over the data to train on
    temp            -- contrastive loss temperature
    """
    model.train()
    loss_fn = NTXEntLoss(temp)
    loss_total = 0
    scaler = GradScaler()

    for x1,x2 in tqdm(loader, desc="Batches", total=len(loader), leave=False, dynamic_ncols=True):

        with autocast():

            model.zero_grad(set_to_none=True)
            loss = loss_fn(model(x1.float().to(device, non_blocking=True)),
                           model(x2.float().to(device, non_blocking=True))).unsqueeze(0)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        loss_total += loss.item()
        scaler.update()

    return model, optimizer, loss_total / len(loader)

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="SimCLR training")
    P.add_argument("--wandb", default=1, choices=[0, 1], type=int,
        help="Whether to use W&B logging or not")
    P.add_argument("--data_path", default=data_dir, type=str,
        help="path to data if not in normal place")
    P.add_argument("--data", choices=["cifar10", "miniImagenet", "gen_miniImagenet"],
        default="cifar10",
        help="dataset to load images from")
    P.add_argument("--eval", default="val", choices=["val", "cv", "test"],
        help="The data to evaluate linear finetunings on")
    P.add_argument("--resume", default=None, type=str,
        help="file to resume from")
    P.add_argument("--suffix", default="", type=str,
        help="suffix")
    P.add_argument("--res", nargs="+",
                   default=None, choices=[32, 64, 128, 256], type=int,
        help="image resolution to load")

    # Non-hyperparameter arguments
    P.add_argument("--num_workers", default=24, type=int,
        help="Number of workers for data loading")
    P.add_argument("--eval_iter", default=10, type=int,
        help="number of epochs between linear evaluations")
    P.add_argument("--save_iter", default=100, type=int,
        help="save a model every --save_iter epochs")
    P.add_argument("--unreal_augs", default=1, type=int, choices=[0, 1],
        help="whether to use augs that can take an image off the real manifold")

    # Hyperparameter arguments
    P.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"],
        help="Resnet backbone to use")

    P.add_argument("--color_s", default=1, type=float,
        help="color distortion strength")
    P.add_argument("--gaussian_blur", choices=[0, 1], type=int, default=0,
        help="include Gaussian blur in data augmentation")
    P.add_argument("--crop_size", type=int, default=32,
        help="resolution at which to feed images to the network")

    P.add_argument("--bs", default=1000, type=int,
        help="batch size")
    P.add_argument("--epochs", default=1000, type=int,
        help="number of epochs")
    P.add_argument("--lars", default=1, choices=[0, 1],
        help="whether or not to use LARS")
    P.add_argument("--lr", default=1e-3, type=float,
        help="base learning rate")
    P.add_argument("--mm", nargs="+", default=(.9, .999), type=float,
        help="momentum (one arg for SGD, two—beta1 and beta2 for Adam)")
    P.add_argument("--n_ramp", default=10, type=int,
        help="Number of linear ramp epochs at start of training")
    P.add_argument("--opt", choices=["adam", "sgd"], default="adam",
        help="optimizer")
    P.add_argument("--proj_dim", default=128, type=int,
        help="dimension of projection space")
    P.add_argument("--temp", default=.5, type=float,
        help="contrastive loss temperature")
    P.add_argument("--trust", default=.001, type=float,
        help="LARS trust coefficient")
    P.add_argument("--seed", default=0, type=int,
        help="random seed")
    args = P.parse_args()

    ############################################################################
    # Check arguments
    ############################################################################
    if not args.save_iter % args.eval_iter == 0:
        tqdm.write("WARNING: training will save a checkpoint without direct evaluation. Ensure --save_iter % --eval_iter is zero to avoid this.")
    if args.opt == "sgd" and len(args.mm) == 2:
        raise ValueError("--mm must be a single momentum parameter if --opt is 'sgd'.")
    if args.data in no_val_split_datasets and args.eval == "val":
        raise ValueError("The requested dataset has no validation split. Run with --eval test or cv instead.")

    # This needs to be a tuple or a float depending on its length
    args.mm = args.mm[0] if len(args.mm) == 1 else args.mm

    ############################################################################
    # Load prior state if it exists, otherwise instantiate a new training run.
    ############################################################################
    if args.resume is not None:
        run_id, resume_data = wandb_load(args.resume)
        cur_seed = set_seed(resume_data["seed"])
        data_path = args.data_path

        wandb.init(id=run_id, resume="must", project="isicle")
        wandb.save("*.pt")
        model = resume_data["model"].to(device)
        optimizer = resume_data["optimizer"]
        last_epoch = resume_data["last_epoch"]
        args = resume_data["args"]
        args.data_path = data_path

        save_dir = simclr_folder(args)
    else:
        cur_seed = set_seed(args.seed)
        args.run_id = wandb.util.generate_id()
        save_dir = simclr_folder(args)

        wandb.init(anonymous="allow", id=args.run_id, project="isicle-simclr",
            mode="online" if args.wandb else "disabled", config=args,
            name=f"{args.data}_{args.backbone}_{args.run_id}{suffix_str(args)}")

        model = HeadedResNet(args.backbone, args.proj_dim,
            head_type="projection",
            small_image=(args.data in small_image_datasets)).to(device)
        if args.opt == "adam":
            optimizer = Adam(get_param_groups(model, args.lars), lr=args.lr,
                betas=args.mm, weight_decay=1e-6)
        elif args.opt == "sgd":
            optimizer = SGD(get_param_groups(model, args.lars), lr=args.lr,
                momentum=args.mm, weight_decay=1e-6, nesterov=True)
        else:
            raise ValueError(f"--opt was {args.opt} but must be one of 'adam' or 'sgd'")
        optimizer = LARS(optimizer, args.trust) if args.lars else optimizer
        last_epoch = -1

    tqdm.write(dict_to_nice_str(vars(args)))

    ############################################################################
    # Instantiate the scheduler and get the data
    ############################################################################
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
        first_cycle_steps=args.epochs, max_lr=args.lr,  min_lr=1e-6,
        warmup_steps=args.n_ramp, last_epoch=last_epoch)
    data_tr, data_eval = get_data_splits(args.data, args.eval, res=args.res,
        data_path=args.data_path)

    if args.unreal_augs:
        augs_tr, augs_fn, augs_te = get_contrastive_augs(crop_size=args.crop_size,
            gaussian_blur=args.gaussian_blur, color_s=args.color_s)
    else:
        augs_tr, augs_fn, augs_te = get_real_augs(crop_size=args.crop_size)
   
    data_ssl = ManyTransformsDataset(data_tr, augs_tr, augs_tr)
    loader = DataLoader(data_ssl, shuffle=True, batch_size=args.bs,
        drop_last=True, num_workers=args.num_workers, pin_memory=True,
        **seed_kwargs(cur_seed))

    tqdm.write(f"Dataset length {len(data_tr)}")

    ############################################################################
    # Begin training!
    ############################################################################
    for e in tqdm(range(max(last_epoch + 1, 1), args.epochs + 1), desc="Epochs", dynamic_ncols=True):
        model, optimizer, loss_tr = one_epoch_contrastive(model, optimizer,
            loader, args.temp)

        ########################################################################
        # LOG RESULTS OF THE EPOCH. Perform a classification cross validation if
        # desired, and otherwise print/log results or merely that the epoch
        # happened
        ########################################################################
        if e % args.eval_iter == 0 and not e == 0 and args.eval_iter > 0:
            val_acc_avg, val_acc_std = classification_eval(model.backbone,
                data_tr, data_eval, augs_fn, augs_te, data_name=args.data,
                data_split=args.eval, trials=1, num_workers=args.num_workers)

            wandb.log({"epoch": e, "loss_tr": loss_tr / len(loader),
                "acc_val": val_acc_avg, "lr": scheduler.get_lr()[0]})
            tqdm.write(f"End of epoch {e} | lr {scheduler.get_lr()[0]:.5f} | loss_tr {loss_tr / len(loader):.5f} | val acc {val_acc_avg:.5f} ± {val_acc_std:.5f}")
        else:
            wandb.log({"epoch": e, "loss_tr": loss_tr / len(loader),
                "lr": scheduler.get_lr()[0]})
            tqdm.write(f"End of epoch {e} | lr {scheduler.get_lr()[0]:.5f} | loss_tr {loss_tr / len(loader):.5f}")

        if e % args.save_iter == 0 and not e == 0:
            wandb_save({"model": model, "optimizer": optimizer, "args": args,
                "last_epoch": e}, f"{simclr_folder(args)}/{e}.pt")
            tqdm.write("Saved training state")

        scheduler.step()
