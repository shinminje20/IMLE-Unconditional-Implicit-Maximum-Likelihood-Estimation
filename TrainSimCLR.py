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

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="SimCLR training")
    P.add_argument("--wandb", default="online",
        choices=["online", "offline", "disabled"],
        help="How to use W&B logging")
    P.add_argument("--eval", default="val", choices=["val", "cv", "test"],
        help="The data to evaluate linear finetunings on")
    P.add_argument("--resume", default=None, type=str,
        help="file to resume from")
    P.add_argument("--suffix", default="", type=str,
        help="suffix")
    P.add_argument("--data", choices=["cifar10", "miniImagenet", "gen_miniImagenet"],
        default="cifar10",
        help="dataset to load images from")
    P.add_argument("--data_path", default=data_dir, type=str,
        help="path to data if not in normal place")
    P.add_argument("--res", default=128, choices=[32, 64, 128, 256], type=int,
        help="image resolution to load")

    # Non-hyperparameter arguments
    P.add_argument("--eval_iter", default=10, type=int,
        help="number of epochs between linear evaluations")
    P.add_argument("--save_iter", default=100, type=int,
        help="save a model every --save_iter epochs")
    P.add_argument("--unreal_augs", default=1, type=int, choices=[0, 1],
        help="whether to use augs that can take an image off the real manifold")

    # Arguments controlling the kind/amount of augmentation. They only take
    # effect when --unreal_augs is set to 1
    P.add_argument("--color_s", default=1, type=float,
        help="color distortion strength")
    P.add_argument("--gaussian_blur", choices=[0, 1], type=int, default=0,
        help="include Gaussian blur in data augmentation")

    # Hyperparameter arguments
    P.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"],
        help="Resnet backbone to use")
    P.add_argument("--bs", default=1000, type=int,
        help="batch size")
    P.add_argument("--epochs", default=1000, type=int,
        help="number of epochs")
    P.add_argument("--lars", default=1, choices=[0, 1],
        help="whether or not to use LARS")
    P.add_argument("--lr", default=1e-3, type=float,
        help="base learning rate")
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

    ############################################################################
    # Load prior state if it exists, otherwise instantiate a new training run.
    ############################################################################
    if args.resume is not None:
        run_id, resume_data = wandb_load(args.resume)
        cur_seed = set_seed(resume_data["seed"])
        data_path = args.data_path

        wandb.init(id=run_id, resume="must", project="isicle-simclr", config=args)
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
            mode=args.wandb, config=args)

        model = HeadedResNet(args.backbone, args.proj_dim,
            head_type="projection",
            small_image=(args.res < 64)).to(device)
        if args.lars:
            optimizer = Adam(lars_params(model), lr=args.lr, weight_decay=1e-6)
            optimizer = LARS(optimizer, args.trust)
        else:
            optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
        last_epoch = -1

    tqdm.write(f"{'-' * 40}\n{dict_to_nice_str(vars(args))}\n{'-' * 40}")

    ############################################################################
    # Instantiate data-related and remaining components.
    ############################################################################
    data_tr, data_eval = get_data_splits(args.data,
        eval_str=args.eval,
        res=args.res,
        data_path=args.data_path)

    if args.unreal_augs:
        augs_tr, augs_fn, augs_te = get_contrastive_augs(
            res=args.res,
            gaussian_blur=args.gaussian_blur,
            color_s=args.color_s)
    else:
        augs_tr, augs_fn, augs_te = get_real_augs(res=args.res)
   
    data_ssl = ManyTransformsDataset(data_tr, augs_tr, augs_tr)
    loader = DataLoader(data_ssl, shuffle=True, batch_size=args.bs,
        drop_last=True, num_workers=24, pin_memory=True,
        **seed_kwargs(cur_seed))

    loss_fn = NTXEntLoss(args.temp)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
        first_cycle_steps=args.epochs * len(loader),
        max_lr=args.lr,  min_lr=1e-6,
        warmup_steps=args.n_ramp * len(loader), last_epoch=last_epoch)

    ############################################################################
    # Begin training!
    ############################################################################
    for e in tqdm(range(max(last_epoch + 1, 1), args.epochs + 1),
        desc="Epochs",
        dynamic_ncols=True):

        model.train()
        epoch_loss = 0
        scaler = GradScaler()

        for batch_idx,(x1,x2) in tqdm(enumerate(loader),
            desc="Batches",
            total=len(loader),
            leave=False,
            dynamic_ncols=True):
            
            with autocast():
                model.zero_grad(set_to_none=True)
                fx1 = model(x1.float().to(device, non_blocking=True))
                fx2 = model(x2.float().to(device, non_blocking=True))
                loss = loss_fn(fx1, fx2).unsqueeze(0)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.detach()

            wandb.log({"training loss": loss.detach(),
                    "lr": scheduler.get_lr()[0]},
                    step=e * len(loader) + batch_idx)

        epoch_loss = epoch_loss.item()
        if e % args.eval_iter == 0 and not e == 0 and args.eval_iter > 0:
            val_acc_avg, val_acc_std = classification_eval(model.backbone,
                data_eval, "cv", augs_fn, augs_te,
                data_name=args.data,
                data_split=args.eval,
                trials=2,
                num_workers=24)

            wandb.log({"validation accuracy": val_acc_avg},
                step=e * len(loader) + batch_idx)
            tqdm.write(f"End of epoch {e} | lr {scheduler.get_lr()[0]:.5f} | average loss over epoch {epoch_loss / len(loader):.5f} | val acc {val_acc_avg:.5f} ± {val_acc_std:.5f}")
        else:
            tqdm.write(f"End of epoch {e} | lr {scheduler.get_lr()[0]:.5f} | average loss over epoch {epoch_loss / len(loader):.5f}")

        if e % args.save_iter == 0 and not e == 0:
            save_checkpoint({"model": model.cpu(),
                "last_epoch": e, "args": args, "scheduler": scheduler,
                "optimizer": optimizer}, f"{save_dir}/{e}.pt")
            tqdm.write("Saved training state")

        