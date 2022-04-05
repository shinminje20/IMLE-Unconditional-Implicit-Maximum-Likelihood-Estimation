"""File for generating lots of images."""
import argparse
from collections import defaultdict
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torchvision.utils import save_image
from TrainGenerator import *
from Data import *
from utils.Utils import *
from CAMNet import *
from tqdm import tqdm
from torchvision import transforms

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="CAMNet training")
    P.add_argument("--data", default="cifar10",
        help="data to train on")
    P.add_argument("--data_path", default=f"{project_dir}/data", type=str,
        help="path to data if not in normal place")
    P.add_argument("--seed", type=int, default=0,
        help="random seed")
    P.add_argument("--resume", type=str, default=None,
        help="WandB run to resume from")
    P.add_argument("--res", type=int, required=True,
        help="resolutions to see data at")
    P.add_argument("--gpus", type=int, default=[0], nargs="+",
        help="GPU ids")
    P.add_argument("--bs", default=128, type=int,
        help="GPU ids")
    P.add_argument("--output_dir", required=True,
        help="GPU ids")
    P.add_argument("--split", default="train")
    P.add_argument("--epochs", type=int, default=100)
    args = P.parse_args()

    _, resume_data = wandb_load(args.resume)
    set_seed(args.seed)
    data_path = args.data_path
    model = resume_data["model"].to(device)
    corruptor = resume_data["corruptor"].to(device)

    data = ImageFolder(f"{args.data_path}/{args.data}_{args.res}x{args.res}/{args.split}", transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=args.bs, shuffle=False, pin_memory=True, num_workers=24, drop_last=False)

    ############################################################################
    # Map each index in the dataset to its index for its label
    ############################################################################
    label2data_idxs = defaultdict(lambda: [])
    data_idxs2ys = torch.zeros(len(data), dtype=torch.long)
    for batch_idx,(_,ys) in enumerate(loader):
        ys = ys.long()
        data_idxs = torch.arange(len(ys), dtype=torch.long) + batch_idx * args.bs
        data_idxs2ys[data_idxs] = ys

        for y,data_idx in zip(ys, data_idxs):
            label2data_idxs[y.item()].append(data_idx.item())

    data_idx2label_idx = [None] * len(data)
    for data_idxs in label2data_idxs.values():
        for label_idx,data_idx in enumerate(data_idxs):
            data_idx2label_idx[data_idx] = label_idx

    data_idx2file_base = [f"{args.output_dir}/{data_idxs2ys[d]}/image{data_idx2label_idx[d]}" for d in range(len(data))]

    for f in data_idx2file_base:
        if not os.path.exists(os.path.dirname(f)): os.makedirs(f)

    z_dims = get_z_dims(model)

    for e in tqdm(range(args.epochs), desc="(1/2) sampled epochs"):

        with torch.no_grad():
            for batch_idx,(xs,ys) in tqdm(enumerate(loader), desc="Batches", leave=False, total=len(loader)):
                data_idxs = torch.arange(len(xs)) + batch_idx * args.bs

                with autocast():
                    cx = corruptor(xs.to(device, non_blocking=True))
                    codes = [generate_z(args.bs, z, sample_method="normal") for z in z_dims]
                    fx = model(cx, codes, loi=-1)

                fx = [image for image in fx]
                files = [f"{data_idx2file_base[d]}_aug{e}.jpg" for d in data_idxs]

                for image,file in zip(fx,files):
                    save_image(image, file)
