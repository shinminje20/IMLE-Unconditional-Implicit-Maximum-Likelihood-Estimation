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

from TrainGeneratorWandB import *

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--data", default="camnet3", choices=datasets,
        help="data to generate images from")
    P.add_argument("--split", default="train",
        help="split of the dataset to use")
    P.add_argument("--data_path", default=f"{project_dir}/data", type=str,
        help="path to data if not in normal place")
    P.add_argument("--seed", type=int, default=0,
        help="random seed")
    P.add_argument("--resume", type=str, default=None,
        help="WandB run to resume from")
    P.add_argument("--gpus", type=int, default=[0, 1], nargs="+",
        help="GPU ids")
    P.add_argument("--bs", default=128, type=int,
        help="GPU ids")
    P.add_argument("--output_dir", required=True,
        help="GPU ids")
    P.add_argument("--epochs", type=int, default=100,
        help="number of epochs of training data to generate")
    P.add_argument("--suffix", default="", required=True,
        help="suffix for the dataset")
    args = P.parse_args()

    resume_data = torch.load(args.resume)
    curr_args = args
    generator_args = resume_data["args"]
    cur_seed = set_seed(resume_data["seed"])
    model = resume_data["model"].to(device)
    corruptor = resume_data["corruptor"].to(device)

    z_gen = partial(get_z_gen,
        get_z_dims(generator_args),
        sample_method=generator_args.sample_method)

    in_res = generator_args.res[0]
    out_res = generator_args.res[-1]
    tqdm.write("Input resolution: {in_res}x{in_res} | Output resolution: {out_res}x{out_res}")
    generated_data_name = f"{data_dir}/{args.data}_{out_res}x{out_res}-{args.suffix}/{args.split}"
    tqdm.write(f"Will write generated dataset to {generated_data_name}")

    data = ImageFolder(
        f"{args.data_path}/{args.data}_{in_res}x{in_res}/{args.split}", transform=transforms.ToTensor())
    loader = DataLoader(data,
        batch_size=args.bs,
        shuffle=False,
        pin_memory=True,
        num_workers=24)

    ############################################################################
    # Get a mapping from indices to [data] to the index of an image in its label
    ############################################################################
    label2idxs = defaultdict(lambda: [])
    for idx,(_,ys) in enumerate(loader):
        idxs = range(idx * args.bs, min(len(data), (idx+1) * args.bs))
        for image_idx,y in zip(idxs, ys.tolist()):
            label2idxs[y].append(image_idx)

    data_idx2label_idx = {}
    for image_idxs in label2idxs.values():
        for label_idx,image_idx in enumerate(image_idxs):
            data_idx2label_idx[image_idx] = label_idx

    ############################################################################
    # Generate images!
    ############################################################################
    for e in tqdm(range(args.epochs * 2),
        desc="1/2 contrastive learning epochs",
        dynamic_ncols=True):

        for idx,(xs,ys) in tqdm(enumerate(loader),
            desc="Batches",
            leave=False,
            dynamic_ncols=True):

            cx = corruptor(xs)
            codes = z_gen(bs, level="all", input="show_components")
            images = model(cx, codes, loi=-1)

            image_idxs = range(idx * args.bs, min(len(data), (idx+1) * args.bs))
            for image,y,data_idx in zip(images, ys, image_idxs):
                
                folder = f"{generated_data_name}/{int(y)}"
                if not os.path.exists(folder):
                    os.makedirs(folder)

                filename = f"{folder}/{data_idx2label_idx[data_idx]}_aug{e}.png"
                save_image(image, filename)


    ############################################################################
    # Map each index in the dataset to its index for its label
    ############################################################################
    # label2data_idxs = defaultdict(lambda: [])
    # data_idxs2ys = torch.zeros(len(data), dtype=torch.long)
    # for batch_idx,(_,ys) in enumerate(loader):
    #     ys = ys.long()
    #     data_idxs = torch.arange(len(ys), dtype=torch.long) + batch_idx * args.bs
    #     data_idxs2ys[data_idxs] = ys

    #     for y,data_idx in zip(ys, data_idxs):
    #         label2data_idxs[y.item()].append(data_idx.item())

    # data_idx2label_idx = [None] * len(data)
    # for data_idxs in label2data_idxs.values():
    #     for label_idx,data_idx in enumerate(data_idxs):
    #         data_idx2label_idx[data_idx] = label_idx

    # data_idx2file_base = [f"{args.output_dir}/{data_idxs2ys[d]}/image{data_idx2label_idx[d]}" for d in range(len(data))]

    # for f in data_idx2file_base:
    #     if not os.path.exists(os.path.dirname(f)): os.makedirs(f)

    # z_dims = get_z_dims(model)

    # for e in tqdm(range(args.epochs), desc="(1/2) sampled epochs"):

    #     with torch.no_grad():
    #         for batch_idx,(xs,ys) in tqdm(enumerate(loader), desc="Batches", leave=False, total=len(loader)):
    #             data_idxs = torch.arange(len(xs)) + batch_idx * args.bs

    #             with autocast():
    #                 cx = corruptor(xs.to(device, non_blocking=True))
    #                 codes = [generate_z(args.bs, z, sample_method="normal") for z in z_dims]
    #                 fx = model(cx, codes, loi=-1)

    #             fx = [image for image in fx]
    #             files = [f"{data_idx2file_base[d]}_aug{e}.jpg" for d in data_idxs]

    #             for image,file in zip(fx,files):
    #                 save_image(image, file)
