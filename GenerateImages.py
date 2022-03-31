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

class PreAugmentedImageFolder(Dataset):
    """A (nearly) drop-in replacement for an ImageFolder for use where some
    augmentations are pre-generated.

    Args:
    source      -- path to an folder of images laid out for an ImageFolder.
                    However, the layout and file naming convention is more
                    constrained:

                    📂 source
                     ┣ 📂 class1
                     ┃  ┣ 📜 image1_aug0.jpg
                     ┃  ┣ 📜 image1_aug1.jpg
                     ┃  ┣ 📜 ...
                     ┃  ┣ 📜 image1_augN.jpg
                     ┃  ┗ ...
                     ┣ 📂class2
                    ┃  ┣ 📜 image1_aug0.jpg
                    ┃  ┣ 📜 image1_aug1.jpg
                    ┃  ┣ 📜 ...
                    ┃  ┣ 📜 image1_augN.jpg
                    ┃  ┗ ...
    transform   -- transform to apply
    """
    def __init__(self, source, transform=None):
        ########################################################################
        # Validate that we can actually build the dataset. It must be that:
        # 1) All classes have the same number of images
        # 3) Each class must have the same number of augmentations per image
        ########################################################################
        n_images = [len(os.listdir(c)) for c in os.listdir(source)]
        if not len(set(n_images)) == 1:
            raise ValueError(f"All classes in `{source}` must have the same number of images, but got the following numbers: {n_images}")

        cls2n_augs = {}
        for cls in tqdm(os.listdir(source), leave=False, desc="Validating dataset"):
            img2augs = defaultdict(lambda: [])
            for image_aug in os.listdir(f"{source}/{cls}"):
                img = os.path.basename(image_aug)
                img = img[:image.find("_")]
                img2augs[img].add(image_aug)

            n_augs = [len(augs) for augs in img2augs.values()]
            if not len(set(n_augs)) == 1:
                raise ValueError(f"All images in {source}/{cls} must have the same number of augmentations, but got the following numbers: {n_augs}")
            cls2n_augs[cls] = n_augs[0]

        n_augs = [len(augs) for augs in cls2n_augs.values()]
        if not len(set(n_augs)) == 1:
            raise ValueError(f"All classes in {source}/{cls} must have the same number of augmentations per image, but got the following numbers: {n_augs}")

        ########################################################################
        # Construct the dataset
        ########################################################################
        super(PreAugmentedImageFolder, self).__init__()
        self.num_augs = n_augs[0]
        self.source = ImageFolder(source, transform=transform)
        self.source_len = len(self.source)

    def __len__(self): return len(self.source) // self.num_augs

    def __getitem__(self, idx):
        return self.source[idx * self.source_len + random.randint(0, self.num_augs)]

def convert_to_images(order_four_tensor, labels):
    """
    """

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
