import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--data", type=str, required=True,
        help="name of resized dataset, not including split")
    args = P.parse_args()

    args.data = f"{args.data.rstrip('/')}/train"
    data = ImageFolder(args.data, transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=256, drop_last=False, num_workers=24)
    
    channel_totals = torch.zeros(3, device=device)
    for x,_ in tqdm(loader, desc="Calculating mean", leave=True):
        mean = torch.mean(x.to(device), dim=[2, 3])
        mean = torch.sum(mean, axis=0)
        channel_totals += mean
    channel_means = channel_totals / len(data)
    
    channel_stds = torch.zeros(3, device=device)
    for x,_ in tqdm(loader, desc="Calculating standard deviation", leave=True):
        bs, c, h, w = x.shape
        mean = channel_means.view(1, 3, 1, 1).expand(bs, -1, h, w)
        std = torch.square(x.to(device) - mean)
        std = torch.sum(torch.mean(std, dim=[2, 3]), axis=0)
        channel_stds += std
    channel_stds = torch.sqrt(channel_stds / len(data))

    channel_means = channel_means.cpu().tolist()
    channel_stds = channel_stds.cpu().tolist()

    tqdm.write(f"Channel means: {channel_means} | channel stds: {channel_stds}")
    
