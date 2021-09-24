import os
import torch
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"

################################################################################
# I/O Utils
################################################################################
state_sep_str = "=" * 40

def opts_str(args):
    """Returns the options string of [args]."""
    return f"-{'-'.join(args.options)}-" if len(args.options) > 0 else "-"

def suffix_str(args):
    """Returns the suffix string of [args]."""
    return f"-{args.suffix}" if not args.suffix == "" else ""

def load_experiment(file):
    """Returns a (model, optimizer, last_epoch, args, results) tuple from
    [file].
    """
    data = torch.load(file)
    model = data["model"].to(device)
    last_epoch = data["last_epoch"]
    optimizer = data["optimizer"]
    args = data["args"]
    results = data["results"]
    return model, optimizer, last_epoch, args, results

def save_model(model, optimizer, last_epoch, args, results, folder):
    """Saves input experiment objects to the [last_epoch].pt file [folder]."""
    torch.save({
        "model": model.cpu(),
        "optimizer": optimizer,
        "last_epoch": last_epoch,
        "args": args,
        "results": results,
    }, f"{folder}/{last_epoch}.pt")

def generator_folder(model, optimizer, last_epoch, args, results):
    """Returns the folder to save a generator trained with [args] to."""
    folder = f"Models/generator-{args.data}-{opts_str(args)}{suffix_str(args)}"
    if not os.path.exists(folder): os.makedirs(folder)
    return folder

def resnet_folder(model, optimizer, last_epoch, args, results):
    """Returns the folder to save a resnet trained with [args] to."""
    folder = f"Models/resnets-{args.backbone}-{args.data}{opts_str(args)}{suffix_str(args)}"
    if not os.path.exists(folder): os.makedirs(folder)
    return folder
