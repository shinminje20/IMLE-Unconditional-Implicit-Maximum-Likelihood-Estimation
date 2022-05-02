"""Subsamples a dataset in a specific way."""

import argparse
from collections import defaultdict
import os
import numpy as np
from DataUtils import *

def flatten(xs):
    """Returns collection [xs] after recursively flattening into a list."""
    if isinstance(xs, list) or isinstance(xs, set) or isinstance(xs, tuple):
        result = []
        for x in xs:
            result += flatten(x)
        return result
    else:
        return [xs]

def get_smaller_dataset(source, splits=["train", "val", "test"],
    split_n_cls=[10, 10, 10], split_npc=[100, 100, 100], seed=0, new_name=None):
    """Creates and returns the path to a smaller dataset containing a subsample
    of [source].

    Args:
    source      -- path to the data source. Images in [source] should be found
                    via the paths like source/split/class/image.
    splits      -- names of splits to subsample
    split_n_cls -- list of integers where the ith integer is the number of
                    classes to include when subsampling from the ith split, or
                    'all' to include all classes in the ith split
    split_npc   -- list of integers where the ith integer is the number of
                    imags per class to include when subsampling from the ith
                    split, or 'all' to include all images in a class for the ith
                    split
    seed        -- seed for subsampling
    """
    def get_images(class_path, n):
        """Returns a list of [n] images sampled randomly from [class_path]."""
        n = len(list(os.listdir(class_path))) if n == "all" else int(n)
        selected_images = random.sample(os.listdir(class_path), n)
        return [f"{class_path}/{s}" for s in selected_images]

    ############################################################################
    # Check if inputs are valid
    ############################################################################
    if not (len(splits) == len(split_n_cls) and len(splits) == len(split_npc)):
        raise ValueError()

    source = f"{data_dir}/{source}" if not "/" in source else source
    if not os.path.exists(source):
        raise FileNotFoundError(f"Couldn't find specified dataset {source}")

    for split in os.listdir(source):
        if not os.path.exists(f"{source}/{split}"):
            raise ValueError(f"Split '{split}' requested but could not be found at {source}/{split}")

    ############################################################################
    # Generate the new splits
    ############################################################################

    # Resort arguments so that they are in alphabetical order
    # sorted_idxs = np.array(sorted(range(len(splits)), key=lambda i: splits[i]))
    # split_n_cls = np.array(split_n_cls)[sorted_idxs]
    # split_npc = np.array(split_npc)[sorted_idxs]
    #
    # print(sorted_idxs, split_n_cls, split_npc)

    split2paths = defaultdict(lambda: [])
    for split, n_cls, npc in zip(splits, split_n_cls, split_npc):

        print(split, n_cls, npc)

        random.seed(seed)
        classes = random.sample(
            os.listdir(f"{dataset}/{split}"),
            len(os.listdir(f"{dataset}/{split}")) if n_cls == "all" else n_cls)
        images = [get_images(f"{dataset}/{split}/{c}", npc) for c in classes]
        split2paths[split] = flatten(images)

    ############################################################################
    # Copy files
    ############################################################################
    source_size = find_data_res(source)
    desc = [f"{split}_{n_cls}_{npc}"
            for split,n_cls,npc in zip(splits, split_n_cls, split_npc)]

    if new_name is None:
        new_source = f"{source.replace(source_size, '')}-{'-'.join(desc)}_{source_size}"
    else:
        new_source = f"{data_dir}/{new_name}_{source_size}"

    new_source = fix_data_path(new_source)

    for split in splits:
        for path in split2paths[split]:
            path_class = os.path.basename(os.path.dirname(path))
            path_dir = f"{new_source}/{split}/{path_class}"
            if not os.path.exists(path_dir):
                os.makedirs(path_dir)
            shutil.copy(path, f"{path_dir}/{os.path.basename(path)}")

    return new_source

def has_suffix(s):
    """Returns if string [s] has a data suffix."""
    return "_deci" in s or "_centi" in s or "_milli" in s

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="Dataset subsampling for faster CaMNetting")
    P.add_argument("--datasets", required=True, type=str, nargs="+",
        help="path to dataset to subsample. Note that there's some custom logic if you use a * character")
    P.add_argument("--splits", type=str, nargs="+",
        help="splits of --dataset to subsample")
    P.add_argument("--n_cls", type=str, nargs="+",
        help="number of classes per split, one number per split")
    P.add_argument("--npc", type=str, nargs="+",
        help="number of images per class, one number per split")
    P.add_argument("--seed", type=int, default=0,
        help="random seed")
    P.add_argument("--new_name", type=str,
        help="new name if desired")
    P.add_argument("--also_cls_first", action="store_true",
        help="also make a class-first dataset split")
    args = P.parse_args()

    if not has_res(args.datasets[0]):
        data_name = args.datasets[0].replace("*", "")
        datasets = [c for c in os.listdir(data_dir) if data_name in c]
        datasets = [c for c in datasets if not has_suffix(c)]
        datasets = [c for c in datasets if has_res(c)]
        args.datasets = datasets
        tqdm.write(f"Using {datasets} as datasets")

    args.datasets = [d.rstrip("/") for d in args.datasets]
    args.npc = ["all" if n == "all" else int(n) for n in args.npc]
    args.n_cls = ["all" if n == "all" else int(n) for n in args.n_cls]

    all_datasets = []
    for dataset in args.datasets:
        all_datasets.append(get_smaller_dataset(dataset, splits=args.splits,
                                                split_npc=args.npc,
                                                split_n_cls=args.n_cls,
                                                seed=args.seed,
                                                new_name=args.new_name))

    if args.also_cls_first:
        for dataset in all_datasets:
            make_cls_first(dataset)
            tqdm.write(f"Made a class-first copy of {dataset}")
