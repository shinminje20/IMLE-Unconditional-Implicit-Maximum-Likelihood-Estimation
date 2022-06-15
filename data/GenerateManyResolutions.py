import argparse
import os
import random
import shutil
from DataUtils import *

def is_all_or_positive_int(value):
    if value == "all" or (value.isdecimal() and int(value) > 0):
        return value
    else:
        raise argparse.ArgumentTypeError(f"{value} Must be 'all' or positive integer")

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--data", required=True,
        help="Path to base dataset")
    P.add_argument("--res", nargs="+", type=int,
        help="Resolutions to generate data at")
    P.add_argument("--splits", required=True, nargs="+",
        help="Splits to create data for")
    P.add_argument("--ncls", required=True, nargs="+",
        type=is_all_or_positive_int,
        help="Splits to create data for")
    P.add_argument("--npc", required=True, nargs="+",
        type=is_all_or_positive_int,
        help="Splits to create data for")
    P.add_argument("--seed", type=int, default=0,
        help="Splits to create data for")
    P.add_argument("--new_name", type=str, required=True,
        help="new name if desired")
    P.add_argument("--splits_have_same_classes",
        type=int, default=1, choices=[0, 1],
        help="new name if desired")
    args = P.parse_args()

    ############################################################################
    # Ensure arguments are valid. It must be the case that
    # a) the base dataset exists
    # b) all desired splits actually exist in the base dataset
    # c) the number of requested splits, ncls, and npc match
    ############################################################################
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Couldn't find {args.data}")

    for s in args.splits:
        if not os.path.exists(f"{args.data}/{s}"):
            raise FileNotFoundError(f"Couldn't find {args.data}/{s}")

    if not len(args.splits) == len(args.ncls) == len(args.npc):
        raise ValueError(f"Length of --splits, --npc, and --ncls must match, but got {args.splits}, {args.npc}, and {args.ncls}")

    if args.splits_have_same_classes:
        first_split_classes = os.listdir(f"{args.data}/{args.splits[0]}")
        for s in args.splits:
            split_classes = os.listdir(f"{args.data}/{s}")
            if not split_classes == first_split_classes:
                raise ValueError(f"Got classes {first_split_classes} for the first split but {split_classes} for split {s}. Unset --splits_have_same_classes to have different classes in each split")

    ############################################################################
    # Get the data
    ############################################################################
    if args.splits_have_same_classes:
        # In this case, each split shares the same classes, but not necessarily
        # the same examples from each class.
        classes = os.listdir(f"{args.data}/{args.splits[0]}")
        classes = random.sample(classes,
            k=len(classes) if args.ncls[0] == "all" else int(args.ncls[0]))
        split2classes = {s: classes for s in args.splits}
    else:
        # In this case, the ith listed split has the ith listed number of
        # classes, and each split might not have the same classes
        split2classes = {}
        for s,ncls in zip(args.splits, args.ncls):
            classes = os.listdir(f"{args.data}/{s}")
            split2classes[s] = random.sample(classes,
                k=len(classes) if ncls == "all" else int(ncls))
    
    split2classes2files = defaultdict(lambda: defaultdict(lambda: []))
    for s,npc in zip(args.splits, args.npc):
        for c in split2classes[s]:
            files = [f"{args.data}/{s}/{c}/{f}"
                for f in os.listdir(f"{args.data}/{s}/{c}")] 
            files = random.sample(files,
                k=len(files) if npc == "all" else int(npc))
            split2classes2files[s][c] = files

    ############################################################################
    # Resize the data
    ############################################################################
    if os.path.exists(args.new_name): shutil.rmtree(args.new_name)

    for s in args.splits:
        for c in split2classes2files[s]:
            if not os.path.exists(f"{args.new_name}/{s}/{c}"):
                os.makedirs(f"{args.new_name}/{s}/{c}")
            for f in split2classes2files[s][c]:
                shutil.copy(f, f.replace(args.data, args.new_name))

    for r in args.res:
        resize_dataset(args.new_name, r)

    shutil.rmtree(args.new_name)
