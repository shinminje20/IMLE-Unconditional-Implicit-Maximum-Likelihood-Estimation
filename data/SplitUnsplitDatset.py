import argparse
from collections import defaultdict
import os
import shutil
import random

def rand_partition(xs, frac):
    """Returns an (a, b) tuple where [a] contains a [frac]-fraction of the
    elements of [xs], and [b] contains the rest.
    """
    idxs = set(random.sample(range(len(xs)), k=int(len(xs) * frac)))
    a = [x for idx,x in enumerate(xs) if idx in idxs]
    b = [x for idx,x in enumerate(xs) if not idx in idxs]
    return a, b

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--data")
    P.add_argument("--replace", type=int, default=0, choices=[0, 1],
        help="Whether to replace the dataset given in --data")
    P.add_argument("--splits", required=True, nargs="+",
        help="Split names")
    P.add_argument("--split_fracs", required=True, nargs="+", type=float,
        help="Fraction of data for each split")
    args = P.parse_args()

    assert len(args.split_fracs) == len(args.splits)
    assert sum(args.split_fracs) == 1

    class2images = {}
    for c in os.listdir(args.data):
        images = [f"{args.data}/{c}/{f}" for f in os.listdir(f"{args.data}/{c}")]
        class2images[c] = images

    split2classes2images = defaultdict(lambda: defaultdict(lambda: []))
    for idx,(s,frac) in enumerate(zip(args.splits, args.split_fracs)):
        for c in class2images:
            if idx == len(args.splits) - 1:
                split2classes2images[s][c] = class2images[c]
            else:
                for_split, remaining = rand_partition(class2images[c], frac)
                split2classes2images[s][c] = for_split
                class2images[c] = remaining

    for s in split2classes2images:
        for c in split2classes2images[s]:
            for image in split2classes2images[s][c]:
                if args.replace:
                    raise NotImplementedError()
                else:
                    new_name = image.replace(f"{args.data}", f"{args.data}/{s}")
                    if not os.path.exists(os.path.dirname(new_name)):
                        os.makedirs(os.path.dirname(new_name))
                    shutil.copy(image, new_name)
