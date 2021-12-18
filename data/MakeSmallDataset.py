"""Subsamples a dataset in a specific way."""

import argparse

from DataUtils import *

def get_smaller_dataset(dataset, n_cls_tr=10, n_cls_val=5, npc_tr=100,
    npc_val=100, seed=0):
    """Creates a dataset in the same folder as [dataset] that is a subset of
    [dataset], and returns the absolute path to the new dataset.

    Args:
    dataset         -- absolute path to the dataset to create from
    n_classes_tr    -- number of classes in the training split
    n_classes_val   -- number of classes in the val split
    npc_tr          -- number of images per class in the training split, or
                        'all' for all of them
    npc_val         -- number of images per class in the validation split, or
                        'all' for all of them
    seed            -- random seed to use
    """
    def get_images(class_path, n):
        """Returns a list of [n] images sampled randomly from [class_path]."""
        n = len(list(os.listdir(class_path))) if n == "all" else n
        selected_images = random.sample(os.listdir(class_path), n)
        return [f"{class_path}/{s}" for s in selected_images]

    # If [dataset] isn't an absolute path, correct it!
    dataset = f"{data_dir}/{dataset}" if not "/" in dataset else dataset

    # Subsample the training and validation images
    random.seed(seed)
    cls_tr = random.sample(os.listdir(f"{dataset}/train"), n_cls_tr)
    images_tr = [get_images(f"{dataset}/train/{c}", npc_tr) for c in cls_tr]
    cls_val = random.sample(os.listdir(f"{dataset}/val"), n_cls_val)
    images_val = [get_images(f"{dataset}/val/{c}", npc_val) for c in cls_val]

    # Name the dataset and make a folder for it
    new_dataset = f"{dataset}-{n_cls_tr}-{npc_tr}-{n_cls_val}-{npc_val}-{seed}"
    if os.path.exists(new_dataset): shutil.rmtree(new_dataset)
    os.makedirs(new_dataset)

    # Copy the image files to the locations for the new dataset
    for cls,images in zip(cls_tr, images_tr):
        cls_path = f"{new_dataset}/train/{cls}"
        os.makedirs(cls_path)
        for image in images:
            shutil.copy(image, f"{cls_path}/{os.path.basename(image)}")
    for cls,images in zip(cls_val, images_val):
        cls_path = f"{new_dataset}/val/{cls}"
        os.makedirs(cls_path)
        for image in images:
            shutil.copy(image, f"{cls_path}/{os.path.basename(image)}")

    return new_dataset

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="Dataset subsampling for faster CaMNetting")
    P.add_argument("--data", choices=["tinyImagenet", "miniImagenet"],
        required=True,
        help="also make a class-first dataset split")
    P.add_argument("--n_cls_tr", required=True, type=int,
        help="number of training classes")
    P.add_argument("--npc_tr", default=-1, type=int,
        help="number of images per training class. -1 for all of them")
    P.add_argument("--n_cls_val", required=True, type=int,
        help="number of validation classes")
    P.add_argument("--npc_val", default=-1, type=int,
        help="number of images per validation class. -1 for all of them")
    P.add_argument("--seed", default=0, type=int,
        help="random seed")
    P.add_argument("--also_cls_first", action="store_true",
        help="also make a class-first dataset split")
    args = P.parse_args()

    args.npc_tr = "all" if args.npc_tr == -1 else args.npc_tr
    args.npc_val = "all" if args.npc_val == -1 else args.npc_val

    new_dataset = get_smaller_dataset(args.data,
        n_cls_tr=args.n_cls_tr,
        n_cls_val=args.n_cls_val,
        npc_tr=args.npc_tr,
        npc_val=args.npc_val,
        seed=args.seed)

    if args.also_cls_first:
        make_cls_first(new_dataset)
        tqdm.write(f"Made a class-first copy of the dataset")
