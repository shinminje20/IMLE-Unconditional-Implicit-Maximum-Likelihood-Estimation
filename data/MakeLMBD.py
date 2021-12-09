"""Creates an LMDB dataset out of an ImageFolder-organized folder of images, but
with class-first organization."""

import argparse
import os
import lmdb
from PIL import Image

from DataUtils import *

def get_valid_image_file(image_file, size):
    """Returns and creates an image created from [image_file] that is
    [size]x[size], while preserving aspect ratio. **Preserving aspect ratio is
    important since we want to stay on the real manifold.**

    The image is scaled so that its largest dimension is [size], and is then
    padded so the result is [size]x[size].

    Args:
    image_file  -- the image file
    size        -- the size to make the image
    """
    image = Image.open(image_file)
    w, h = image.size
    resize_ratio = min(size / w, size / h)
    new_w, new_h = int(w * resize_ratio), int(h * resize_ratio)
    image = image.resize((new_w, new_h), resample=Image.BICUBIC)

    background = Image.new(image.mode, (size, size), (0, 0, 0))
    background



    if image.size[0] < size:
        back
    elif image.size[1] < size:

    else:
        assert w == h


def create_lmdb_for_category(cls_split_path, lmdb_save_paths, scales, map_sizes,
    size):
    """
    """
    # Many images won't be [size]x[size]. To handle this, we resize the image so
    # that the minimum side length is [size], and then crop a [size]x[size]
    # square from the center of the image
    image_files = [f"{cls_split_path}/{f}" for f in ost.listdir(cls_split_path)]
    images = [get_valid_image(image_file, size) for image_file in image_files]

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="LMDB dataset creation")
    P.add_argument("--image_folder", required=True,
        help="path to folder containing images")
    P.add_argument("--size", type=int, default=64,
        help="base image size")
    P.add_argument("--scales", default=[.0625, .125, .25, .5, 1], type=int, nargs="+",
        help="image resultions to create")
    args = P.parse_args()

    num_bad, num_good = 0, 0
    # Make sure all included images are at least [args.size] x [args.size]:
    for dir_1 in os.listdir(args.image_folder):
        if os.path.isdir(f"{args.image_folder}/{dir_1}"):
            for dir_2 in os.listdir(f"{args.image_folder}/{dir_1}"):
                for f in os.listdir(f"{args.image_folder}/{dir_1}/{dir_2}"):
                    file = f"{args.image_folder}/{dir_1}/{dir_2}/{f}"
                    image = Image.open(file)
                    w, h = image.size
                    if w < args.size or h < args.size:
                        num_bad += 1
                        image.show()
                    else:
                        num_good += 1

    print(num_good, num_bad)

    assert 0
    result_path = f"{data_dir}/lmdb/{dataset}"
    tqdm.write(f"Will save data to {result_path}")

    # If the input folder isn't class-first, make a version of it that is
    dirs_under_image_folder = [d.lower() for d in os.listdir(args.image_folder)]
    if "train" in dirs_under_image_folder and "test" in dirs_under_image_folder:
        tqdm.write("Detected --image_folder to be split-first. Making a copy that is class-first.")
        cls_first_image_folder = make_cls_first(args.image_folder)
    else:
        tqdm.write("Detected --image_folder to be class-first.")
        cls_first_image_folder = args.image_folder

    lmbd_save_paths = [f"{result_path}_{split}_{int(args.size * scale)}.lmdb"
        for scale in args.scales]
    map_sizes = [0 for _ in args.scales]

    splits = ["train", "val", "test"]
    for split in tqdm(splits, desc="Iterating over splits"):
        for idx,cls in tqdm(enumerate(os.listdir(cls_first_image_folder))):
            create_lmdb_for_category(f"{cls_first_image_folder}/{cls}/{split}",
                                     lmbd_save_paths,
                                     args.scales,
                                     map_sizes,
                                     args.size)
