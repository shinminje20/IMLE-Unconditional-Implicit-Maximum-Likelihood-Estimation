import argparse
import os
from PIL import Image
import shutil
from tqdm import tqdm

def zero_pad_and_resize(file, size=256):
    """Returns the image in [file] with the smaller side zero padded to match
    the larger side, and the result downsampled to [size].
    """
    with Image.open(file) as image:
        h, w = image.size

        # In this case, we need increase the width via padding and downsample
        if h > w and h >= size:
            result = Image.new(mode=image.mode, size=(h, h))
            result.paste(image, (0, (h - w) // 2))
        # In this case, we need increase the height via padding and downsample
        elif w > h and w >= size:
            result = Image.new(mode=image.mode, size=(w, w))
            result.paste(image, ((w - h) // 2, 0))
        # In this case, the image is already square and we need to downsample
        elif w == h and w >= size:
            result = image
        # In this case, we need to increase both dimensions
        elif h < size and w < size:
            result = Image.new(mode=image.mode, size=(size, size))
            result.paste(image, ((size - h) // 2, (size - w) // 2))
        else:
            raise ValueError(f"Impossible Case A")

        result = result.resize((size, size), resample=Image.BICUBIC)
        return result

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--data", required=True, type=str,
        help="Path to data folder. Should be structured as data_folder/split/class/image")
    P.add_argument("--replace", choices=[0, 1], type=int, default=0,
        help="Replace files rather than create a copy")
    P.add_argument("--output_dir", default=None, type=str,
        help="File create results in. Defaults to the current data directory")
    P.add_argument("--size", type=int, default=256,
        help="Output image size")
    P.add_argument("--splits", nargs="+", default=None,
        help="Splits to resize")
    args = P.parse_args()

    if args.replace and args.output_dir is None:
        args.output_dir = args.data
        tqdm.write("--replace specified, so new images will be written on top of old ones")
    elif not args.replace and not args.output_dir is None:
        tqdm.write(f"--replace not specified; new images will be written to {args.output_dir}")
    elif not args.replace and args.output_dir is None:
        raise ValueError("If you don't desire files to be replaced, you must specify an output directory")
    elif args.replace and not args.output_dir is None:
        raise ValueError("If you wish to replace files, you may not specify an output directory")
    else:
        raise ValueError(f"Impossible Case B")

    args.splits = os.listdir(args.data) if args.splits is None else args.splits

    for split in tqdm(args.splits, desc="splits"):
        for cls in tqdm(os.listdir(f"{args.data}/{split}"), desc="classes", leave=False):
            for image_file in os.listdir(f"{args.data}/{split}/{cls}"):
                new_image = zero_pad_and_resize(
                    f"{args.data}/{split}/{cls}/{image_file}",
                    size=args.size)

                if not os.path.exists(f"{args.output_dir}/{split}/{cls}"):
                    os.makedirs(f"{args.output_dir}/{split}/{cls}")

                new_image.save(f"{args.output_dir}/{split}/{cls}/{image_file}")
