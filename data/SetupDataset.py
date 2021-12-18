"""Downloads and sets up a dataset."""

import argparse
from collections import defaultdict
import gdown
import zipfile

from DataUtils import *

def gdown_unzip(url, result):
    """Downloads the file at Google drive URL [url], unzips it, and removes any
    hidden files that are not `.` and `..`.
    """
    zip_path = f"{data_dir}/{result}.zip"
    gdown.download(url, zip_path, quiet=False)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(path=data_dir)
    os.remove(zip_path)
    remove_bad_files(data_dir)

    return f"{data_dir}/{result}"

data2url = {
    # Data from https://mtl.yyliu.net/download/Lmzjm9tX.html
    "miniImagenet": "https://drive.google.com/u/1/uc?id=15iRcb_h5od0GsTkBGRmVTR8LyfJGuJ2k&export=download",
    # Data prepared via https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4
    "tinyImagenet": "https://drive.google.com/u/1/uc?id=1mCQOMcVbN0XT_uwcUjU1gdMvVRpBcX3w&export=download",
}

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="Dataset downloading and creation")
    P.add_argument("--data", choices=["tinyImagenet", "miniImagenet"],
        help="also make a class-first dataset split")
    P.add_argument("--also_cls_first", action="store_true",
        help="also make a class-first dataset split")
    P.add_argument("--size", default=None, type=int,
        help="size to make the images if different from 84x84")
    args = P.parse_args()

    dataset_dir = gdown_unzip(data2url[args.data], args.data)
    tqdm.write("Dataset created")

    if args.size is not None:
        resize_dataset(dataset_dir, (args.size, args.size))
        tqdm.write(f"Dataset resized to {args.size}x{args.size}")

    if args.also_cls_first:
        make_cls_first(dataset_dir)
        tqdm.write(f"Made a class-first copy of the dataset")
