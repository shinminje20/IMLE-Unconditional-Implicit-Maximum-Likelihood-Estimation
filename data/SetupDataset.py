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
    tqdm.write("Unzipping dataset...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(path=data_dir)
    os.remove(zip_path)
    tqdm.write("Removing potentially bad files...")
    remove_bad_files(data_dir)

    return f"{data_dir}/{result}"

data2url = {
    # Data from the Cedar cluster
    "camnet3": "https://drive.google.com/u/1/uc?id=1Yly8SJitGnA25opej57TRLlSyoGmqeIN&export=download",
    # Data from https://www.kaggle.com/swaroopkml/cifar10-pngs-in-folders
    "cifar10": "https://drive.google.com/u/1/uc?id=1_CILJzGysgPLflV2HxXSNMjVzlJat6hs&export=download",
    # Data from https://mtl.yyliu.net/download/Lmzjm9tX.html
    "miniImagenet": "https://drive.google.com/u/1/uc?id=15iRcb_h5od0GsTkBGRmVTR8LyfJGuJ2k&export=download",
    # Data prepared via https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4
    "tinyImagenet": "https://drive.google.com/u/1/uc?id=1mCQOMcVbN0XT_uwcUjU1gdMvVRpBcX3w&export=download",
}

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="Dataset downloading and creation")
    P.add_argument("--data",
        choices=["tinyImagenet", "miniImagenet", "camnet3", "cifar10", "camnet3_lmdb", "camnet3_deci_lmdb", "camnet3_centi_lmdb"],
        help="also make a class-first dataset split")
    P.add_argument("--also_cls_first", action="store_true",
        help="also make a class-first dataset split")
    P.add_argument("--sizes", default=[16, 32, 64, 128, 256], type=int,
        nargs="+",
        help="sizes to make the images. -1 or zero for no resizing")
    P.add_argument("--use_existing", type=str,
        help="use existing dataset instead of downloading one")
    args = P.parse_args()

    if args.use_existing is None:
        tqdm.write(f"----- Downloading base dataset -----")
        dataset_dir = gdown_unzip(data2url[args.data], args.data)
    else:
        tqdm.write(f"----- Using existing dataset {args.use_existing} -----")
        args.data = args.use_existing
        dataset_dir = args.use_existing

    if len(args.sizes) == 1 and args.sizes[0] <= 0:
        tqmd.write("----- Not resizing dataset -----")
        all_datasets = [dataset_dir]
    else:
        tqdm.write(f"----- Generating new resolutions: {args.sizes} -----")
        all_datasets = [dataset_dir] + [resize_dataset(dataset_dir, s)
            for s in tqdm(args.sizes)]

    if args.also_cls_first and not "lmdb" in args.data:
        tqdm.write(f"----- Making class-first copies -----")
        for dataset in all_datasets:
            make_cls_first(dataset)
            tqdm.write(f"Made a class-first copy of {dataset}")
