"""
Script to set up the MiniImagenet dataset.
The images/splits inherent here are from
https://mtl.yyliu.net/download/Lmzjm9tX.html.
"""
import argparse
from collections import defaultdict
import csv
import os
import gdown
import zipfile
import shutil

from DataUtils import *

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="SimCLR training")
    P.add_argument("--also_cls_first", action="store_true",
        help="Also make a class-first dataset split")
    args = P.parse_args()

    # Create directory for MiniImagenet
    # data_dir = os.path.dirname(os.path.abspath(__file__))
    miniImagenet_dir = f"{data_dir}/miniImagenet"

    # Download the data and splits
    url = "https://drive.google.com/u/1/uc?id=15iRcb_h5od0GsTkBGRmVTR8LyfJGuJ2k&export=download"
    zip_path = f"{data_dir}/miniImagenet_data.zip"
    gdown.download(url, zip_path, quiet=False)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(path=data_dir)
    os.remove(zip_path)

    if args.also_cls_first:
        make_cls_first(miniImagenet_dir)
