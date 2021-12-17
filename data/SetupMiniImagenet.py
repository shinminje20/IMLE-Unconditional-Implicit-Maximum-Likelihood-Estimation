"""
Script to set up the MiniImagenet dataset.
The images/splits inherent here are from
https://mtl.yyliu.net/download/Lmzjm9tX.html.
"""
import argparse

from DataUtils import *

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="SimCLR training")
    P.add_argument("--also_cls_first", action="store_true",
        help="Also make a class-first dataset split")
    args = P.parse_args()

    miniImagenet_dir = gdown_unzip(
        "https://drive.google.com/u/1/uc?id=15iRcb_h5od0GsTkBGRmVTR8LyfJGuJ2k&export=download",
        "miniImagenet")

    if args.also_cls_first:
        make_cls_first(miniImagenet_dir)
