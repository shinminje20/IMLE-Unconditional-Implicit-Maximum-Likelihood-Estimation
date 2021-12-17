"""
Script to set up the tinyImagenet dataset.
The images/splits inherent here are from prepared via
https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4.
"""
import argparse

from DataUtils import *

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="SimCLR training")
    P.add_argument("--also_cls_first", action="store_true",
        help="Also make a class-first dataset split")
    args = P.parse_args()

    tinyImagenet_dir = gdown_unzip(
        "https://drive.google.com/u/1/uc?id=1mCQOMcVbN0XT_uwcUjU1gdMvVRpBcX3w&export=download",
        "tinyImagenet")

    if args.also_cls_first:
        make_cls_first(f"{data_dir}/{tinyImagenet_dir}")
