import argparse
from collections import defaultdict
import os
import shutil

data_dir = os.path.dirname(os.path.abspath(__file__))
split2camnet_split = {"train": "train", "val": "validation", "test": "test"}

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="Preparing data for CaMNet")
    P.add_argument("--data_folder", required=True, type=str,
        help="folder containing images with folder/split/class/image organization")
    args = P.parse_args()

    class2split2file = defaultdict(lambda: defaultdict(lambda: []))

    for split in ["train", "val", "test"]:
        for cls in os.listdir(f"{args.data_folder}/{split}"):
            for image in os.listdir(f"{args.data_folder}/{split}/{cls}"):
                image = f"{args.data_folder}/{split}/{cls}/{image}"
                class2split2file[cls][split].append(image)

    for cls in class2split2file:
        for split in class2split2file[cls]:

            folder = f"{data_dir}/camnet_data/{os.path.basename(args.data_folder)}/{cls}/{split2camnet_split[split]}"
            if not os.path.exists(folder):
                os.makedirs(folder)

            for image_file in class2split2file[cls][split]:
                shutil.copy(image_file, f"{folder}/{os.path.basename(image_file)}")
