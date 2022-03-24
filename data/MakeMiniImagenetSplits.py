import argparse
import csv
import os
import shutil

"""Download the split files from here:
https://github.com/yaoyao-liu/mini-imagenet-tools/tree/main/csv_files
"""

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--train_csv", required=True, type=str)
    P.add_argument("--val_csv", required=True, type=str)
    P.add_argument("--test_csv", required=True, type=str)
    P.add_argument("--data_dir", type=str, required=True)

    args = P.parse_args()

    train_classes = set()
    val_classes = set()
    test_classes = set()

    with open(args.train_csv, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for idx,(_,cls) in enumerate(reader):
            if idx == 0:
                continue
            else:
                train_classes.add(cls)

    with open(args.val_csv, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for idx,(_,cls) in enumerate(reader):
            if idx == 0:
                continue
            else:
                val_classes.add(cls)

    with open(args.test_csv, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for idx,(_,cls) in enumerate(reader):
            if idx == 0:
                continue
            else:
                test_classes.add(cls)

    for classes,split in zip([train_classes, val_classes, test_classes], ["train", "val", "test"]):
        split_path = f"{args.data_dir}/{split}"

        if not os.path.exists(split_path):
            os.makedirs(split_path)

        for folder in os.listdir(args.data_dir):
            if folder in classes:
                shutil.move(f"{args.data_dir}/{folder}", f"{split_path}/{folder}")
