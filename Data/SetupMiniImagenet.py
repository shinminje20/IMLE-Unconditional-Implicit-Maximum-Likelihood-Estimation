"""Script to set up the MiniImagenet dataset."""
from collections import defaultdict
import csv
import os
import gdown
import zipfile
import shutil

# Create directory for MiniImagenet
data_dir = os.path.dirname(os.path.abspath(__file__))
miniImagenet_dir = f"{data_dir}/miniImagenet"

# Download the data and splits
url = "https://drive.google.com/u/1/uc?id=1XmneyaXBZZyCZFn7UWnni1HIBuvWlFJB&export=download"
zip_path = f"{data_dir}/miniImagenet_data.zip"
gdown.download(url, zip_path, quiet=False)
with zipfile.ZipFile(zip_path) as z:
    z.extractall(path=data_dir)
os.remove(zip_path)

# Create splits so we can use an ImageFolder-based dataset. I'm using fewshot
# learning terminology because it'd be cool to investigate out-of-distribution
# generalization. Traditional splitting can be added later.
csv2split = {"train.csv": "base", "val.csv": "novel", "test.csv": "test"}

for split_file, split in csv2split.items():
    class2images = defaultdict(lambda: [])

    with open(f"{miniImagenet_dir}/{split_file}", newline="") as f:
        csv_reader = csv.reader(f)
        next(csv_reader) # Skip the first row
        for row in csv_reader:
            class2images[row[1]].append(row[0])

    os.makedirs(f"{miniImagenet_dir}/{split}")
    for cls,images in class2images.items():
        cls_dir = f"{miniImagenet_dir}/{split}/{cls}"
        os.makedirs(cls_dir)
        for image in images:
            os.rename(f"{miniImagenet_dir}/images/{image}", f"{cls_dir}/{image}")

# Of course, we maybe also want to train in a traditional train-val-test setup.
# The split sizes allow for training with the same amount of data as in the
# fewshot learning splits, which could be interesting!
for default_split in ["train", "val", "default_test"]:
    split_dir = f"{miniImagenet_dir}/{default_split}"
    if not os.path.exists("split_dir"):
        os.makedirs(split_dir)

for fewshot_split in csv2split.values():
    for dir in os.listdir(f"{miniImagenet_dir}/{fewshot_split}"):

        dir_path = f"{miniImagenet_dir}/{fewshot_split}/{dir}"
        n_images = len(list(os.listdir(dir_path)))
        os.makedirs(f"{miniImagenet_dir}/train/{dir}")
        os.makedirs(f"{miniImagenet_dir}/val/{dir}")
        os.makedirs(f"{miniImagenet_dir}/default_test/{dir}")

        for idx,file in enumerate(os.listdir(dir_path)):
            file_path = f"{dir_path}/{file}"

            if idx / n_images < .64:
                shutil.copy(file_path, f"{miniImagenet_dir}/train/{dir}/{file}")
            elif  idx / n_images  < .8:
                shutil.copy(file_path, f"{miniImagenet_dir}/val/{dir}/{file}")
            else:
                shutil.copy(file_path, f"{miniImagenet_dir}/default_test/{dir}/{file}")
