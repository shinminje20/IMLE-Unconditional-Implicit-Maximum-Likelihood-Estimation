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
url = "https://drive.google.com/u/1/uc?id=1PYNmmi3d2YldGlJvC0uEo_RnYDSJy0G-&export=download"
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
