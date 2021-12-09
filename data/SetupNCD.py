import os
import gdown
import zipfile
import shutil

url = "https://drive.google.com/u/0/uc?id=1hWvuwH1ZXghBWWVMvsPeEs0Z-g1zRJhp&export=download"
zip_path = "./ColorfulOriginal.zip"
path = "./data/ColorfulOriginal"
base_test_idxs = []
base_val_idxs = [] # We're not actually interested in test accuracy
split_names = ["train", "val", "test", "all-data"]

try:
    shutil.rmtree(path)
except:
    pass

################################################################################
# Download and extract the ColorfulOriginals dataset
################################################################################
gdown.download(url, zip_path, quiet=False)
with zipfile.ZipFile(zip_path) as z:
    z.extractall("./data")
os.remove(zip_path)

################################################################################
# Partition the dataset into the correct splits
################################################################################

D = {s: [] for s in split_names}
dirs_to_delete = []
for dir in os.listdir(path):
    if not os.path.isdir(f"{path}/{dir}"):
        continue

    for idx,img in enumerate(sorted(os.listdir(f"{path}/{dir}"))):
        file = f"{path}/{dir}/{img}"

        D["all-data"].append(file)

        if idx in base_test_idxs:
            D["test"].append(file)
        elif idx in base_val_idxs:
            D["val"].append(file)
        else:
            D["train"].append(file)

    dirs_to_delete.append(f"{path}/{dir}")


################################################################################
# Make folders based on the splits
################################################################################

for s in split_names:
    split_path = f"{path}/{s}"
    if not os.path.exists(split_path):
        os.makedirs(split_path)

    for file in D[s]:
        filename = os.path.basename(file)

        if not s == "all-data":
            filedir = f"{split_path}/{os.path.basename(os.path.dirname(file))}"
        else:
            filedir = split_path

        if not os.path.exists(filedir): os.makedirs(filedir)

        shutil.copy(file, f"{filedir}/{filename}")

################################################################################
# Delete now-empty folders
################################################################################
for dir in dirs_to_delete:
    shutil.rmtree(dir)
