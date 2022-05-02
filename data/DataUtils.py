from collections import defaultdict
import os
from PIL import Image
import random
import shutil
from tqdm import tqdm
import re

data_dir = os.path.dirname(os.path.abspath(__file__))

def fix_data_path(path):
    """Corrects errors in a path that can happen due to data renaming."""
    return path.replace("__", "_").replace("_-", "-").replace("--", "-")
    
def has_res(data_name):
    try:
        find_data_res(data_name)
        return True
    except ValueError:
        return False

def find_data_res(data_name, return_int=False):
    """Returns the size of images in [data_name]. Concretely, this is the
    first instance of a substring 'A1...AnxB1...Bm' where 'A1...An' and
    'B1...Bm'can be interpreted as digits to an integer.
    """
    matches = re.findall('\d*x\d', data_name)
    if len(matches) == 0:
        raise ValueError()
    elif return_int:
        height, width = matches[0].split("x")
        return int(height), int(width)
    else:
        return matches[0]

def get_all_files(path, extensions=[], acc=set()):
    """Returns a list of all files under [path] with an extension in
    [extensions] or simply all the files if [extensions] is empty.
    """
    if os.path.isdir(path):
        for f in os.listdir(path):
            get_all_files(f"{path}/{f}", extensions=extensions, acc=acc)
    else:
        acc.add(path)
    return acc

def resize_dataset(path, new_size):
    """Generates a new dataset matching that under [path] but with the images
    resized to [new_size].
    """
    new_path = f"{path}_{new_size}x{new_size}"
    tqdm.write(f"\nWill output new dataset to {new_path}")

    files_list = get_all_files(path)
    for f in tqdm(files_list, desc=f"Resizing images to {new_size}x{new_size}"):
        # Ensure the folder we will put the image in exists
        new_folder = os.path.dirname(f).replace(path, new_path)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

        # Generate a resized copy of the image and save it to the new folder
        img = Image.open(f)
        new_img = img.resize((new_size, new_size), Image.BICUBIC)
        new_img.save(f.replace(path, new_path))

    return new_path

def remove_bad_files(f, bad_files=[".DS_Store"]):
    """Recursively removes bad files from folder or file [f]."""
    if os.path.isdir(f):
        for item in os.listdir(f):
            if item in bad_files:
                os.remove(f"{f}/{item}")
            elif os.path.isdir(f"{f}/{item}"):
                remove_bad_files(f"{f}/{item}")
            else:
                pass
    else:
        pass

def make_cls_first(data_folder, cls_first_folder=f"{data_dir}/cls_first"):
    """Returns a path to a data folder that is identical to [data_folder] but
    organized class first. This data folder is populated, ie. this function
    makes a copy of [data_folder] under [cls_first_folder] with the classes
    above splits in the file hierarchy. Therefore, an image normally accessed
    via

        data_folder/split/class/x.png

    would be accessed via

        cls_first_folder/data_folder/class/split/x.png

    Args:
    data_folder         -- split-first data folder
    cls_first_folder    -- path to folders containing class-first data
    """
    class2split2file = defaultdict(lambda: defaultdict(lambda: []))

    for split in os.listdir(data_folder):
        if os.path.isdir(f"{data_folder}/{split}"):
            for cls in os.listdir(f"{data_folder}/{split}"):
                if os.path.isdir(f"{data_folder}/{split}/{cls}"):
                    for image in os.listdir(f"{data_folder}/{split}/{cls}"):
                        class2split2file[cls][split].append(
                            f"{data_folder}/{split}/{cls}/{image}")

    for cls in tqdm(class2split2file, desc="Copying files to class-first directory"):
        for split in class2split2file[cls]:

            folder = f"{cls_first_folder}/{os.path.basename(data_folder)}/{cls}/{split}"
            if not os.path.exists(folder):
                os.makedirs(folder)

            for image_file in class2split2file[cls][split]:
                shutil.copy(image_file, f"{folder}/{os.path.basename(image_file)}")

    return f"{cls_first_folder}/{os.path.basename(data_folder)}"

def get_smaller_dataset(dataset, n_cls_tr=10, n_cls_val=5, npc_tr=100,
    npc_val=100, seed=0):
    """Creates a dataset in the same folder as [dataset] that is a subset of
    [dataset], and returns the absolute path to the new dataset.

    Args:
    dataset         -- absolute path to the dataset to create from
    n_classes_tr    -- number of classes in the training split
    n_classes_val   -- number of classes in the val split
    npc_tr          -- number of images per class in the training split, or
                        'all' for all of them
    npc_val         -- number of images per class in the validation split, or
                        'all' for all of them
    seed            -- random seed to use
    """
    def get_images(class_path, n):
        """Returns a list of [n] images sampled randomly from [class_path]."""
        n = len(list(os.listdir(class_path))) if n == "all" else n
        selected_images = random.sample(os.listdir(class_path), n)
        return [f"{class_path}/{s}" for s in selected_images]

    # If [dataset] isn't an absolute path, correct it! Then, make sure it exists
    dataset = f"{data_dir}/{dataset}" if not "/" in dataset else dataset
    if not os.path.exists(dataset):
        raise FileNotFoundError(f"Couldn't find specified dataset {dataset}")

    # Subsample the training and validation images
    random.seed(seed)
    cls_tr = random.sample(os.listdir(f"{dataset}/train"), n_cls_tr)
    images_tr = [get_images(f"{dataset}/train/{c}", npc_tr) for c in cls_tr]
    cls_val = random.sample(os.listdir(f"{dataset}/val"), n_cls_val)
    images_val = [get_images(f"{dataset}/val/{c}", npc_val) for c in cls_val]

    # Name the dataset and make a folder for it
    new_dataset = f"{dataset}-{n_cls_tr}-{npc_tr}-{n_cls_val}-{npc_val}-{seed}"
    if os.path.exists(new_dataset): shutil.rmtree(new_dataset)
    os.makedirs(new_dataset)

    # Copy the image files to the locations for the new dataset
    for cls,images in zip(cls_tr, images_tr):
        cls_path = f"{new_dataset}/train/{cls}"
        os.makedirs(cls_path)
        for image in images:
            shutil.copy(image, f"{cls_path}/{os.path.basename(image)}")
    for cls,images in zip(cls_val, images_val):
        cls_path = f"{new_dataset}/val/{cls}"
        os.makedirs(cls_path)
        for image in images:
            shutil.copy(image, f"{cls_path}/{os.path.basename(image)}")

    return new_dataset
