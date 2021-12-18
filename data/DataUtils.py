from collections import defaultdict
import os
from PIL import Image
import random
import shutil
from tqdm import tqdm

data_dir = os.path.dirname(os.path.abspath(__file__))

def get_all_files(path, extensions=[], acc=[]):
    """Returns a list of all files under [path] with an extension in
    [extensions] or simply all the files if [extensions] is empty.
    """
    if os.path.isdir(path):
        for f in os.listdir(path):
            get_all_files(f"{path}/{f}", extensions=extensions, acc=acc)
    else:
        acc.append(path)
    return acc

def resize_dataset(path, new_size):
    """Resizes every image under [path] to be [new_size]."""
    tqdm.write(f"Getting list of all images under {path}")
    files_list = get_all_files(path)
    for f in tqdm(files_list, desc="Resizing images"):
        img = Image.open(f)
        img = img.resize(new_size, Image.BICUBIC)
        img.save(f)

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
