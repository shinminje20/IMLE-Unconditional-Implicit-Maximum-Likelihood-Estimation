from collections import defaultdict
import os
import shutil
from tqdm import tqdm

data_dir = os.path.dirname(os.path.abspath(__file__))

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
                for image in os.listdir(f"{data_folder}/{split}/{cls}"):
                    image = f"{data_folder}/{split}/{cls}/{image}"
                    class2split2file[cls][split].append(image)

    for cls in tqdm(class2split2file, desc="Copying files to class-first directory"):
        for split in class2split2file[cls]:

            folder = f"{cls_first_folder}/{os.path.basename(data_folder)}/{cls}/{split}"
            if not os.path.exists(folder):
                os.makedirs(folder)

            for image_file in class2split2file[cls][split]:
                shutil.copy(image_file, f"{folder}/{os.path.basename(image_file)}")

    return f"{cls_first_folder}/{os.path.basename(data_folder)}"
