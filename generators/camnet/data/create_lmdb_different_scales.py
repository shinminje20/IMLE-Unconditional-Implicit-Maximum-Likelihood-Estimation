import sys
import os.path
import glob
import pickle
import lmdb
import cv2
import numpy as np
from PIL import Image
import PIL

import os
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.progress_bar import ProgressBar


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def creat_lmdb_per_category(_img_folder, _lmdb_save_paths, _scales, _map_sizes, width_check=256):
    img_list = sorted(glob.glob(_img_folder))
    images = []
    _count_correct_images = 0
    _count_incorrect_images = 0

    print('Read images I...')
    pbar = ProgressBar(len(img_list))
    for i, v in enumerate(img_list):
        pbar.update('Read {}'.format(v))
        img = Image.open(v)
        if img.size[0] > width_check or img.size[1] > width_check:
            _count_incorrect_images += 1
            continue
        images.append(img)
        _count_correct_images += 1


    if _count_correct_images == 0:
        tqdm.write("Found no 'correct images', so will not create dataset for this split.")
        return 0, 0

    for j, scale in enumerate(_scales):
        down_lmdb_save_path = _lmdb_save_paths[j]
        down_dataset = []
        down_data_size = 0

        print('Read images II...')
        pbar = ProgressBar(len(img_list))
        print("There are ", len(images), "images", _count_correct_images, _count_incorrect_images)

        for i, img in enumerate(images):
            pbar.update('Read {}'.format(i))
            down_img = np.array(
                img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), resample=PIL.Image.BICUBIC)
                , dtype=np.uint8)
            if len(down_img.shape) < 3:
                down_img = cv2.cvtColor(down_img, cv2.COLOR_GRAY2BGR)
            else:
                down_img = cv2.cvtColor(down_img, cv2.COLOR_RGB2BGR)
            down_dataset.append(down_img)
            down_data_size += down_img.nbytes

        _map_sizes[j] += down_data_size
        down_env = lmdb.open(down_lmdb_save_path, map_size=_map_sizes[j] * 1000)  # check the map_size
        print('Finish reading {} images.\nWrite lmdb...'.format(len(img_list)))

        pbar = ProgressBar(len(img_list))
        with down_env.begin(write=True) as txn:  # txn is a Transaction object
            for i, v in enumerate(img_list):
                pbar.update('Write {}'.format(v))
                base_name = os.path.splitext(os.path.basename(v))[0]
                key = base_name.encode('ascii')
                data = down_dataset[i]
                if data.ndim == 2:
                    H, W = data.shape
                    C = 1
                else:
                    H, W, C = data.shape

                meta_key = (base_name + '.meta').encode('ascii')
                meta = '{:d}, {:d}, {:d}'.format(H, W, C)
                txn.put(key, data.astype(np.uint8))
                txn.put(meta_key, meta.encode('ascii'))
        print('Finish writing down_lmdb.')

        # create keys cache
        keys_cache_file = os.path.join(down_lmdb_save_path, '_keys_cache.p')
        down_env = lmdb.open(down_lmdb_save_path, readonly=True, lock=False, readahead=False, meminit=False)
        with down_env.begin(write=False) as txn:
            print('Create lmdb keys cache: {}'.format(keys_cache_file))
            keys = [key.decode('ascii') for key, _ in txn.cursor()]
            pickle.dump(keys, open(keys_cache_file, "wb"))
        print('Finish creating down_lmdb keys cache.')

    return _count_correct_images, _count_incorrect_images


"""
Expected structure:
contains folders related to each category
input_images_root:
    - category_1:
        - test
        - train
        - val
    - category_2:
        - test
        - train
        - val
    ...
    - category_n:
        - test
        - train
        - val
"""
################################################################################
################################################################################
################################################################################
# ISICLE Edits
################################################################################
import argparse
P = argparse.ArgumentParser(description="Preparing data for CaMNet")
P.add_argument("--input_images_root", required=True, type=str,
    help="folder containing images with folder/class/split/image organization")
P.add_argument("--root_save_lmdb", required=True, type=str,
    help="output folder")
P.add_argument("--dataset_name", required=True, type=str,
    help="dataset name")
P.add_argument("--width", required=True, type=int,
    help="dataset name")
P.add_argument("--modes", default=['train', 'val', 'test'], nargs="+",
    help="dataset name")
args = P.parse_args()
print("*** ENSURE that the input dataset contains 256x256 images ***")

root_save_lmdb = f"{args.root_save_lmdb.strip('/')}/"
input_images_root = f"{args.input_images_root.strip('/')}/*"
dataset_name = args.dataset_name
width = args.width
################################################################################

# parameters
scales = [1., 1 / 2., 1 / 4., 1 / 8., 1 / 16.]
dataset_modes = ['train', 'val', 'test']  # support: 'train', 'test', 'val'
# width = 256
# root_save_lmdb = "/path/to/save/lmdbs/"
# input_images_root = '/path/to/root/raw/images/*'
# dataset_name = "Dataset_name"

categories_list = sorted(glob.glob(input_images_root))
print(*categories_list, sep='\n')

for db_mode in dataset_modes:
    print("---------------------Starting---------------------")
    print("Dataset mode is : {}".format(db_mode))

    lmdb_save_paths = []
    map_sizes = []

    for scale in scales:
        lmdb_save_paths.append(root_save_lmdb + dataset_name + f"_{db_mode}_{int(width * scale)}x{int(width * scale)}.lmdb")
        map_sizes.append(0)

    for counter, category_path in enumerate(categories_list):
        print("Start Reading data form the category_{} : {}".format(counter, category_path))
        img_folder = category_path
        if db_mode == "train":
            img_folder += "/train/*"
        elif db_mode == "test":
            img_folder += "/test/*"
        elif db_mode == "val":
            img_folder += "/val/*"

        count_correct_images, count_incorrect_images = creat_lmdb_per_category(
            img_folder,
            lmdb_save_paths,
            scales,
            map_sizes,
            width_check=args.width)

        print("correct images : {}, incorrect images : {}".format(count_correct_images, count_incorrect_images))
        print("################################################################")

print("Finish...")

print("Check lmdb datasets integrity")
for scale in scales:
    for db_mode in dataset_modes:

        lmdb_save_path = root_save_lmdb + dataset_name + f"_{db_mode}_{int(width * scale)}x{int(width * scale)}.lmdb"
        print(lmdb_save_path)
        if os.path.exists(lmdb_save_path):
            lmdb_env = lmdb.open(lmdb_save_path, readonly=True)
            print(lmdb_env.stat())
            print("----------------")
