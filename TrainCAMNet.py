"""Script for training a CAMNet model. Mostly, this is just a wrapper over
having to deal with annoying JSONs.
"""
import argparse
from tqdm import tqdm

from utils.NestedNamespace import *
from utils.Utils import *

train_config = {
  "name": None, # Set by this program
  "use_tb_logger": True,
  "model": "CAMNet",
  "scale": 16,
  "task": None, # Set by this program
  "HR_W": 256,
  "HR_H": 256,
  "gpu_ids": [
    0
  ],
  "datasets": {
    "train": {
      "name": "Train_Super_Resolution",
      "mode": "LRHR_four_levels",
      "dataroot_HR": "/path/to/train/HR/data/256x256",
      "dataroot_LR": "/path/to/train/LR/data/16x16",
      "dataroot_D1": "/path/to/train/D1/data/32x32",
      "dataroot_D2": "/path/to/train/D2/data/64x64",
      "dataroot_D3": "/path/to/train/D3/data/128x128",
      "subset_file": None,
      "use_shuffle": True,
      "use_flip": True,
      "use_rot": True,
      "n_workers": 6,
      "batch_size_per_month": 300,
      "batch_size_per_day": 1,
      "iters_per_example": 25
    },
    "val": {
      "name": "Validation_Super_Resolution",
      "mode": "LRHR",
      "dataroot_HR": "/path/to/validation/HR/data/256x256",
      "dataroot_LR": "/path/to/validation/LR/data/16x16"
    }
  },
  "path": {
    "root": "/path/to/CAM-Net",
    "pretrain_model_G": None,
  },
  "network_G": {
    "which_model_G": "CAMNet",
    "num_dense_channels": [
      256,
      192,
      128,
      64
    ],
    "num_residual_channels": [
      128,
      64,
      64,
      64
    ],
    "num_blocks": 6,
    "in_nc": 3,
    "out_nc": 3,
    "code_nc": 5,
    "map_nc": 128,
    "latent_nc": 512,
    "use_noise_encoder": True,
    "no_upsample": False,
  },
  "train": {
    "lr_G": 1e-4,
    "weight_decay_G": 0,
    "beta1_G": 0.9,
    "lr_scheme": "MultiStepLR",
    "lr_steps": [
      85000
    ],
    "lr_gamma": 1.0,
    "use_dci": True,
    "inter_supervision": True,
    "dci_num_comp_indices": 2,
    "dci_num_simp_indices": 10,
    "num_samples_per_img": 120,
    "sample_perturbation_magnitude": 0,
    "zero_code": True,
    "num_months": 20,
    "num_days": 1e4,
    "manual_seed": 0,
    "val_freq": 5e3,
    "pixel_weight": 0,
    "pixel_criterion": "l1"
  },
  "logger": {
    "print_freq": 200,
    "save_checkpoint_freq": 5e3
  }
}

test_config = {
  "name": "test",
  "use_tb_logger": True,
  "model": "CAMNet",
  "scale": 16,
  "HR_W": 256,
  "HR_H": 256,
  "task": "ColorizationSuperResolution",
  "gpu_ids": [
    0
  ],
  "": 10,
  "datasets": {
    "val": {
      "name": "Test",
      "mode": "LRHR",
      "dataroot_HR": f"{project_dir}/generators/camnet/data/camnet_three_val_256.lmdb",
      "dataroot_LR": f"{project_dir}/generators/camnet/data/camnet_three_val_16.lmdb",
    }
  },
  "path": {
    "root": ".",
    "pretrain_model_G": None
  },
  "network_G": {
    "which_model_G": "CAMNet",
    "num_dense_channels": [
      256,
      192,
      128,
      64
    ],
    "num_residual_channels": [
      128,
      64,
      64,
      64
    ],
    "num_blocks": 6,
    "in_nc": 1,
    "out_nc": 2,
    "code_nc": 5,
    "map_nc": 128,
    "latent_nc": 512,
    "use_noise_encoder": True,
    "no_upsample": False
  }
}

def get_camnet_data_names_tr(args):
    """Returns the names of the CAMNet datasets required by [args] as a
    dictionary.
    """
    if args.data == "camnet_three":
        return {
            "train": {
                "dataroot_HR": f"{project_dir}/generators/camnet/data/camnet_three_train_256.lmdb",
                "dataroot_LR": f"{project_dir}/generators/camnet/data/camnet_three_train_16.lmdb",
                "dataroot_D1": f"{project_dir}/generators/camnet/data/camnet_three_train_32.lmdb",
                "dataroot_D2": f"{project_dir}/generators/camnet/data/camnet_three_train_64.lmdb",
                "dataroot_D3": f"{project_dir}/generators/camnet/data/camnet_three_train_128.lmdb",
            },
            "val": {
                "dataroot_HR": f"{project_dir}/generators/camnet/data/camnet_three_val_256.lmdb",
                "dataroot_LR": f"{project_dir}/generators/camnet/data/camnet_three_val_16.lmdb",
            }
        }
    else:
        raise ValueError("Unknown dataset")

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="CAMNet training")
    P.add_argument("--task", choices=["Colorization", "ColorizationSuperResolution"],
        default="ColorizationSuperResolution",
        help="task for training CAMNet")
    P.add_argument("--data", default="camnet_three",
        choices=["camnet_three"],
        help="data to train on")

    P.add_argument("--epochs", default=20, type=int,
        help="number of epochs (months) to train for")
    P.add_argument("--num_days", default=1e4, type=int,
        help="number of days per month. This is the number of iterations per minibatch, and may be larger than --bs / --bs_day, in which case training will loop over each batch multiple times.")
    P.add_argument("--bs", default=400, type=int,
        help="batch size. Across any minibatch, the latent code is constant"),
    P.add_argument("--bs_day", default=1, type=int,
        help="batch size for each iteration")
    P.add_argument("--suffix", default="",
        help="optional training suffix")
    P.add_argument("--options", default=[], nargs="+",
        help="options")
    P.add_argument("--gpu_ids", nargs="+", type=int, default=[0, 1],
        help="GPU IDs")

    P.add_argument("--use_dci", default=1, type=int,
        help="whether or not to use DCI")

    args = NestedNamespace(P.parse_args())
    args.options = sorted([
        f"bs{args.bs}",
        f"bs_day{args.bs_day}",
        f"epochs{args.epochs}",
        f"gpus_{'_'.join([str(x) for x in args.gpu_ids])}",
        f"hierarchical{args.use_dci}"
    ])
    tqdm.write(f"Begin CAMNet training with configuration:\n{str(args)}")

    ############################################################################
    # Create the training config
    ############################################################################
    train_opts = train_config
    train_opts["task"] = args.task
    train_opts["name"] = camnet_folder(args).replace(f"{project_dir}/models/camnet/", "")
    train_opts["gpu_ids"] = args.gpu_ids
    train_opts["path"]["root"] = camnet_folder(args)
    train_opts["datasets"]["train"]["batch_size_per_day"] = args.bs_day
    train_opts["datasets"]["train"]["batch_size_per_month"] = args.bs
    train_opts["train"]["num_months"] = args.epochs
    train_opts["train"]["num_days"] = args.num_days
    train_opts["train"]["use_dci"] = bool(args.use_dci)
    train_opts["datasets"] = NestedNamespace.to_dict(
        NestedNamespace.leaf_union(train_opts["datasets"],
                                   get_camnet_data_names_tr(args)))

    ############################################################################
    # Create the test config
    ############################################################################
    test_opts = train_config
    test_opts["task"] = args.task
    test_opts["name"] = camnet_folder(args).replace(f"{project_dir}/models/camnet/", "")
    test_opts["gpu_ids"] = args.gpu_ids
    test_opts["path"]["root"] = camnet_folder(args)
    test_opts["path"]["model"] = f"{camnet_folder(args)}/models/latest.pth"



    config_save_path = f"{camnet_folder(args)}/train_config.json"
    dict_to_json(train_opts, config_save_path)
    tqdm.write(f"Run the following command to start CAMNet:\npython generators/camnet/train.py -opt {config_save_path}")

    config_save_path = f"{camnet_folder(args)}/test_config.json"
    dict_to_json(test_opts, config_save_path)
