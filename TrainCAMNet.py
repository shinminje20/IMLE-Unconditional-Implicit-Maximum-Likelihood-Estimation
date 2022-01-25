"""Script for training a CAMNet model. It interprets command-line arguments to
generate training and test config JSONs, which are saved to the model's folder.

Because CAMNet's code base is written very differently from that of ISICLE, the
command line arguments don't work quite the same way.
"""
import argparse
from tqdm import tqdm
import os

from data.DataUtils import *
from utils.NestedNamespace import *
from utils.Utils import *

def build_data_config_tr(data, bs, mini_bs, iters_per_example, task):
    """Returns the CAMNet training data configuration.

    Args:
    no_res_data_path    -- path to data before resolution
    bs                  -- batch size (batch size in which codes are found)
    mini_bs              -- batch size per data (mini_bs)
    iters_per_example   -- number of times an example is seen per month (epoch)
    """
    if data == "camnet3":
        dataroot_LR = f"{project_dir}/generators/camnet/data/camnet3_train_16x16.lmdb"
        dataroot_D1 = f"{project_dir}/generators/camnet/data/camnet3_train_32x32.lmdb"
        dataroot_D2 = f"{project_dir}/generators/camnet/data/camnet3_train_64x64.lmdb"
        dataroot_D3 = f"{project_dir}/generators/camnet/data/camnet3_train_128x128.lmdb"
        dataroot_HR = f"{project_dir}/generators/camnet/data/camnet3_train_256x256.lmdb"
    if data == "camnet3_deci":
        dataroot_LR = f"{project_dir}/generators/camnet/data/camnet3_deci_train_16x16.lmdb"
        dataroot_D1 = f"{project_dir}/generators/camnet/data/camnet3_deci_train_32x32.lmdb"
        dataroot_D2 = f"{project_dir}/generators/camnet/data/camnet3_deci_train_64x64.lmdb"
        dataroot_D3 = f"{project_dir}/generators/camnet/data/camnet3_deci_train_128x128.lmdb"
        dataroot_HR = f"{project_dir}/generators/camnet/data/camnet3_deci_train_256x256.lmdb"
    if data == "camnet3_centi":
        dataroot_LR = f"{project_dir}/generators/camnet/data/camnet3_centi_train_16x16.lmdb"
        dataroot_D1 = f"{project_dir}/generators/camnet/data/camnet3_centi_train_32x32.lmdb"
        dataroot_D2 = f"{project_dir}/generators/camnet/data/camnet3_centi_train_64x64.lmdb"
        dataroot_D3 = f"{project_dir}/generators/camnet/data/camnet3_centi_train_128x128.lmdb"
        dataroot_HR = f"{project_dir}/generators/camnet/data/camnet3_centi_train_256x256.lmdb"
    else:
        raise ValueError()

    if task == "Colorization":
        mode = "Co_four_levels"
    elif task == "ColorizationSuperResolution":
        mode = "LRHR_four_levels",

    return {
      "name": f"Train_{task}",
      "mode": mode,
      "dataroot_HR": dataroot_HR,
      "dataroot_LR": dataroot_LR,
      "dataroot_D1": dataroot_D1,
      "dataroot_D2": dataroot_D2,
      "dataroot_D3": dataroot_D3,
      "dataroot_HR_Color": dataroot_HR,
      "subset_file": None,
      "use_shuffle": True,
      "use_flip": True,
      "use_rot": True,
      "n_workers": 8,
      "batch_size_per_month": bs,
      "batch_size_per_day": mini_bs,
      "iters_per_example": iters_per_example
    }

def build_data_config_eval(data, task):
    """Returns the CAMNet evaluation data configuration.

    Args:
    no_res_data_path    -- path to data before resolution
    """
    if data == "camnet3":
        dataroot_LR = f"{project_dir}/generators/camnet/data/camnet3_val_16x16.lmdb"
        dataroot_HR = f"{project_dir}/generators/camnet/data/camnet3_val_256x256.lmdb"
    if data == "camnet3_deci":
        dataroot_LR = f"{project_dir}/generators/camnet/data/camnet3_deci_val_16x16.lmdb"
        dataroot_HR = f"{project_dir}/generators/camnet/data/camnet3_deci_val_256x256.lmdb"
    if data == "camnet3_centi":
        dataroot_LR = f"{project_dir}/generators/camnet/data/camnet3_centi_val_16x16.lmdb"
        dataroot_HR = f"{project_dir}/generators/camnet/data/camnet3_centi_val_256x256.lmdb"
    else:
        raise ValueError()

    if task == "Colorization":
        mode = "Co_four_levels"
    elif task == "ColorizationSuperResolution":
        mode = "LRHR",

    return {
      "name": f"Validation_{task}",
      "mode": mode,
      "dataroot_LR": dataroot_LR,
      "dataroot_HR": dataroot_HR,
      "dataroot_HR_Color": dataroot_HR,
    }

train_config = {
  "name": None,
  "use_tb_logger": True,
  "model": "CAMNet",
  "scale": 16,
  "task": None,
  "HR_W": 256,
  "HR_H": 256,
  "gpu_ids": [0],
  "datasets": {"train": None, "val": None},
  "path": {
    "root": "/path/to/CAM-Net",
    "pretrain_model_G": None,
  },
  "network_G": {
    "which_model_G": "CAMNet",
    "num_dense_channels": [256, 192, 128,  64],
    "num_residual_channels": [128, 64, 64, 64],
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
    "lr_steps": [85000],
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
  "gpu_ids": [0],
  "multiple": 1,
  "datasets": {"val": None},
  "path": {
    "root": ".",
    "pretrain_model_G": None
  },
  "network_G": {
    "which_model_G": "CAMNet",
    "num_dense_channels": [256, 192, 128, 64],
    "num_residual_channels": [128, 64, 64, 64],
    "num_blocks": 6,
    "in_nc": 3,
    "out_nc": 3,
    "code_nc": 5,
    "map_nc": 128,
    "latent_nc": 512,
    "use_noise_encoder": True,
    "no_upsample": False
  }
}

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="CAMNet training")
    P.add_argument("--task", choices=["Colorization", "ColorizationSuperResolution"],
        default="ColorizationSuperResolution",
        help="task for training CAMNet")
    P.add_argument("--data", required=True, type=str,
        help="path to folder containing training and validation LMDB data, not including resolution")
    P.add_argument("--epochs", default=20, type=int,
        help="number of epochs (months) to train for")
    P.add_argument("--num_days", default=1e4, type=int,
        help="number of days per month. This is the number of iterations per minibatch, and may be larger than --bs / --mini_bs, in which case training will loop over each batch multiple times.")
    P.add_argument("--bs", default=400, type=int,
        help="batch size. Across any minibatch, the latent code is constant"),
    P.add_argument("--mini_bs", default=1, type=int,
        help="batch size for each iteration")
    P.add_argument("--iters_per_example", default=4, type=int,
        help="iterations per example on each epoch")
    P.add_argument("--suffix", default="",
        help="optional training suffix")
    P.add_argument("--options", default=[], nargs="+",
        help="options")
    P.add_argument("--gpu_ids", nargs="+", type=int, default=[0, 1],
        help="GPU IDs")
    P.add_argument("--code_bs", type=int, default=120)
    args = P.parse_args()

    args.options = sorted([
        f"bs{args.bs}",
        f"code_bs{args.code_bs}",
        f"epochs{args.epochs}",
        f"gpus_{'_'.join([str(x) for x in args.gpu_ids])}",
        f"ipe{args.iters_per_example}",
        f"mini_bs{args.mini_bs}",
    ])

    ############################################################################
    # Create the training config
    ############################################################################
    train_opts = train_config
    train_opts["task"] = args.task
    train_opts["name"] = camnet_folder(args).replace(f"{project_dir}/models/camnet/", "")
    train_opts["scale"] = 16
    train_opts["HR_H"] = 256
    train_opts["HR_W"] = 256
    train_opts["gpu_ids"] = args.gpu_ids
    train_opts["path"]["root"] = camnet_folder(args)
    train_opts["datasets"]["train"] = build_data_config_tr(args.data,
                                                            args.bs,
                                                            args.mini_bs,
                                                            args.iters_per_example,
                                                            args.task)
    train_opts["datasets"]["val"] = build_data_config_eval(args.data, args.task)
    train_opts["train"]["num_months"] = args.epochs
    train_opts["mini_batch_size"] = args.code_bs
    train_opts["network_G"]["in_nc"] = 1 if args.task == "Colorization" else 3


    ############################################################################
    # Create the test config
    ############################################################################
    test_opts = test_config
    test_opts["task"] = args.task
    test_opts["name"] = camnet_folder(args).replace(f"{project_dir}/models/camnet/", "")
    test_opts["gpu_ids"] = args.gpu_ids
    test_opts["path"]["root"] = camnet_folder(args)
    test_opts["path"]["model"] = f"{camnet_folder(args)}/models/latest.pth"
    test_opts["datasets"]["val"] = build_data_config_eval(args.data, args.task)
    test_opts["network_G"]["in_nc"] = 1 if args.task == "Colorization" else 3

    config_save_path = f"{camnet_folder(args)}/train_config.json"
    dict_to_json(train_opts, config_save_path)
    os.system(f"python generators/camnet/train.py -opt {config_save_path}")

    config_save_path = f"{camnet_folder(args)}/test_config.json"
    dict_to_json(test_opts, config_save_path)
