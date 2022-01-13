"""Script for training a CAMNet model. Mostly, this is just a wrapper over
having to deal with annoying JSONs.
"""
import argparse
from tqdm import tqdm

from utils.NestedNamespace import *
from utils.Utils import *

{
  "name": None,
  "use_tb_logger": true,
  "model": "CAMNet",
  "scale": 16,
  "task": None,
  "gpu_ids": [
    0
  ],
  "multiple": 10,
  "datasets": {
    "val": {
      "name": "Test",
      "mode": "LRHR",
      "dataroot_HR": "/path/to/test/HR/data",
      "dataroot_LR": "/path/to/test/LR/data"
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
    "use_noise_encoder": false,
    "no_upsample": false
  }
}

def get_camnet_data_names(args):
    """Returns the names of the CAMNet datasets required by [args] as a
    dictionary.
    """
    if args.data == "camnet_three":
        return {
            "val": {
                "dataroot_HR": f"{project_dir}/generators/camnet/data/camnet_three_val_256.lmdb",
                "dataroot_LR": f"{project_dir}/generators/camnet/data/camnet_three_val_16.lmdb",
            }
        }
    else:
        raise ValueError("Unknown dataset")

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="CAMNet training")
    P.add_argument("--model", required=True)

    args = NestedNamespace(P.parse_args())
    tqdm.write(f"Begin CAMNet training with configuration:\n{str(args)}")

    default_opts["datasets"] = NestedNamespace.to_dict(
        NestedNamespace.leaf_union(default_opts["datasets"],
                                   get_camnet_data_names(args)))

    print(default_opts)

    config_save_path = f"{camnet_folder(args)}/test_config.json"
    dict_to_json(default_opts, config_save_path)
    tqdm.write(f"Run the following command to start CAMNet:\npython generators/camnet/test.py -opt {config_save_path}")
