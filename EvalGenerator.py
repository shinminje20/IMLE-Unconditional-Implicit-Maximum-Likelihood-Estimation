import argparse
from utils.Utils import *
from TrainGenerator import *

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="CAMNet training")
    P.add_argument("--resume", type=str, default=None,
        help="WandB run to resume from")
    P.add_argument("--data_folder_path", default=f"{project_dir}/data", type=str,
        help="path to data if not in normal place")
    P.add_argument("--gpus", type=int, default=[], nargs="+",
        help="GPU ids")
    args = P.parse_args()

    run_id, resume_data = wandb_load(args.resume)
    model = resume_data["model"].to(device)
    optimizer = resume_data["optimizer"]
    corruptor = resume_data["corruptor"].to(device)
    last_epoch = resume_data["last_epoch"]
    run_args = resume_data["args"]
    run_args.data_folder_path = args.data_folder_path
    run_args.gpus = gpus
    args = run_args

    data_tr, data_eval = get_data_splits(args.data, args.eval, args.res,
        data_path=args.data_folder_path)
    data_tr = GeneratorDataset(data_tr, get_gen_augs())
    data_eval = GeneratorDataset(data_eval, get_gen_augs())

    tqdm.write(f"Color space settings: input {args.in_color_space} | internal {args.color_space} | output {args.out_color_space}")

    show_image_grid(get_images(corruptor, model, data_eval, **vars(args)))
