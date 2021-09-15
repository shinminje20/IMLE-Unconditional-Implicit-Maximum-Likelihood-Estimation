import os
import torch
import tqdm

device = "cpu"
def _set_device(args):
    global device
    if args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    elif args.device == "cuda" and not torch.cuda.is_available():
        raise Exception("CUDA not available")
    elif args.device == "cpu":
        device = "cpu"
    else:
        raise ValueError(f"Get device {args.device} but must be one of 'cuda' or 'cpu'")

################################################################################
# I/O Utils
################################################################################
state_sep_str = "=" * 40

def load_generator(file):
    """Returns a (model, optimizer, last_epoch, args, results) tuple from
    [file].
    """
    data = torch.load(file)
    model = data["model"].to(device)
    last_epoch = data["last_epoch"]
    optimizer = data["optimizer"]
    args = data["args"]
    results = data["results"]
    return model, optimizer, last_epoch, args, results

def save_generator(model, optimizer, last_epoch, args, results):
    """Returns the folder to save a generator trained with [args] to."""
    folder = f"Models/generator-{args.data}-{args.suffix}"
    if not os.path.exists(folder): os.makedirs(folder)
    file = f"{folder}/{file}.pt"

    torch.save({
        "model": model.cpu(),
        "optimizer": optimizer,
        "last_epoch": last_epoch,
        "args": args,
        "results": results,
    }, file)

################################################################################
#
################################################################################
save_now = False
plot_now = False
def signal_handler_menu(sig, frame):
    global save_now
    global plot_now

    tqdm.write("======== Script Control ========")
    tqdm.write(f"Current script: {os.path.abspath(__file__)}")

    while True:
        try:
            option = raw_input("(Q)uit, (S)ave, (P)lot or (C)ontinue? ")
        except NameError:   # Python 3
            option = input("(Q)uit, (S)ave, (P)lot or (C)ontinue? ")

        valid_option = False
        if "s" in option.lower():
            save_now = True
            valid_option = True
        if "p" in option.lower():
            plot_now = True
            valid_option = True
        if "c" in option.lower():
            valid_option = True
        if option.lower() == "q":
            raise KeyboardInterrupt
        if valid_option:
            break
        else:
            tqdm.write("Invalid option.")

def _user_confirm():
    while True:
        try:
            option = raw_input("(Y)es or (N)o? ")
        except NameError:
             option = input("(Y)es or (N)o? ")
        if option.lower() == "y":
            break
        elif option.lower() == "n":
            raise KeyboardInterrupt
        else:
            print("Invalid option.")
